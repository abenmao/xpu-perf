/***************************************************************************************************
 * Dense Prefill Forward Kernel for XPU (Q-token tiled)
 * Each WG handles ONE h_q head across kQTokens consecutive Q tokens.
 * Q tokens tiled into M dimension → high compute intensity (64 FLOP/byte).
 * No GQA head packing: each WG processes one head independently.
 * Variable-length sequences via cu_seqlens_q / cu_seqlens_k.
 * Direct bf16 output via non-SplitK epilogue path.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include <cute/tensor.hpp>

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Problem shape for dense prefill
struct DensePrefillProblemShape {
  int total_q;
  int num_heads_q;      // full h_q (e.g. 128)
  int num_heads_kv;     // typically 1
  int head_size_qk;     // 512 or 576
  int head_size_vo;     // 512 or 576
  int total_kv;
  int num_groups;       // h_q / h_kv / kQPacked (kept for API compat)
  int batch_size;
};

///////////////////////////////////////////////////////////////////////////////
// Main dense prefill kernel (Q-token tiled, no Split-K)
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class DensePrefillFwdKernel {
 public:
  using ProblemShape = ProblemShape_;
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = ElementQ;

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;

  using StrideQ = Stride<int, _1>;    // 2D: (row_stride, 1) for per-head Q
  using StrideO = Stride<int, _1>;    // 2D: (row_stride, 1) for per-head O

  // M = number of Q tokens per WG tile
  static constexpr int kQTokens = get<0>(TileShapeQK{});

  // Shared storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static constexpr bool Sink = CollectiveEpilogue::Sink;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  // Device-side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;          // [total_q, h_q, d_qk] contiguous
    StrideQ dQ;
    ElementO* Out;              // [total_q, h_q, d_qk] bf16
    StrideO dOut;
    ElementLSE* lse;            // [total_q, h_q] float32
    ElementLSE* max_logits;     // [total_q, h_q] float32
    const ElementSink* sm_sink; // always nullptr for dense prefill
    const void* kv_ptr;         // [total_kv, h_kv, d_qk] bf16 KV cache
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_groups = 1;
    const int* cu_seqlens_q = nullptr;
    const int* cu_seqlens_k = nullptr;
    int batch_size = 0;
  };

  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //
  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_groups,
            args.cu_seqlens_q, args.cu_seqlens_k, args.batch_size)};
  }

  static bool can_implement(Arguments const& args) {
    return CollectiveMainloop::can_implement(args.mainloop) &&
           CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const&) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const&, void* = nullptr, cudaStream_t = nullptr,
      CudaHostAdapter* = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;

    int d_qk = s.head_size_qk;
    int d_vo = s.head_size_vo;
    int h_q  = s.num_heads_q;
    int h_kv = s.num_heads_kv;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto coord = tile_scheduler.get_block_coord();
      int blk_v          = get<0>(coord);
      int head_q         = get<1>(coord);
      int head_kv        = get<2>(coord);
      int q_abs_start    = get<3>(coord);
      int actual_q_count = get<4>(coord);
      int batch_q_start  = get<5>(coord);
      int k_start        = get<6>(coord);
      int k_end          = get<7>(coord);

      // Skip empty sequences or out-of-range tiles
      if (k_end <= k_start || actual_q_count <= 0) continue;

      int seqlen_k = k_end - k_start;

      // Q tensor: per-head 2D view (actual_q_count, d_qk) with row stride h_q*d_qk
      // This selects Q[:, head_q, :] starting from q_abs_start
      auto Q_ptr = const_cast<ElementQ*>(p.Q)
                   + static_cast<long>(q_abs_start) * h_q * d_qk
                   + static_cast<long>(head_q) * d_qk;
      auto Q_2D = make_tensor(
          make_gmem_ptr(Q_ptr),
          make_layout(make_shape(actual_q_count, d_qk),
                      make_stride(int(h_q * d_qk), _1{})));

      // O tensor: per-head 2D view (actual_q_count, d_vo)
      auto O_ptr = p.Out
                   + static_cast<long>(q_abs_start) * h_q * d_vo
                   + static_cast<long>(head_q) * d_vo;
      auto O_2D = make_tensor(
          make_gmem_ptr(O_ptr),
          make_layout(make_shape(actual_q_count, d_vo),
                      make_stride(int(h_q * d_vo), _1{})));

      // K/V tensors: both are views of the same KV cache memory
      // KV cache layout: [total_kv, h_kv, d_qk] contiguous
      int kv_stride = h_kv * d_qk;
      auto kv_base = const_cast<ElementK*>(
          reinterpret_cast<const ElementK*>(p.kv_ptr))
          + static_cast<long>(k_start) * kv_stride
          + head_kv * d_qk;

      // K: (seqlen_k, d_qk) row-major — for QK^T GEMM
      auto K_2D = make_tensor(
          make_gmem_ptr(kv_base),
          make_layout(make_shape(seqlen_k, d_qk),
                      make_stride(int(kv_stride), _1{})));

      // V: (d_qk, seqlen_k) transposed — for PV GEMM (same base pointer)
      auto V_2D = make_tensor(
          make_gmem_ptr(kv_base),
          make_layout(make_shape(d_qk, seqlen_k),
                      make_stride(_1{}, int(kv_stride))));

      // Accumulators
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Compute block range and causal info
      constexpr int k_tile = get<1>(TileShapeQK{});
      int blk_k0 = 0;
      int blk_k1 = cute::ceil_div(seqlen_k, k_tile);

      // Q positions within batch (for causal masking)
      int first_q_pos = q_abs_start - batch_q_start;

      // Main loop — process full KV range
      auto blk_qv = make_coord(0, blk_v);
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      mainloop(
          Q_2D,
          K_2D,
          V_2D,
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          first_q_pos,
          actual_q_count,
          blk_k0,
          blk_k1,
          thr_id,
          seqlen_k);

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue — normalize and write O (non-SplitK path)
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};

      epilogue(
          O_2D,
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          thr_id);

      // Write LSE (as exp_sums) and max_logits (in log2 scale) per Q token.
      // flash_api.cpp converts: lse = max_logits*ln2 + ln(exp_sums), max_logits *= ln2
      // After reduce<1>, each lane holds a different row's value in element 0:
      // lane k holds the value for local row k (lanes 0..rows_per_sg-1 are valid).
      int sg_id = thr_id / intel::sg_size;
      int lane_id = thr_id % intel::sg_size;
      constexpr int rows_per_sg = kQTokens / int(SGPerWG::value);
      int row = sg_id * rows_per_sg + lane_id;
      if (lane_id < rows_per_sg && row < actual_q_count) {
        int q_abs = q_abs_start + row;
        p.lse[q_abs * h_q + head_q] = tA_sum(0);
        p.max_logits[q_abs * h_q + head_q] = tA_max(0);
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel
