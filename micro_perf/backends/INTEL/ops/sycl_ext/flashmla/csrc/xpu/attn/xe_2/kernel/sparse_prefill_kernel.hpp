/***************************************************************************************************
 * Sparse Prefill Forward Kernel for XPU
 * No Split-K: each WG processes one (sq, group, hkv) position's full topk.
 * Direct bf16 output via non-SplitK epilogue path.
 * GQA groups fused into grid — single kernel launch.
 *
 * Reuses SparseFwdMainloop (TiledMMA QK/PV, SLM gather) and
 * SparseFwdEpilogue (non-SplitK output write).
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include <cute/tensor.hpp>

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Problem shape for sparse prefill
struct SparsePrefillProblemShape {
  int s_q;
  int num_heads_q;      // full h_q (e.g. 128)
  int num_heads_kv;     // typically 1
  int head_size_qk;     // 576
  int head_size_vo;     // 512 or 576
  int topk;
  int total_tokens;
  int num_groups;       // h_q / h_kv / kQPacked
};

///////////////////////////////////////////////////////////////////////////////
// Main sparse prefill kernel (no Split-K)
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class SparsePrefillFwdKernel {
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

  using StrideQ = Stride<int, _1, int, int>;
  using StrideO = Stride<int, _1, int, int>;

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
    const ElementQ* Q;          // [s_q, h_q, d_qk] contiguous
    StrideQ dQ;
    ElementO* Out;              // [s_q, h_q, d_v] bf16 or float (matches ElementO)
    StrideO dOut;
    ElementLSE* lse;            // [s_q, h_q] float32
    ElementLSE* max_logits;     // [s_q, h_q] float32
    const ElementSink* sm_sink; // [h_q] float32 or nullptr
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_groups = 1;
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
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_groups)};
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

    constexpr int kQPacked = get<0>(TileShapeQK{});  // 8
    int head_group_q = kQPacked;
    int d_qk = s.head_size_qk;
    int d_vo = s.head_size_vo;
    int h_q  = s.num_heads_q;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_kv, idx_sq, group_idx] =
          tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);

      // Compute topk range (no Split-K: full range)
      int actual_topk = s.topk;
      if (params.mainloop.topk_length) {
        int tkl = params.mainloop.topk_length[idx_sq];
        if (tkl < actual_topk) actual_topk = tkl;
      }
      if (actual_topk <= 0) continue;

      // Head offset within h_q for this GQA group
      int h_offset = group_idx * kQPacked;

      // Q tensor: shape [hg, d_qk, hkv, sq] with explicit strides.
      // Physical: Q[sq, hq, d] → offset = sq*h_q*d_qk + hq*d_qk + d
      // Base pointer shifted by h_offset*d_qk for this group.
      // Stride types: (int, _1, int, int) to match TensorQ type from StrideQ.
      auto Q_group = const_cast<ElementQ*>(p.Q)
                     + static_cast<long>(h_offset) * d_qk;
      auto shape_Q = make_shape(int(head_group_q), int(d_qk),
                                int(s.num_heads_kv), int(s.s_q));
      auto stride_Q = make_stride(int(d_qk), _1{},
                                  int(head_group_q * d_qk), int(h_q * d_qk));
      Tensor Q = make_tensor(make_gmem_ptr(Q_group),
                             make_layout(shape_Q, stride_Q));

      // O tensor: same stride pattern.
      auto O_group = p.Out + static_cast<long>(h_offset) * d_vo;
      auto shape_O = make_shape(int(head_group_q), int(d_vo),
                                int(s.num_heads_kv), int(s.s_q));
      auto stride_O = make_stride(int(d_vo), _1{},
                                  int(head_group_q * d_vo), int(h_q * d_vo));
      Tensor O = make_tensor(make_gmem_ptr(O_group),
                             make_layout(shape_O, stride_O));

      // Accumulators
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Main loop — process full topk range
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      mainloop(
          Q(_, _, head_kv, idx_sq),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          idx_sq,       // idx_b = idx_sq (indices are [s_q, topk])
          0,            // idx_sq within batch = 0 (each sq is one "batch")
          head_kv,
          head_group_q,
          0,            // tk_start = 0 (full range)
          actual_topk,  // tk_end = full topk
          thr_id,
          shared_storage.mainloop);

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue — use SplitK path with num_kv_splits=1 to write exp_sums/max_logits.
      // Sink is never applied in kernel (host-side post-processing).
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};

      // exp_sums/max_logits: shape [hg, splits=1, hkv, sq] with explicit strides.
      // Physical: lse[sq, hq] → offset = sq*h_q + hq
      // Stride types: (int, _1, int, int) to match TensorLSE type.
      auto shape_lse = make_shape(int(head_group_q), int(1),
                                  int(s.num_heads_kv), int(s.s_q));
      auto stride_lse = make_stride(int(1), _1{},
                                    int(head_group_q), int(h_q));
      auto lse_group = p.lse + h_offset;
      auto ml_group = p.max_logits + h_offset;
      Tensor exp_sums = make_tensor(make_gmem_ptr(lse_group),
                                    make_layout(shape_lse, stride_lse));
      Tensor max_logits_t = make_tensor(make_gmem_ptr(ml_group),
                                        make_layout(shape_lse, stride_lse));

      // Dummy sink tensor (Sink template is always false for prefill)
      auto shape_sink = make_shape(s.num_heads_kv, head_group_q);
      auto layout_sink = make_ordered_layout(shape_sink, Step<_1, _0>{});
      Tensor sinks = make_tensor(
          make_gmem_ptr(static_cast<ElementSink*>(nullptr)), layout_sink);

      epilogue(
          O(_, _, head_kv, idx_sq),
          tArA, tA_max, tA_sum,
          blk_qv, thr_id,
          exp_sums(_, _, head_kv, idx_sq),
          max_logits_t(_, _, head_kv, idx_sq),
          0,               // idx_kv_split = 0
          head_group_q,
          sinks,
          1);              // num_kv_splits = 1
    }
  }
};

}  // namespace cutlass::fmha::kernel
