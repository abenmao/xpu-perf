/***************************************************************************************************
 * Sparse Decode Forward Kernel + Reduce Split-K for XPU
 * Follows the structure of XeFMHAFwdSplitKVKernel and ReduceSplitK in paged_decode_kernel.hpp.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include <cute/tensor.hpp>

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
// Problem shape for sparse decode
struct SparseDecodeProblemShape {
  int batch;
  int num_heads_q, num_heads_kv;
  int seq_len_qo;       // typically 1 for decode
  int head_size_qk;     // 576
  int head_size_vo;     // 512 or 576 (for MLA, output is d_v)
  int topk;             // number of sparse indices per query
  int total_tokens;     // total KV tokens in cache
};

///////////////////////////////////////////////////////////////////////////////
// Main sparse decode kernel
template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class SparseDecodeFwdKernel {
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

  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;
  static constexpr bool Sink = CollectiveEpilogue::Sink;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  // Device-side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    ElementO* Oaccum;
    StrideO dOaccum;
    ElementLSE* exp_sums;
    StrideO dExp_sums;
    ElementLSE* max_logits;
    StrideO dMax_logits;
    const ElementSink* sm_sink;
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = 1;
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
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    if (args.num_kv_splits > max_num_kv_splits) return false;
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
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits_;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [blk_q, blk_v, head_kv, idx_b, idx_kv_split] =
          tile_scheduler.get_block_coord();
      auto blk_qv = make_coord(blk_q, blk_v);

      // Compute token range for this split
      int topk = s.topk;
      int actual_topk = topk;
      if (params.mainloop.topk_length) {
        int tkl = params.mainloop.topk_length[idx_b];
        if (tkl < actual_topk) actual_topk = tkl;
      }

      int tiles_per_split = (actual_topk + num_kv_splits - 1) / num_kv_splits;
      int tk_start = idx_kv_split * tiles_per_split;
      int tk_end = cute::min(tk_start + tiles_per_split, actual_topk);

      if (tk_start >= tk_end) continue;

      // Q tensor: [head_group_q, head_size_qk, num_heads_kv, batch]
      auto shape_Q = make_shape(head_group_q, s.head_size_qk, s.num_heads_kv, s.batch);
      auto layout_q = make_ordered_layout(shape_Q, Step<_1, _0, _2, _3>{});
      auto dcQ = const_cast<ElementQ*>(p.Q);
      Tensor Q = make_tensor(make_gmem_ptr(dcQ), layout_q);

      // O tensor: [head_group_q, head_size_vo, num_heads_kv, num_kv_splits, batch]
      auto shape_O = make_shape(head_group_q, s.head_size_vo, s.num_heads_kv, num_kv_splits, s.batch);
      auto layout_o = make_ordered_layout(shape_O, Step<_1, _0, _2, _3, _4>{});
      Tensor O = make_tensor(make_gmem_ptr(p.Oaccum), layout_o);

      // exp_sums / max_logits: [head_group_q, num_kv_splits, num_heads_kv, batch]
      auto shape_lse = make_shape(head_group_q, num_kv_splits, s.num_heads_kv, s.batch);
      auto layout_lse = make_ordered_layout(shape_lse, Step<_1, _0, _2, _3>{});
      Tensor exp_sums = make_tensor(make_gmem_ptr(p.exp_sums), layout_lse);
      Tensor max_logits = make_tensor(make_gmem_ptr(p.max_logits), layout_lse);

      // Sink tensor: [num_heads_kv, head_group_q]
      auto shape_sink = make_shape(s.num_heads_kv, head_group_q);
      auto layout_sink = make_ordered_layout(shape_sink, Step<_1, _0>{});
      Tensor sinks = make_tensor(
          make_gmem_ptr(const_cast<ElementSink*>(p.sm_sink)), layout_sink);

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Main loop
      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);

      mainloop(
          Q(_, _, head_kv, idx_b),
          tArA,
          tA_max,
          tA_sum,
          blk_qv,
          idx_b,
          0,  // idx_sq = 0 for decode
          head_kv,
          head_group_q,
          tk_start,
          tk_end,
          thr_id,
          shared_storage.mainloop);

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      if constexpr (Sink) {
        auto sinks_per_kv = sinks(head_kv, _);
        epilogue(
            O(_, _, head_kv, idx_kv_split, idx_b),
            tArA, tA_max, tA_sum,
            blk_qv, thr_id,
            exp_sums(_, _, head_kv, idx_b),
            max_logits(_, _, head_kv, idx_b),
            idx_kv_split, head_group_q,
            sinks_per_kv, num_kv_splits);
      } else {
        epilogue(
            O(_, _, head_kv, idx_kv_split, idx_b),
            tArA, tA_max, tA_sum,
            blk_qv, thr_id,
            exp_sums(_, _, head_kv, idx_b),
            max_logits(_, _, head_kv, idx_b),
            idx_kv_split, head_group_q,
            sinks, num_kv_splits);
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Reduce Split-K kernel for sparse decode
template <class ProblemShape_, class TileScheduler_, class FMHAKernel_>
class SparseReduceSplitK {
 public:
  using ProblemShape = ProblemShape_;
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  using ElementO = typename FMHAKernel_::ElementO;
  using StrideO = typename FMHAKernel_::StrideO;
  using TileShapeO = typename FMHAKernel_::TileShapeO;
  using TileShapeQK = typename FMHAKernel_::TileShapeQK;
  using ElementLSE = typename FMHAKernel_::ElementLSE;
  using SGPerWG = typename FMHAKernel_::SGPerWG;

  constexpr static int num_vals_per_thread =
      int(get<1>(TileShapeO{}) / (SGPerWG::value * intel::sg_size));

  struct KernelArguments {
    ProblemShape shape;
    ElementO* O;
    StrideO dO;
    const ElementO* Oaccum;
    StrideO dOaccum;
    const ElementLSE* exp_sums;
    StrideO dExp_sums;
    const ElementLSE* max_logits;
    StrideO dMax_logits;
    ElementLSE* lse;  // final lse output
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = 1;
  };

  struct Params {
    KernelParams kernel;
    TileSchedulerParams scheduler;
  };

  struct SharedStorage {
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits> max_logits_slm_array;
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits> exp_sums_slm_array;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static Params to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    if (args.num_kv_splits > FMHAKernel_::max_num_kv_splits) return false;
    return true;
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

    int thr_id = int(ThreadIdxX());
    auto num_kv_splits = params.scheduler.num_kv_splits;

    auto num_heads_q = s.num_heads_q;
    auto head_size_vo = s.head_size_vo;

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [seq_idx, head_q, idx_b] = tile_scheduler.get_block_coord();

      if (seq_idx >= s.seq_len_qo) continue;

      // Compute topk range
      int actual_topk = s.topk;
      int tiles_per_split = (actual_topk + num_kv_splits - 1) / num_kv_splits;

      // Layout: O[seq, head_dim, head_q, batch]
      auto shape_O = make_shape(s.seq_len_qo, head_size_vo, num_heads_q, s.batch);
      auto shape_Oaccum = make_shape(s.seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, s.batch);
      auto shape_lse = make_shape(s.seq_len_qo, num_kv_splits, num_heads_q, s.batch);

      auto stride_O = cutlass::make_cute_packed_stride(StrideO{}, shape_O);
      auto stride_Oaccum = cutlass::make_cute_packed_stride(StrideO{}, shape_Oaccum);
      auto stride_lse = cutlass::make_cute_packed_stride(StrideO{}, shape_lse);

      Tensor Oaccum = make_tensor(
          make_gmem_ptr(const_cast<ElementO*>(p.Oaccum)),
          make_layout(shape_Oaccum, stride_Oaccum));
      Tensor O = make_tensor(
          make_gmem_ptr(p.O),
          make_layout(shape_O, stride_O));
      Tensor exp_sums = make_tensor(
          make_gmem_ptr(const_cast<ElementLSE*>(p.exp_sums)),
          make_layout(shape_lse, stride_lse));
      Tensor max_logits_t = make_tensor(
          make_gmem_ptr(const_cast<ElementLSE*>(p.max_logits)),
          make_layout(shape_lse, stride_lse));

      // Step 1: Load max_logits and exp_sums to SLM
      ElementLSE global_max_logits{
          cutlass::platform::numeric_limits<ElementLSE>::lowest()};
      ElementLSE global_exp_sums{0};

      if (thr_id < num_kv_splits && thr_id * tiles_per_split < actual_topk) {
        ElementLSE cur_max = max_logits_t(seq_idx, thr_id, head_q, idx_b);
        ElementLSE cur_exp = exp_sums(seq_idx, thr_id, head_q, idx_b);
        global_max_logits = sycl::max(global_max_logits, cur_max);
        shared_storage.max_logits_slm_array[thr_id] = cur_max;
        shared_storage.exp_sums_slm_array[thr_id] = cur_exp;
      }

      sycl::group_barrier(get_work_group<3>());

      global_max_logits = reduce_over_group(
          get_work_group<1>(), global_max_logits, sycl::maximum<>());
      global_max_logits =
          sycl::group_broadcast(get_work_group<1>(), global_max_logits, 0);

      // Step 2: Reduce across splits
      for (int idx = thr_id; idx < head_size_vo;
           idx += SGPerWG::value * intel::sg_size) {
        ElementLSE acc = 0;
        global_exp_sums = 0;

        for (int i = 0; i < num_kv_splits; ++i) {
          if (i * tiles_per_split >= actual_topk) break;

          ElementLSE local_max = shared_storage.max_logits_slm_array[i];
          ElementLSE local_exp = shared_storage.exp_sums_slm_array[i];

          // Skip splits with no valid tokens (exp_sums=0, max=-inf)
          if (local_exp == ElementLSE(0)) continue;

          ElementLSE rescale =
              sycl::native::exp2(local_max - global_max_logits);

          // Guard: when attn_sink overflows exp_sums to inf, the FMHA
          // epilogue normalizes O_accum to 0 (via 1/inf=0). Multiplying
          // 0 * inf yields NaN in IEEE 754. Detect and treat as 0.
          ElementLSE o_val = static_cast<ElementLSE>(
              Oaccum(seq_idx, idx, i * num_heads_q + head_q, idx_b));
          ElementLSE adjusted = o_val * local_exp;
          if (adjusted != adjusted) adjusted = ElementLSE(0);  // NaN → 0
          acc += adjusted * rescale;
          global_exp_sums += local_exp * rescale;
        }

        ElementLSE inv_sum = (global_exp_sums == ElementLSE(0))
            ? ElementLSE(0) : ElementLSE(1) / global_exp_sums;
        acc *= inv_sum;
        // Output is bf16 — convert from float accumulator
        // O layout: (seq_len_qo, head_size_vo, num_heads_q, batch)
        // with Step<_1, _0, _2, _3> → head_size_vo contiguous, seq stride = head_size_vo
        auto* out_bf16 = reinterpret_cast<bfloat16_t*>(p.O);
        int64_t flat_offset = ((int64_t)idx_b * num_heads_q + head_q) * head_size_vo + idx;
        out_bf16[flat_offset] = static_cast<bfloat16_t>(acc);
      }

      // Write lse: lse = max_logits / log2(e) + ln(exp_sums)
      // max_logits is in log2 space (scaled by log2(e)), convert back to natural log
      if (thr_id == 0) {
        constexpr double kLn2 = 0.6931471805599453;
        ElementLSE final_max = global_max_logits;
        ElementLSE final_exp = ElementLSE(0);
        for (int i = 0; i < num_kv_splits; ++i) {
          if (i * tiles_per_split >= actual_topk) break;
          ElementLSE local_exp = shared_storage.exp_sums_slm_array[i];
          if (local_exp == ElementLSE(0)) continue;
          ElementLSE local_max = shared_storage.max_logits_slm_array[i];
          ElementLSE rescale = sycl::native::exp2(local_max - final_max);
          final_exp += local_exp * rescale;
        }
        // lse = max_logits * ln(2) + ln(exp_sums)
        ElementLSE lse_val = final_max * static_cast<ElementLSE>(kLn2)
                             + sycl::log(final_exp);
        // lse layout: [batch, num_heads_q, seq_len_qo]
        int64_t lse_offset = ((int64_t)idx_b * num_heads_q + head_q) * s.seq_len_qo + seq_idx;
        p.lse[lse_offset] = lse_val;
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel
