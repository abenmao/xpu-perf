/***************************************************************************************************
 * Sparse Prefill Configuration + Kernel Launcher for XPU
 * No Split-K, no reduce kernel, GQA groups fused into grid.
 * Reuses SparseFwdMainloop and SparseFwdEpilogue.
 **************************************************************************************************/
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/device_memory.h"
#include <cute/tensor.hpp>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "collective/chunk_prefill_mainloop.hpp"
#include "collective/chunk_prefill_epilogue.hpp"
#include "collective/sparse_prefill_scheduler.hpp"
#include "collective/sparse_mainloop.hpp"
#include "collective/sparse_epilogue.hpp"
#include "kernel/sparse_prefill_kernel.hpp"

#include "fmha_utils.hpp"
#include "sparse_attn.hpp"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_ = void,
    int PipelineStages = 1,
    bool Sink = false,
    int TILE_K = 32,
    typename ElementQ = bfloat16_t,
    typename ElementO = bfloat16_t,
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideO = Stride<int, _1, int, int>>
struct SparsePrefillConfig {

  static void kernel_dispatch(
      sycl::queue& queue,
      const sparse_prefill_args_t& args) {

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    // ---- Type assembly (same MMA atoms as SparseDecodeConfig) ----
    static constexpr int SGTileQ =
        get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();

    using MMAOperation = XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>;

    using SubgroupLayoutPV = cute::conditional_t<
        is_void_v<SubgroupLayoutPV_>,
        decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
        SubgroupLayoutPV_>;

    using TiledMMAQK = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapeQK>,
        SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapePV>,
        SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    // Dummy tensors for type derivation
    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(
          make_gmem_ptr(&val),
          make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));
    using TensorLSE = decltype(make_dummy_tensor(float{}, StrideO{}));

    // Mainloop (shared with decode)
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::SparseFwdMainloop<
        MainloopDispatchPolicy,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TILE_K,
        TensorQ,
        void>;

    // Epilogue (shared with decode)
    using CollectiveEpilogue = cutlass::fmha::collective::SparseFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        TensorLSE,
        void,
        Sink>;

    // Prefill scheduler (different from decode)
    using Scheduler = cutlass::fmha::kernel::SparsePrefillTileScheduler;

    using ProblemShapeType = cutlass::fmha::kernel::SparsePrefillProblemShape;

    // Prefill kernel (different from decode — no Split-K)
    using FMHAKernel = cutlass::fmha::kernel::SparsePrefillFwdKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    // ---- Build arguments ----
    using ElementLSE = typename FMHAKernel::ElementLSE;
    using ElementSink = typename FMHAKernel::ElementSink;
    using SGPerWG = typename FMHAKernel::SGPerWG;
    using CollM = typename FMHAKernel::CollectiveMainloop;
    using ElementS = typename CollM::ElementS;

    constexpr int kQPacked = get<0>(TileShapeQK{});
    int head_group_q = kQPacked;

    ProblemShapeType shape;
    shape.s_q = args.s_q;
    shape.num_heads_q = args.h_q;
    shape.num_heads_kv = args.h_kv;
    shape.head_size_qk = args.d_qk;
    shape.head_size_vo = args.d_qk;  // full d_qk; Python slices to d_v
    shape.topk = args.topk;
    shape.total_tokens = args.total_tokens;
    shape.num_groups = args.num_groups;

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<const ElementQ*>(args.q),
         StrideQ{},
         reinterpret_cast<ElementO*>(args.out),
         StrideO{},
         reinterpret_cast<ElementLSE*>(args.lse),
         reinterpret_cast<ElementLSE*>(args.max_logits),
         reinterpret_cast<const ElementSink*>(args.attn_sink)},
        {static_cast<ElementS>(args.sm_scale),
         reinterpret_cast<const int*>(args.indices),
         args.topk,
         reinterpret_cast<const int*>(args.topk_length),
         args.kv,
         args.h_kv,
         args.d_qk,
         args.d_v,
         args.total_tokens,
         0,   // fp8_model = NONE
         0,   // bytes_per_token
         0},  // page_block_size
        {},
        hw_info,
        args.num_groups};

    auto params = FMHAKernel::to_underlying_arguments(arguments, nullptr);

    // ---- Launch single kernel (no reduce needed) ----
    run_kernel<FMHAKernel>(queue, params);
  }

 private:
  template <class Kernel>
  static void run_kernel(
      sycl::queue& queue,
      typename Kernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = Kernel::get_block_shape();
    dim3 const grid = Kernel::get_grid_shape(params);
    int smem_size = Kernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size)};
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>,
        intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};

    auto event =
        compat::experimental::launch<cutlass::device_kernel<Kernel>>(
            policy, queue, params);
    EventManager::getInstance().addEvent(event);
  }
};
