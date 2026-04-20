/***************************************************************************************************
 * Dense Prefill Configuration + Kernel Launcher for XPU
 * No Split-K, no reduce kernel, GQA groups fused into grid.
 * Reuses DensePrefillFwdMainloop and SparseFwdEpilogue.
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
#include "collective/dense_prefill_scheduler.hpp"
#include "collective/dense_prefill_mainloop.hpp"
#include "collective/sparse_epilogue.hpp"
#include "kernel/dense_prefill_kernel.hpp"

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
    typename StrideQ = Stride<int, _1>,
    typename StrideO = Stride<int, _1>>
struct DensePrefillConfig {

  static void kernel_dispatch(
      sycl::queue& queue,
      const dense_prefill_args_t& args) {

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    // ---- Type assembly (same MMA atoms as SparsePrefillConfig) ----
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

    // K and V are 2D views of the same KV cache memory
    using StrideK = Stride<int, _1>;       // (token_stride, _1)
    using StrideV = Stride<_1, int>;       // (_1, token_stride)
    using TensorK = decltype(make_dummy_tensor(ElementQ{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementQ{}, StrideV{}));

    // Mainloop (dense prefill with TiledCopy 2D block loads for K/V)
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::DensePrefillFwdMainloop<
        MainloopDispatchPolicy,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TILE_K,
        TensorQ,
        TensorK,
        TensorV,
        void,    // TiledCopyQ (auto-derive)
        void,    // TiledCopyK (auto-derive)
        void>;   // TiledCopyV (auto-derive)

    // Epilogue (reused from sparse)
    using CollectiveEpilogue = cutlass::fmha::collective::SparseFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        TensorLSE,
        void,
        Sink>;

    // Dense prefill scheduler
    using Scheduler = cutlass::fmha::kernel::DensePrefillTileScheduler;

    using ProblemShapeType = cutlass::fmha::kernel::DensePrefillProblemShape;

    // Dense prefill kernel
    using FMHAKernel = cutlass::fmha::kernel::DensePrefillFwdKernel<
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

    ProblemShapeType shape;
    shape.total_q = args.total_q;
    shape.num_heads_q = args.h_q;
    shape.num_heads_kv = args.h_kv;
    shape.head_size_qk = args.d_qk;
    shape.head_size_vo = args.d_qk;  // full d_qk; Python slices to d_v
    shape.total_kv = args.total_kv;
    shape.num_groups = args.num_groups;
    shape.batch_size = args.batch_size;

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<const ElementQ*>(args.q),
         StrideQ{},
         reinterpret_cast<ElementO*>(args.out),
         StrideO{},
         reinterpret_cast<ElementLSE*>(args.lse),
         reinterpret_cast<ElementLSE*>(args.max_logits),
         nullptr,  // no sm_sink for dense prefill
         args.kv},  // KV cache pointer
        {static_cast<ElementS>(args.sm_scale),
         args.is_causal},
        {},
        hw_info,
        args.num_groups,
        reinterpret_cast<const int*>(args.cu_seqlens_q),
        reinterpret_cast<const int*>(args.cu_seqlens_k),
        args.batch_size};

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
