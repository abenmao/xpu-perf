/***************************************************************************************************
 * Sparse Decode Configuration + Kernel Launcher for XPU
 * Follows the structure of PagedDecodeConfig + DecodeKernelLauncher in paged_decode.hpp.
 **************************************************************************************************/
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/device_memory.h"
#include <cute/tensor.hpp>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

// The dense mainloop provides XeDefault and get_sg_layout_pv
#include "collective/chunk_prefill_mainloop.hpp"
#include "collective/chunk_prefill_epilogue.hpp"
#include "collective/sparse_decode_scheduler.hpp"
#include "collective/sparse_mainloop.hpp"
#include "collective/sparse_epilogue.hpp"
#include "kernel/sparse_decode_kernel.hpp"

#include "fmha_utils.hpp"
#include "sparse_attn.hpp"

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////
// Configuration (analogous to PagedDecodeConfig)
// All type assembly happens inside run(), matching the dense code pattern.
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
    typename ElementO = float,
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideO = Stride<int, _1, int, int>>
struct SparseDecodeConfig {

  static void kernel_dispatch(
      sycl::queue& queue,
      const sparse_decode_args_t& args) {

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    // ---- Type assembly (same pattern as PagedDecodeConfig::run) ----
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

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::SparseFwdMainloop<
        MainloopDispatchPolicy,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TILE_K,
        TensorQ,
        void>;  // GmemTiledCopyQ = void -> default

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::SparseFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        TensorLSE,
        void,
        Sink>;

    // Schedulers
    using Scheduler = cutlass::fmha::kernel::SparseDecodeTileScheduler;
    using ReduceScheduler = cutlass::fmha::kernel::SparseReduceSplitKTileScheduler;

    using ProblemShapeType = cutlass::fmha::kernel::SparseDecodeProblemShape;

    // Kernels
    using FMHAKernel = cutlass::fmha::kernel::SparseDecodeFwdKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    using ReduceSplitKernel = cutlass::fmha::kernel::SparseReduceSplitK<
        ProblemShapeType,
        ReduceScheduler,
        FMHAKernel>;

    // ---- Build arguments ----
    using ElementLSE = typename FMHAKernel::ElementLSE;
    using ElementSink = typename FMHAKernel::ElementSink;
    using SGPerWG = typename FMHAKernel::SGPerWG;
    using CollM = typename FMHAKernel::CollectiveMainloop;
    using ElementS = typename CollM::ElementS;

    int num_kv_splits = args.num_splits;
    ProblemShapeType shape;
    shape.batch = args.batch;
    shape.num_heads_q = args.h_q;
    shape.num_heads_kv = args.h_kv;
    shape.seq_len_qo = args.s_q;
    shape.head_size_qk = args.d_qk;
    // Use full head_dim_k for VO dimension (same as dense decode).
    // TileShapeOutput has 576 V-dim; O_accum must match.
    // The reduce kernel writes 576-dim output; Python slices to d_v.
    shape.head_size_vo = args.d_qk;
    shape.topk = args.topk;
    shape.total_tokens = args.total_tokens;

    int head_group_q = shape.num_heads_q / shape.num_heads_kv;

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<ElementQ*>(args.q),
         StrideQ{},
         reinterpret_cast<ElementO*>(args.o_accum),
         StrideO{},
         reinterpret_cast<ElementLSE*>(args.lse_accum),
         StrideO{},
         reinterpret_cast<ElementLSE*>(args.max_logits_accum),
         StrideO{},
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
         static_cast<int>(args.fp8_model),
         args.bytes_per_token,
         args.page_block_size},
        {},
        hw_info,
        num_kv_splits};

    auto params = FMHAKernel::to_underlying_arguments(arguments, nullptr);

    // ---- Launch main kernel ----
    run_kernel<FMHAKernel>(queue, params);

    // ---- Always launch reduce kernel (converts float→bf16 output) ----
    {
      typename ReduceSplitKernel::Arguments reduce_arg{
          {shape,
           reinterpret_cast<ElementO*>(args.out),
           StrideO{},
           reinterpret_cast<const ElementO*>(args.o_accum),
           StrideO{},
           reinterpret_cast<const ElementLSE*>(args.lse_accum),
           StrideO{},
           reinterpret_cast<const ElementLSE*>(args.max_logits_accum),
           StrideO{},
           reinterpret_cast<ElementLSE*>(args.lse)},
          hw_info,
          num_kv_splits};

      auto reduce_params = ReduceSplitKernel::to_underlying_arguments(
          reduce_arg, nullptr);

      run_reduce<FMHAKernel, ReduceSplitKernel>(queue, reduce_params);
    }
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

  template <class FMHAKernel_, class ReduceKernel>
  static void run_reduce(
      sycl::queue& queue,
      typename ReduceKernel::Params reduce_params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = ReduceKernel::get_block_shape();
    dim3 const grid = ReduceKernel::get_grid_shape(reduce_params);
    int smem_size = ReduceKernel::SharedStorageSize;

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
        compat::experimental::launch<cutlass::device_kernel<ReduceKernel>>(
            policy, queue, reduce_params);
    EventManager::getInstance().addEvent(event);
  }
};
