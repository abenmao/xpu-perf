/***************************************************************************************************
 * Copyright (C) 2025 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief CUTLASS Intel BMG Group Gemm

    This file is almost a complete copy of 04_bmg_grouped_gemm,
    except that it's used for FP8 (E5M2 & E4M3) datatype inputs.

    This example demonstrates fusing multiple GEMM operations into one kernel.

    Note that the scalar arguments to e.g. the standard 00_bmg_gemm example, have been
    replaced with vector equivalents, as each individual GEMM has its own inputs and outputs, which
    needn't be contiguous in memory. For example, where 00_bmg_gemm receives an `ElementA *`
    defining Matrix A, grouped gemm receives a `ElementA **`, i.e. a pointer to pointers, each
    pointing to a distinct Matrix A. Likewise, each individual GEMM operation may have its own alpha
    and beta factors for linear combination. This example demonstrates two approaches: the user can
    provide `options.alpha` and `options.beta`, in which case they will apply to all GEMMs;
    otherwise, random values are generated per GEMM.

    Group GEMM scheduling (cutlass::gemm::GroupScheduler) is more complex than standard GEMM,
    because each GEMM may have a unique size, only known at runtime. Thus, the scheduler will
    distribute an a priori unknown number of tiles to each work-group. See
    include/cutlass/gemm/kernel/xe_gemm_array_cooperative.hpp for implementation.

    Note that for simplicity, this example sets every GEMM in the group to the same shape.

    Verification for this example is a conventional GEMM kernel, executed iteratively per group.

    To build & run this example (from your build dir):

      $ ninja 09_bmg_grouped_gemm_fp8
      $ ./examples/sycl/09_bmg_grouped_gemm_fp8/09_bmg_grouped_gemm_fp8

    Call with `--help` for information about available options.

    Note: the code may spill registers once compiled which will result in sub-optimal performance. This is because
    of an issue inside Intel Graphics Compiler (IGC) related to VectorAliasBBThreshold being debugged internally.
    To avoid register spills, build the example by setting the environment variable:
      $ export IGC_VectorAliasBBThreshold=10000
*/
#include <pybind11/pybind11.h>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_array_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "include/sycl_common.hpp"
#include "include/helper.h"

#include <cfloat>
#include <stdexcept>
#include <string>

using namespace cute;
using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>; // <M,N,K> per group

using ElementAccumulator = float;     // <- data type of accumulator
using ElementComputeEpilogue = float; // <- data type of epilogue operations
using ElementOutput = half_t;         // <- data type of elements in output matrix D

///////////////////////////////////////////////////////////////////////////////////////////////////


// Command line options parsing
struct Options {

  float alpha, beta;
  int iterations, verify;

  // Aligned with xpu-perf int8 moe_quant_group_gemm interface.
  int num_tokens, hidden_size, new_hidden_size;
  int num_experts, topk, ep_size, ep_rank;
  int quant_group_size = -1;
  std::string dtype, w_dtype, compute_dtype, dst_dtype;

  // Internal grouped-gemm shape consumed by this fp8 example.
  int m, n, k, groups;
  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;

  Options() : alpha(FLT_MAX), beta(FLT_MAX), iterations(100), verify(1),
              num_tokens(48), hidden_size(8192), new_hidden_size(8192),
              num_experts(128), topk(8), ep_size(8), ep_rank(0), quant_group_size(-1),
              dtype("float8"), w_dtype("float8"), compute_dtype("float8"), dst_dtype("float16"),
              m(num_tokens), n(new_hidden_size), k(hidden_size), groups(num_experts / ep_size) {
    rebuild_problem_sizes_host_from_dispatch();
  }

  std::vector<int> compute_expert_dispatch_token_count() const {
    int num_experts_per_rank = num_experts / ep_size;
    int experts_start_idx = ep_rank * num_experts_per_rank;
    int experts_end_idx = experts_start_idx + num_experts_per_rank;

    std::vector<std::vector<int>> experts_idx_for_each_rank;
    experts_idx_for_each_rank.reserve(ep_size);
    for (int rank_idx = 0; rank_idx < ep_size; ++rank_idx) {
      int start_idx = rank_idx * num_experts_per_rank;
      int end_idx = start_idx + num_experts_per_rank;
      std::vector<int> experts;
      experts.reserve(num_experts_per_rank);
      for (int expert_idx = start_idx; expert_idx < end_idx; ++expert_idx) {
        experts.push_back(expert_idx);
      }
      experts_idx_for_each_rank.push_back(experts);
    }

    std::vector<int> experts_array;
    experts_array.reserve(num_experts);
    for (int local_idx = 0; local_idx < num_experts_per_rank; ++local_idx) {
      for (int rank_idx = 0; rank_idx < ep_size; ++rank_idx) {
        experts_array.push_back(experts_idx_for_each_rank[rank_idx][local_idx]);
      }
    }

    std::vector<int> expert_dispatch_token_count(num_experts_per_rank, 0);
    int cur_expert = 0;
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
      for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
        int expert_idx = experts_array[cur_expert];
        if (expert_idx >= experts_start_idx && expert_idx < experts_end_idx) {
          expert_dispatch_token_count[expert_idx - experts_start_idx] += 1;
        }
        cur_expert += 1;
        if (cur_expert >= num_experts) {
          cur_expert = 0;
        }
      }
    }
    return expert_dispatch_token_count;
  }

  void rebuild_problem_sizes_host_from_dispatch() {
    std::vector<int> expert_dispatch_token_count = compute_expert_dispatch_token_count();
    groups = static_cast<int>(expert_dispatch_token_count.size());

    problem_sizes_host.clear();
    problem_sizes_host.reserve(groups);
    for (int m_group : expert_dispatch_token_count) {
      problem_sizes_host.push_back({m_group, n, k});
    }
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s, std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host) const
  {
    // Number of real-valued multiply-adds
    uint64_t fmas = uint64_t();

    for (auto const & problem : problem_sizes_host) {
      fmas += static_cast<uint64_t>(get<0>(problem)) *
              static_cast<uint64_t>(get<1>(problem)) *
              static_cast<uint64_t>(get<2>(problem));
    }
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * uint64_t(fmas);
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class Gemm
>
struct ExampleRunner {

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  // Host-side allocations
  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  std::vector<ElementAccumulator> alpha_host;
  std::vector<ElementAccumulator> beta_host;

  // Device-side allocations
  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;

  // This example defines all matrices in a single allocation (e.g. block_A), but this is not a
  // requirement. Matrix base pointers are read from device allocation (e.g. ptr_A)
  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  cutlass::DeviceAllocation<const ElementA *> ptr_A;
  cutlass::DeviceAllocation<const ElementB *> ptr_B;
  cutlass::DeviceAllocation<const ElementC *> ptr_C;
  cutlass::DeviceAllocation<ElementOutput *> ptr_D;
  cutlass::DeviceAllocation<ElementOutput *> ptr_ref_D;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  // Note, this is an array of pointers to alpha and beta scaling values per group
  cutlass::DeviceAllocation<ElementAccumulator*> alpha_device;
  cutlass::DeviceAllocation<ElementAccumulator*> beta_device;
  cutlass::DeviceAllocation<ElementAccumulator> block_alpha;
  cutlass::DeviceAllocation<ElementAccumulator> block_beta;

  uint64_t seed = 0;

  //
  // Methods
  //
  template<typename ElementType>
  bool verify(const Options &options) {
    bool passed = true;
    // Verify against individual reference GEMMs
    for (int32_t i = 0; i < options.groups; ++i) {
      auto problem = options.problem_sizes_host.at(i);
      auto M = get<0>(problem);
      auto N = get<1>(problem);
      auto K = get<2>(problem);

      cutlass::DeviceAllocation<half_t> block_A_fp16(block_A.size());
      cutlass::DeviceAllocation<half_t> block_B_fp16(block_B.size());

      // fp8 -> fp16
      convert_dtype<ElementType, half_t, ExampleRunner>(
          block_A.get(),
          block_A_fp16.get(),
          block_A.size()
      );
      convert_dtype<ElementType, half_t, ExampleRunner>(
          block_B.get(),
          block_B_fp16.get(),
          block_B.size()
      );

      cutlass::TensorRef ref_A(block_A_fp16.get() + offset_A.at(i), LayoutA::packed({M, K}));
      cutlass::TensorRef ref_B(block_B_fp16.get() + offset_B.at(i), LayoutB::packed({K, N}));
      cutlass::TensorRef ref_C(block_C.get() + offset_C.at(i), LayoutC::packed({M, N}));
      cutlass::TensorRef ref_D(block_ref_D.get() + offset_D.at(i), LayoutD::packed({M, N}));

      //
      // Compute reference output
      //
      cutlass::reference::device::GemmComplex(
            {M, N, K},
            alpha_host.at(i),
            ref_A,
            cutlass::ComplexTransform::kNone,
            ref_B,
            cutlass::ComplexTransform::kNone,
            beta_host.at(i),
            ref_C,
            ref_D,
            ElementAccumulator(0),
            1,     // batch_count
            M * K, // batch_stride_A
            K * N, // batch_stride_B
            M * N, // batch_stride_C
            M * N  // batch_stride_D
          );

      // Wait for kernel to finish
      compat::wait();

      // Check if output from CUTLASS kernel and reference kernel are equal or not
      passed &= cutlass::reference::device::BlockCompareEqual(block_ref_D.get() + offset_D.at(i), block_D.get() + offset_D.at(i), M * N);
      if(!passed)
        break;
    }
    return passed;
  }

/// Allocates device-side data
void allocate(const Options &options) {
  int64_t total_elements_A = 0;
  int64_t total_elements_B = 0;
  int64_t total_elements_C = 0;
  int64_t total_elements_D = 0;

  // Compute total allocation sizes across group
  for (int32_t i = 0; i < options.groups; ++i) {

    auto problem = options.problem_sizes_host.at(i);
    auto M = get<0>(problem);
    auto N = get<1>(problem);
    auto K = get<2>(problem);

    // Offset into block allocation of each matrix base pointer
    offset_A.push_back(total_elements_A);
    offset_B.push_back(total_elements_B);
    offset_C.push_back(total_elements_C);
    offset_D.push_back(total_elements_D);

    int64_t elements_A = M * K;
    int64_t elements_B = K * N;
    int64_t elements_C = M * N;
    int64_t elements_D = M * N;

    total_elements_A += elements_A;
    total_elements_B += elements_B;
    total_elements_C += elements_C;
    total_elements_D += elements_D;

    stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1}));
    stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1}));
    stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1}));
    stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1}));

  }

  block_A.reset(total_elements_A);
  block_B.reset(total_elements_B);
  block_C.reset(total_elements_C);
  block_D.reset(total_elements_D);
  block_ref_D.reset(total_elements_D);
  block_alpha.reset(options.groups);
  block_beta.reset(options.groups);
}

/// Initialize operands to be used in the GEMM and reference GEMM
template<typename ElementType>
void initialize(const Options &options) {

  uint64_t seed = 2020;

  problem_sizes.reset(options.groups);
  problem_sizes.copy_from_host(options.problem_sizes_host.data());

  //
  // Assign pointers
  //

  std::vector<ElementType *> ptr_A_host(options.groups);
  std::vector<ElementType *> ptr_B_host(options.groups);
  std::vector<ElementC *> ptr_C_host(options.groups);
  std::vector<ElementOutput *> ptr_D_host(options.groups);
  std::vector<ElementAccumulator *> ptr_alpha_host(options.groups);
  std::vector<ElementAccumulator *> ptr_beta_host(options.groups);

  // Compute offsets, alpha & beta over group on host
  for (int32_t i = 0; i < options.groups; ++i) {
    ptr_A_host.at(i) = block_A.get() + offset_A.at(i);
    ptr_B_host.at(i) = block_B.get() + offset_B.at(i);
    ptr_C_host.at(i) = block_C.get() + offset_C.at(i);
    ptr_D_host.at(i) = block_D.get() + offset_D.at(i);
    // Fill host vector of alpha & beta with random values if using per-group values
    alpha_host.push_back((options.alpha == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5 + 1) : options.alpha);
    beta_host.push_back((options.beta == FLT_MAX) ? static_cast<ElementAccumulator>(rand() % 5) : options.beta);
    // Fill host ptr vectors with offset addresses into device alpha/beta blocks
    ptr_alpha_host.at(i) = block_alpha.get() + i;
    ptr_beta_host.at(i) = block_beta.get() + i;
  }

  // Allocate device memory & copy from host
  ptr_A.reset(options.groups);
  // Per-group alpha and beta
  ptr_A.copy_from_host(ptr_A_host.data());

  ptr_B.reset(options.groups);
  ptr_B.copy_from_host(ptr_B_host.data());

  ptr_C.reset(options.groups);
  ptr_C.copy_from_host(ptr_C_host.data());

  ptr_D.reset(options.groups);
  ptr_D.copy_from_host(ptr_D_host.data());

  stride_A.reset(options.groups);
  stride_A.copy_from_host(stride_A_host.data());

  stride_B.reset(options.groups);
  stride_B.copy_from_host(stride_B_host.data());

  stride_C.reset(options.groups);
  stride_C.copy_from_host(stride_C_host.data());

  stride_D.reset(options.groups);
  stride_D.copy_from_host(stride_D_host.data());

  // Per-group alpha and beta ptrs
  alpha_device.reset(options.groups);
  alpha_device.copy_from_host(ptr_alpha_host.data());
  beta_device.reset(options.groups);
  beta_device.copy_from_host(ptr_beta_host.data());

  initialize_block(block_A, seed + 2023);
  initialize_block(block_B, seed + 2022);
  initialize_block(block_C, seed + 2021);
  // Per-group alpha and beta values - note these are not directly passed to kernel - the pointers
  // (alpha_device/beta_device) are passed instead
  block_alpha.copy_from_host(alpha_host.data());
  block_beta.copy_from_host(beta_host.data());
}

  /// Populates a Gemm::Arguments structure from the given commandline options
  typename Gemm::Arguments args_from_options(const Options &options, const cutlass::KernelHardwareInfo& hw_info, bool host_problem_shapes_available = true)
  {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;

    if (options.alpha != FLT_MAX && options.beta != FLT_MAX) {
      // If both alpha/beta are provided (via cmd line args) and are scalar, i.e., same alpha/beta applies to all batches.
      fusion_args.alpha = options.alpha;
      fusion_args.beta = options.beta;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = nullptr;
      fusion_args.beta_ptr_array = nullptr;
      // Single alpha and beta for all groups
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};
    }
    else {
      // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups.
      fusion_args.alpha = 0;
      fusion_args.beta = 0;
      fusion_args.alpha_ptr = nullptr;
      fusion_args.beta_ptr = nullptr;
      fusion_args.alpha_ptr_array = alpha_device.get();
      fusion_args.beta_ptr_array = beta_device.get();
      // One alpha and beta per each group
      fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
      fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};
    }
    using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    // Per-GEMM problem shape info may only exist on the device.
    if (host_problem_shapes_available) {
      arguments = typename Gemm::Arguments {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {options.groups, problem_sizes.get(), options.problem_sizes_host.data()},
        {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
        {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}
      };
    }
    else {
      arguments = typename Gemm::Arguments {
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {options.groups, problem_sizes.get(), nullptr},
        {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
        {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}
      };
    }

    return arguments;
  }

  template<typename ElementType>
  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info, bool host_problem_shapes_available = true) {
    allocate(options);
    initialize<ElementType>(options);

    Gemm gemm_op;

    auto arguments = args_from_options(options, hw_info, host_problem_shapes_available);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();

    if (options.verify != 0) {
      // Verify that the result is correct
      bool passed = verify<ElementType>(options);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm_op.run());
      }
      compat::wait();

      float cute_time = timer.seconds() * 1000;
      double cute_average_time = double(cute_time) / double(options.iterations);
      double gflops = options.gflops(cute_average_time / 1000.0, options.problem_sizes_host);
      if constexpr (std::is_same_v<ElementType, float_e4m3_t>) {
        std::cout << "Datatype: float_e4m3_t"<< std::endl;
      } else if constexpr (std::is_same_v<ElementType, float_e5m2_t>) {
        std::cout << "Datatype: float_e5m2_t"<< std::endl;
      } else {
        static_assert(cutlass::detail::dependent_false<ElementType>, "Not a valid fp8 datatype.");
      }
      std::cout << "  Problem Sizes, Alpha, Beta " << std::endl;
      for (int32_t i = 0; i < options.groups; ++i) {
        std::cout << "    " << options.problem_sizes_host.at(i);
        std::cout << ", " << alpha_host.at(i) << ", " << beta_host.at(i) << std::endl;
      }
      std::cout << "  Groups      : " << options.groups  << std::endl;
      std::cout << "  Avg runtime : " << cute_average_time << " ms" << std::endl;
      std::cout << "  GFLOPS      : " << gflops << std::endl;
    }

    return cutlass::Status::kSuccess;
  }

};


template <typename ElementType, int TileM, int TileN, int TileK, int WarpM, int WarpN>
cutlass::Status run_with_tile(Options& options, const cutlass::KernelHardwareInfo& hw_info) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  using TileShape = Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;

  using TiledMma =
    typename TiledMMAHelper<
      MMA_Atom<XE_DPAS_TT<8, float, half_t>>, // A,B=FP16; accumulator=FP32
      Layout<TileShape>,
      Layout<Shape<cute::Int<WarpM>, cute::Int<WarpN>, _1>, Stride<cute::Int<WarpN>, _1, _0>>
    >::TiledMMA;

  constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<
      PipelineStages,
      cutlass::gemm::KernelXePtrArrayCooperative>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16Group;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<ElementAccumulator, ElementComputeEpilogue,
          ElementAccumulator, ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallBacks = cutlass::epilogue::fusion::FusionCallbacks<EpilogueDispatchPolicy, EpilogueOp, TileShape,
          decltype(tile_shape(TiledMma()))>;
  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
          EpilogueDispatchPolicy,
          TileShape,
          ElementAccumulator,
          cutlass::gemm::TagToStrideC_t<LayoutC*>,
          ElementOutput,
          cutlass::gemm::TagToStrideC_t<LayoutD*>,
          FusionCallBacks,
          XE_2D_U32x8x16_LD_N,
          void, void,
          XE_2D_U16x8x16_ST_N,
          void, void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
          GEMMDispatchPolicy,
          TileShape,
          ElementType,
          cutlass::gemm::TagToStrideA_t<LayoutA*>,
          ElementType,
          cutlass::gemm::TagToStrideB_t<LayoutB*>,
          TiledMma,
          GmemTiledCopyA, void, void, cute::identity,
          GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::GroupScheduler
  >;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  std::cout << "  TileShape   : [" << TileM << ", " << TileN << ", " << TileK << "]" << std::endl;
  std::cout << "  WarpLayout  : [" << WarpM << ", " << WarpN << "]" << std::endl;

  ExampleRunner<Gemm> runner;
  return runner.template run<ElementType>(options, hw_info);
}

template<typename ElementType>
int launcher(Options& options)
{
  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  int max_m = 0;
  for (auto const& problem : options.problem_sizes_host) {
    max_m = std::max(max_m, get<0>(problem));
  }

  cutlass::Status status = cutlass::Status::kSuccess;
  if (max_m <= 8) {
    status = run_with_tile<ElementType, 8, 256, 64, 1, 8>(options, hw_info);
  } else if (max_m <= 16) {
    status = run_with_tile<ElementType, 16, 128, 64, 2, 4>(options, hw_info);
  } else if (max_m <= 32) {
    status = run_with_tile<ElementType, 32, 256, 16, 4, 4>(options, hw_info);
  } else if (max_m <= 64) {
    status = run_with_tile<ElementType, 64, 256, 16, 4, 4>(options, hw_info);
  } else if (max_m <= 128) {
    status = run_with_tile<ElementType, 128, 256, 16, 8, 4>(options, hw_info);
  } else {
    status = run_with_tile<ElementType, 256, 256, 16, 8, 4>(options, hw_info);
  }

  CUTLASS_CHECK(status);
  return 0;
}

int run_moe_quant_grouped_gemm_fp8(
    int num_tokens,
    int hidden_size,
    int new_hidden_size,
    int num_experts,
    int topk,
    int ep_size,
    int ep_rank,
    int quant_group_size,
    float alpha,
    float beta,
    int iterations,
    int verify,
    const std::string& dtype,
    const std::string& w_dtype,
    const std::string& compute_dtype,
    const std::string& dst_dtype) {
  Options options;
  options.num_tokens = num_tokens;
  options.hidden_size = hidden_size;
  options.new_hidden_size = new_hidden_size;
  options.num_experts = num_experts;
  options.topk = topk;
  options.ep_size = ep_size;
  options.ep_rank = ep_rank;
  options.quant_group_size = quant_group_size;
  options.alpha = alpha;
  options.beta = beta;
  options.iterations = iterations;
  options.verify = verify;
  options.dtype = dtype;
  options.w_dtype = w_dtype;
  options.compute_dtype = compute_dtype;
  options.dst_dtype = dst_dtype;

  if (options.dtype == "float8") {
    options.dtype = "float8_e4m3";
  }
  if (options.w_dtype == "float8") {
    options.w_dtype = "float8_e4m3";
  }
  if (options.compute_dtype == "float8") {
    options.compute_dtype = "float8_e4m3";
  }

  if (options.dtype != "float8_e4m3" && options.dtype != "float8_e5m2") {
    throw std::invalid_argument("dtype must be one of: float8, float8_e4m3, float8_e5m2");
  }
  if (options.w_dtype != "float8_e4m3" && options.w_dtype != "float8_e5m2") {
    throw std::invalid_argument("w_dtype must be one of: float8, float8_e4m3, float8_e5m2");
  }
  if (options.compute_dtype != "float8_e4m3" && options.compute_dtype != "float8_e5m2") {
    throw std::invalid_argument("compute_dtype must be one of: float8, float8_e4m3, float8_e5m2");
  }
  if (options.dst_dtype != "float16" && options.dst_dtype != "bfloat16") {
    throw std::invalid_argument("dst_dtype must be one of: float16, bfloat16");
  }
  if (options.num_experts <= 0 || options.ep_size <= 0 || options.topk <= 0) {
    throw std::invalid_argument("num_experts, ep_size and topk must be positive");
  }
  if (options.num_experts % options.ep_size != 0) {
    throw std::invalid_argument("num_experts must be divisible by ep_size");
  }
  if (options.ep_rank < 0 || options.ep_rank >= options.ep_size) {
    throw std::invalid_argument("ep_rank must be in [0, ep_size)");
  }

  options.m = options.num_tokens;
  options.k = options.hidden_size;
  options.n = options.new_hidden_size;
  options.rebuild_problem_sizes_host_from_dispatch();

  int ret = 0;
  if (options.dtype == "float8_e5m2") {
    ret = launcher<cutlass::float_e5m2_t>(options);
  } else {
    ret = launcher<cutlass::float_e4m3_t>(options);
  }

  if (ret != 0) {
    throw std::runtime_error("run_moe_quant_grouped_gemm_fp8 failed");
  }

  return ret;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run_moe_quant_grouped_gemm_fp8",
      &run_moe_quant_grouped_gemm_fp8,
      pybind11::arg("num_tokens") = 48,
      pybind11::arg("hidden_size") = 8192,
      pybind11::arg("new_hidden_size") = 8192,
      pybind11::arg("num_experts") = 128,
      pybind11::arg("topk") = 8,
      pybind11::arg("ep_size") = 8,
      pybind11::arg("ep_rank") = 0,
      pybind11::arg("quant_group_size") = 0,
      pybind11::arg("alpha") = 1.0f,
      pybind11::arg("beta") = 0.0f,
      pybind11::arg("iterations") = 100,
      pybind11::arg("verify") = 1,
      pybind11::arg("dtype") = "float8",
      pybind11::arg("w_dtype") = "float8",
      pybind11::arg("compute_dtype") = "float8",
      pybind11::arg("dst_dtype") = "float16",
      "Run BMG MoE quant grouped GEMM FP8 with CUTLASS SYCL backend.");
}
