/***************************************************************************************************
 * Copyright (C) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 -2026 Intel Corporation, All rights reserved.
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
    \brief CUTLASS Intel BMG Gemm Example.

    This example constructs and executes a simple CUTLASS GEMM kernel on Intel BMG hardware, and
    verifies its correctness with a reference implementation
    (cutlass::reference::device::GemmComplex). The example also provides a performance measurement
    for the GEMM in TFLOPS.

    This example makes use of BMGs subgroup cooperative 2d-block copy operations and DPAS instructions.

    The shapes of the A and B matrix are defined at runtime by `options.m`, `.n` and `.k`, and the
    batch size is defined by `options.l`. The tile shape, which defines how much work is executed by
    a single work-group, is defined at compile time by:
    ```
      using TileShape = Shape<_256, _256, _32>;
    ```
    That is, each work-group processes a tile of M=256, N=256, and iterates over `options.k` in
    blocks of K=32.

    Performance of GEMM on BMG is heavily dependent on prefetching the A and B matrices. That is,
    executing Intel specific prefetch instructions for future iterations to ensure that the required
    blocks of A and B are resident in cache before they are needed.

    To build & run this example (from your build dir):

      $ ninja 00_bmg_gemm
      $ ./examples/sycl/00_bmg_gemm/00_bmg_gemm

    Call with `--help` for information about available options
*/

#include <pybind11/pybind11.h>

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <cute/tensor.hpp>
#include <random>
#include <stdexcept>
#include <string>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "include/sycl_common.hpp"
#include "include/helper.h"
#include "oneapi/mkl.hpp"

using namespace cute;

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  int m, n, k, l, warmup, iterations, verify;
  float alpha, beta;
  std::string dtype;

  Options():
    m(5120), n(4096), k(4096), l(1), warmup(50), iterations(20), verify(1),
    alpha(1.f), beta(0.f), dtype("bf16")
  { }

  void normalize_dtype() {
    if (dtype == "bfloat16") {
      dtype = "bf16";
    } else if (dtype == "float32") {
      dtype = "fp32";
    }
  }

};

template <typename ElementInput>
struct DTypeConfig;

template <>
struct DTypeConfig<bfloat16_t> {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = bfloat16_t;
  using ElementInputB = bfloat16_t;
  using ElementOutput = float;
  using MmaAtom = XE_DPAS_TT<8, float, bfloat16_t>;
};

template <>
struct DTypeConfig<float> {
  using ElementAccumulator = float;
  using ElementComputeEpilogue = float;
  using ElementInputA = float;
  using ElementInputB = float;
  using ElementOutput = float;
};

template <
  class Gemm
>
struct ExampleRunner {

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementA> block_A;
  cutlass::DeviceAllocation<ElementB> block_B;
  cutlass::DeviceAllocation<ElementC> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D; // Reference GEMM result for verification

  //
  // Methods
  //

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
          {M, N, K},
          alpha,
          ref_A,
          cutlass::ComplexTransform::kNone,
          ref_B,
          cutlass::ComplexTransform::kNone,
          beta,
          ref_C,
          ref_D,
          ElementAccumulator(0),
          L,     // batch_count
          M * K, // batch_stride_A
          K * N, // batch_stride_B
          M * N, // batch_stride_C
          M * N  // batch_stride_D
        );

    // CUTLASS on SYCL uses the compatibility library compat for e.g. default in-order queue
    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = cutlass::reference::device::BlockCompareEqual(
      block_ref_D.get(), block_D.get(), block_D.size());

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    // Complete the stride by combining static layout info (StrideA) with runtime size info (M,K,L)
    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  cutlass::Status run(const Options& options, const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(problem_size);

    typename Gemm::GemmKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      {block_A.get(), stride_A, block_B.get(), stride_B},
      {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
      hw_info
    };

    Gemm gemm_op;

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess){
      std::cout << "Invalid Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      std::exit(1);
    }

    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));

    // Run the GEMM
    CUTLASS_CHECK(gemm_op.run());

    compat::wait();

    if (options.verify != 0) {
      // Verify that the result is correct
      bool passed = verify(problem_size, options.alpha, options.beta);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

      if (!passed) return cutlass::Status::kErrorInternal;
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    // Use a dedicated warmup phase so timed iterations are comparable across
    // verify=0/1 paths.
    if (options.warmup > 0) {
      for (int i = 0; i < options.warmup; ++i) {
        gemm_op.run();
      }
      compat::wait();
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        gemm_op.run();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }

};

template <int TileM, int TileN, int TileK, int WarpM, int WarpN, int Stages>
cutlass::Status run_xe_example_with_config(const Options& options,
                                           const cutlass::KernelHardwareInfo& hw_info) {
  using Config = DTypeConfig<bfloat16_t>;
  using ElementAccumulator = typename Config::ElementAccumulator;
  using ElementComputeEpilogue = typename Config::ElementComputeEpilogue;
  using ElementInputA = typename Config::ElementInputA;
  using ElementInputB = typename Config::ElementInputB;
  using ElementOutput = typename Config::ElementOutput;
  using MmaAtom = typename Config::MmaAtom;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using GmemTiledCopyA = void;
  using GmemTiledCopyB = void;

  using TileShape = Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
  using WarpLayout = Layout<
      Shape<cute::Int<WarpM>, cute::Int<WarpN>, _1>,
      Stride<cute::Int<WarpN>, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<MMA_Atom<MmaAtom>, Layout<TileShape>, WarpLayout>::TiledMMA;

  constexpr int PipelineStages = Stages;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1Staged<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;

  using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput,
    ElementComputeEpilogue,
    ElementAccumulator,
    ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy,
    EpilogueOp,
    TileShape,
    decltype(tile_shape(TiledMma()))>;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    void,
    ElementAccumulator,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD>,
    FusionCallbacks,
    void,
    void>;

  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementInputA,
    cutlass::gemm::TagToStrideA_t<LayoutA>,
    ElementInputB,
    cutlass::gemm::TagToStrideB_t<LayoutB>,
    TiledMma,
    GmemTiledCopyA, void, void, cute::identity,
    GmemTiledCopyB, void, void, cute::identity>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  std::cout << "TileShape: [" << TileM << ", " << TileN << ", " << TileK << "]" << std::endl;
  std::cout << "WarpLayout: [" << WarpM << ", " << WarpN << "]" << std::endl;
  std::cout << "PipelineStages: " << Stages << std::endl;
  ExampleRunner<Gemm> runner;
  return runner.run(options, hw_info);
}

cutlass::Status run_xe_example(const Options& options,
                               const cutlass::KernelHardwareInfo& hw_info) {
  if (options.m <= 64) {
    return run_xe_example_with_config<64, 256, 32, 2, 4, 2>(options, hw_info);
  }
  return run_xe_example_with_config<128, 128, 32, 4, 4, 3>(options, hw_info);
}

template <typename Config>
struct ReferenceRunner {
  using ElementAccumulator = typename Config::ElementAccumulator;
  using ElementComputeEpilogue = typename Config::ElementComputeEpilogue;
  using ElementInputA = typename Config::ElementInputA;
  using ElementInputB = typename Config::ElementInputB;
  using ElementOutput = typename Config::ElementOutput;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  using ProblemShapeType = Shape<int, int, int, int>;

  uint64_t seed = 0;

  cutlass::DeviceAllocation<ElementInputA> block_A;
  cutlass::DeviceAllocation<ElementInputB> block_B;
  cutlass::DeviceAllocation<ElementOutput> block_C;
  cutlass::DeviceAllocation<ElementOutput> block_D;
  cutlass::DeviceAllocation<ElementOutput> block_ref_D;

  bool verify(const ProblemShapeType& problem_size,
              ElementComputeEpilogue alpha,
              ElementComputeEpilogue beta) {
    auto [M, N, K, L] = problem_size;

    cutlass::TensorRef ref_A(block_A.get(), LayoutA::packed({M, K}));
    cutlass::TensorRef ref_B(block_B.get(), LayoutB::packed({K, N}));
    cutlass::TensorRef ref_C(block_C.get(), LayoutC::packed({M, N}));
    cutlass::TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    cutlass::reference::device::GemmComplex(
      {M, N, K},
      alpha,
      ref_A,
      cutlass::ComplexTransform::kNone,
      ref_B,
      cutlass::ComplexTransform::kNone,
      beta,
      ref_C,
      ref_D,
      ElementAccumulator(0),
      L,
      M * K,
      K * N,
      M * N,
      M * N);

    compat::wait();

    return cutlass::reference::device::BlockCompareEqual(
      block_ref_D.get(), block_D.get(), block_D.size());
  }

  void initialize(const ProblemShapeType& problem_size) {
    auto [M, N, K, L] = problem_size;

    block_A.reset(static_cast<std::size_t>(M) * K * L);
    block_B.reset(static_cast<std::size_t>(K) * N * L);
    block_C.reset(static_cast<std::size_t>(M) * N * L);
    block_D.reset(static_cast<std::size_t>(M) * N * L);
    block_ref_D.reset(static_cast<std::size_t>(M) * N * L);

    initialize_block(block_A, seed + 2023);
    initialize_block(block_B, seed + 2022);
    initialize_block(block_C, seed + 2021);
  }

  cutlass::Status run(const Options& options) {
    using Element = ElementOutput;

    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};
    int M = options.m;
    int N = options.n;
    int K = options.k;
    int L = options.l;

    initialize(problem_size);

    auto run_fp32_kernel = [&]() {
      Element const* ptr_A = block_A.get();
      Element const* ptr_B = block_B.get();
      Element const* ptr_C = block_C.get();
      Element* ptr_D = block_D.get();
      float alpha = options.alpha;
      float beta = options.beta;

      auto queue = compat::get_default_queue();

      if (beta != 0.0f) {
        std::size_t total = static_cast<std::size_t>(M) * N * L;
        queue.memcpy(ptr_D, ptr_C, total * sizeof(Element));
        compat::wait();
      }

      oneapi::mkl::blas::row_major::gemm_batch(
        queue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        static_cast<std::int64_t>(M),
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(K),
        alpha,
        ptr_A,
        static_cast<std::int64_t>(K),
        static_cast<std::int64_t>(M) * K,
        ptr_B,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(K) * N,
        beta,
        ptr_D,
        static_cast<std::int64_t>(N),
        static_cast<std::int64_t>(M) * N,
        static_cast<std::int64_t>(L));
    };

    run_fp32_kernel();
    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(problem_size, options.alpha, options.beta);
      std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;
      if (!passed) {
        return cutlass::Status::kErrorInternal;
      }
    } else {
      std::cout << "Disposition is skipped." << std::endl;
    }

    if (options.iterations > 0) {
      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.iterations; ++i) {
        run_fp32_kernel();
      }
      compat::wait();

      float cute_time = timer.seconds() / options.iterations;
      double tflops = (2.0 * options.m * options.n * options.k * options.l) * 1e-12;
      std::cout << "Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << 'x' << options.l << std::endl;
      printf("Cutlass GEMM Performance:     [%4.3f]TFlop/s  (%6.4f)ms\n", tflops / cute_time, cute_time*1000);
    }

    return cutlass::Status::kSuccess;
  }
};

cutlass::Status run_fp32_example(const Options& options,
                                 const cutlass::KernelHardwareInfo& hw_info) {
  (void)hw_info;
  ReferenceRunner<DTypeConfig<float>> runner;
  return runner.run(options);
}

int run_moe_gating_gemm(
    int m,
    int n,
    int k,
    int l,
    float alpha,
    float beta,
    int warmup,
    int iterations,
    int verify,
    const std::string& dtype) {
  Options options;
  options.m = m;
  options.n = n;
  options.k = k;
  options.l = l;
  options.alpha = alpha;
  options.beta = beta;
  options.warmup = warmup;
  options.iterations = iterations;
  options.verify = verify;
  options.dtype = dtype;
  options.normalize_dtype();

  if (options.dtype != "bf16" && options.dtype != "fp32") {
    throw std::invalid_argument("dtype must be one of: bf16, bfloat16, fp32, float32");
  }

  cutlass::KernelHardwareInfo hw_info;
  hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  cutlass::Status status = cutlass::Status::kErrorInvalidProblem;
  if (options.dtype == "bf16") {
    status = run_xe_example(options, hw_info);
  } else {
    status = run_fp32_example(options, hw_info);
  }

  if (status != cutlass::Status::kSuccess) {
    throw std::runtime_error(std::string("run_moe_gating_gemm failed: ") + cutlassGetStatusString(status));
  }

  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run_moe_gating_gemm",
      &run_moe_gating_gemm,
      pybind11::arg("m") = 5120,
      pybind11::arg("n") = 4096,
      pybind11::arg("k") = 4096,
      pybind11::arg("l") = 1,
      pybind11::arg("alpha") = 1.0f,
      pybind11::arg("beta") = 0.0f,
      pybind11::arg("warmup") = 200,
      pybind11::arg("iterations") = 100,
      pybind11::arg("verify") = 1,
      pybind11::arg("dtype") = "bf16",
      "Run BMG MoE gating GEMM with CUTLASS SYCL backend.");
}
