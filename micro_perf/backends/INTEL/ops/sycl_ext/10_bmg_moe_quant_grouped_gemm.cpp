/***************************************************************************************************
 * Copyright (C) 2025 - 2026 Intel Corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

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
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "include/helper.h"
#include "include/sycl_common.hpp"

using namespace cute;

namespace {

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using ElementInputA = int8_t;
using ElementInputB = int8_t;
using ElementAccumulator = int32_t;
using ElementComputeEpilogue = float;
using ElementSource = float;
using ElementEpilogueOutput = bfloat16_t;
using ElementFinalOutput = bfloat16_t;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

struct MoeQuantOptions {
  int num_tokens = 48;
  int hidden_size = 8192;
  int new_hidden_size = 8192;
  int num_experts = 128;
  int topk = 8;
  int ep_size = 1;
  int ep_rank = 0;
  int quant_group_size = -1;
  int iterations = 100;
  int verify = 1;

  std::string dtype = "int8";
  std::string w_dtype = "int8";
  std::string compute_dtype = "int8";
  std::string dst_dtype = "bfloat16";

  double gflops(int dispatch_tokens) const {
    double flop = 2.0 * dispatch_tokens * hidden_size * new_hidden_size;
    return flop / 1.0e9;
  }
};

struct MoeQuantInterfaceContract {
  int dispatch_tokens = 0;
  int num_experts_per_rank = 0;
  int max_group_tokens = 0;
  std::vector<int> experts_token_count;
  std::vector<int> experts_token_offset;
  std::vector<int> token_to_expert_index;
};

struct MoeQuantTensors {
  std::vector<ElementInputA> scatter_tokens;
  std::vector<float> scatter_per_token_scale;
  std::vector<ElementInputB> experts_weight_int8;
  // per-column scale [E, N]
  std::vector<float> experts_scale;
  std::vector<int> experts_token_count;
  std::vector<int> experts_token_offset;
  std::vector<ElementFinalOutput> output;
};

std::vector<int> compute_expert_dispatch_token_count(const MoeQuantOptions &options) {
  int num_experts_per_rank = options.num_experts / options.ep_size;
  int experts_start_idx = options.ep_rank * num_experts_per_rank;
  int experts_end_idx = experts_start_idx + num_experts_per_rank;

  std::vector<std::vector<int>> experts_idx_for_each_rank;
  experts_idx_for_each_rank.reserve(options.ep_size);
  for (int rank_idx = 0; rank_idx < options.ep_size; ++rank_idx) {
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
  experts_array.reserve(options.num_experts);
  for (int local_idx = 0; local_idx < num_experts_per_rank; ++local_idx) {
    for (int rank_idx = 0; rank_idx < options.ep_size; ++rank_idx) {
      experts_array.push_back(experts_idx_for_each_rank[rank_idx][local_idx]);
    }
  }

  std::vector<int> expert_dispatch_token_count(num_experts_per_rank, 0);
  int cur_expert = 0;
  for (int token_idx = 0; token_idx < options.num_tokens; ++token_idx) {
    for (int topk_idx = 0; topk_idx < options.topk; ++topk_idx) {
      int expert_idx = experts_array[cur_expert];
      if (expert_idx >= experts_start_idx && expert_idx < experts_end_idx) {
        expert_dispatch_token_count[expert_idx - experts_start_idx] += 1;
      }
      cur_expert += 1;
      if (cur_expert >= options.num_experts) {
        cur_expert = 0;
      }
    }
  }
  return expert_dispatch_token_count;
}

std::vector<int> compute_expert_dispatch_token_offset(const std::vector<int> &token_counts) {
  std::vector<int> token_offsets(token_counts.size(), 0);
  int current_offset = 0;
  for (size_t i = 0; i < token_counts.size(); ++i) {
    token_offsets[i] = current_offset;
    current_offset += token_counts[i];
  }
  return token_offsets;
}

std::vector<int> compute_token_to_expert_index(
    const std::vector<int> &token_counts,
    const std::vector<int> &token_offsets) {
  int dispatch_tokens = token_counts.empty()
      ? 0
      : token_offsets.back() + token_counts.back();
  std::vector<int> token_to_expert_index(dispatch_tokens, -1);
  for (size_t expert_idx = 0; expert_idx < token_counts.size(); ++expert_idx) {
    int begin = token_offsets[expert_idx];
    int end = begin + token_counts[expert_idx];
    for (int token_idx = begin; token_idx < end; ++token_idx) {
      token_to_expert_index[token_idx] = static_cast<int>(expert_idx);
    }
  }
  return token_to_expert_index;
}

int sample_index(int sample_id, int extent, int sample_count) {
  if (extent <= 1 || sample_count <= 1) {
    return 0;
  }
  return (sample_id * (extent - 1)) / (sample_count - 1);
}

ElementInputA activation_qvalue(int token_idx, int k_idx) {
  return static_cast<ElementInputA>(((token_idx * 17 + k_idx * 13) % 255) - 127);
}

ElementInputB weight_qvalue_int8(int expert_idx, int n_idx, int k_idx) {
  return static_cast<ElementInputB>(((expert_idx * 31 + n_idx * 7 + k_idx * 3) % 255) - 127);
}

// Int4 weights stored unpacked: one int8 byte per element, values in [-7, 7].
ElementInputB weight_qvalue_int4(int expert_idx, int n_idx, int k_idx) {
  return static_cast<ElementInputB>(((expert_idx * 31 + n_idx * 7 + k_idx * 3) % 15) - 7);
}

float per_token_scale_value(int token_idx) {
  return 0.5f + static_cast<float>((token_idx % 19) + 1) / 32.0f;
}

float expert_scale_value(int expert_idx, int n_idx) {
  return 0.5f + static_cast<float>(((expert_idx + 1) * (n_idx % 23 + 1)) % 29 + 1) / 64.0f;
}

MoeQuantInterfaceContract build_interface_contract(const MoeQuantOptions &options) {
  MoeQuantInterfaceContract contract;
  contract.experts_token_count = compute_expert_dispatch_token_count(options);
  if (contract.experts_token_count.empty()) {
    throw std::runtime_error("No experts available on current ep_rank");
  }

  contract.num_experts_per_rank = static_cast<int>(contract.experts_token_count.size());
  contract.experts_token_offset = compute_expert_dispatch_token_offset(contract.experts_token_count);
  contract.dispatch_tokens = std::accumulate(
      contract.experts_token_count.begin(), contract.experts_token_count.end(), 0);
    contract.max_group_tokens = contract.experts_token_count.empty()
      ? 0
      : *std::max_element(
        contract.experts_token_count.begin(), contract.experts_token_count.end());
    contract.token_to_expert_index = compute_token_to_expert_index(
      contract.experts_token_count, contract.experts_token_offset);
  return contract;
}

MoeQuantTensors build_tensors(const MoeQuantOptions &options, const MoeQuantInterfaceContract &contract) {
  MoeQuantTensors tensors;

  int E = contract.num_experts_per_rank;
  int N = options.new_hidden_size;
  int K = options.hidden_size;

  tensors.scatter_tokens.resize(static_cast<size_t>(contract.dispatch_tokens) * K);
  tensors.scatter_per_token_scale.resize(contract.dispatch_tokens);
  tensors.experts_token_count = contract.experts_token_count;
  tensors.experts_token_offset = contract.experts_token_offset;
  tensors.output.resize(static_cast<size_t>(contract.dispatch_tokens) * N);

  for (int token_idx = 0; token_idx < contract.dispatch_tokens; ++token_idx) {
    tensors.scatter_per_token_scale[token_idx] = per_token_scale_value(token_idx);
    for (int k_idx = 0; k_idx < K; ++k_idx) {
      tensors.scatter_tokens[static_cast<size_t>(token_idx) * K + k_idx] =
          activation_qvalue(token_idx, k_idx);
    }
  }

  // Per-column scale [E, N] float32.
  tensors.experts_scale.resize(static_cast<size_t>(E) * N);
  for (int expert_idx = 0; expert_idx < E; ++expert_idx) {
    for (int n_idx = 0; n_idx < N; ++n_idx) {
      tensors.experts_scale[static_cast<size_t>(expert_idx) * N + n_idx] =
          expert_scale_value(expert_idx, n_idx);
    }
  }

  // Weight storage: [E, K, N] RowMajor (N-contiguous).
  // For int4, values are stored unpacked as int8 (1 byte/element, range [-7, 7]);
  // the CUTLASS kernel is identical to the int8 path.
  tensors.experts_weight_int8.resize(static_cast<size_t>(E) * N * K);
  const bool use_int4 = (options.w_dtype == "int4");
  for (int expert_idx = 0; expert_idx < E; ++expert_idx) {
    for (int k_idx = 0; k_idx < K; ++k_idx) {
      for (int n_idx = 0; n_idx < N; ++n_idx) {
        size_t offset = (static_cast<size_t>(expert_idx) * K + k_idx) * N + n_idx;
        tensors.experts_weight_int8[offset] = use_int4
            ? weight_qvalue_int4(expert_idx, n_idx, k_idx)
            : weight_qvalue_int8(expert_idx, n_idx, k_idx);
      }
    }
  }

  return tensors;
}

template <class Gemm>
struct ExampleRunner {
  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;
  using StrideD = typename Gemm::GemmKernel::InternalStrideD;

  std::vector<typename ProblemShape::UnderlyingProblemShape> problem_sizes_host;
  std::vector<int64_t> offset_A;
  std::vector<int64_t> offset_B;
  std::vector<int64_t> offset_C;
  std::vector<int64_t> offset_D;

  std::vector<StrideA> stride_A_host;
  std::vector<StrideB> stride_B_host;
  std::vector<StrideC> stride_C_host;
  std::vector<StrideD> stride_D_host;

  cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> problem_sizes;
  cutlass::DeviceAllocation<ElementInputA> block_A;
  cutlass::DeviceAllocation<ElementInputB> block_B;
  cutlass::DeviceAllocation<ElementSource> block_C;
  cutlass::DeviceAllocation<ElementEpilogueOutput> block_D;
  cutlass::DeviceAllocation<float> block_scatter_scale;
  cutlass::DeviceAllocation<float> block_expert_scale;
  cutlass::DeviceAllocation<int> block_experts_token_count;
  cutlass::DeviceAllocation<int> block_experts_token_offset;
  cutlass::DeviceAllocation<int> block_token_to_expert_index;

  cutlass::DeviceAllocation<const ElementInputA *> ptr_A;
  cutlass::DeviceAllocation<const ElementInputB *> ptr_B;
  cutlass::DeviceAllocation<const ElementSource *> ptr_C;
  cutlass::DeviceAllocation<ElementEpilogueOutput *> ptr_D;

  cutlass::DeviceAllocation<StrideA> stride_A;
  cutlass::DeviceAllocation<StrideB> stride_B;
  cutlass::DeviceAllocation<StrideC> stride_C;
  cutlass::DeviceAllocation<StrideD> stride_D;

  void prepare_group_metadata(const MoeQuantOptions &options, const MoeQuantInterfaceContract &contract) {
    problem_sizes_host.clear();
    offset_A.clear();
    offset_B.clear();
    offset_C.clear();
    offset_D.clear();
    stride_A_host.clear();
    stride_B_host.clear();
    stride_C_host.clear();
    stride_D_host.clear();

    int64_t total_elements_A = 0;
    int64_t total_elements_B = 0;
    int64_t total_elements_C = 0;
    int64_t total_elements_D = 0;

    for (int expert_idx = 0; expert_idx < contract.num_experts_per_rank; ++expert_idx) {
      int m = contract.experts_token_count[expert_idx];
      int n = options.new_hidden_size;
      int k = options.hidden_size;

      problem_sizes_host.push_back({m, n, k});
      offset_A.push_back(total_elements_A);
      offset_B.push_back(total_elements_B);
      offset_C.push_back(total_elements_C);
      offset_D.push_back(total_elements_D);

      total_elements_A += static_cast<int64_t>(m) * k;
      total_elements_B += static_cast<int64_t>(k) * n;
      total_elements_C += static_cast<int64_t>(m) * n;
      total_elements_D += static_cast<int64_t>(m) * n;

      stride_A_host.push_back(cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}));
      stride_B_host.push_back(cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}));
      stride_C_host.push_back(cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}));
      stride_D_host.push_back(cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}));
    }

    block_A.reset(total_elements_A);
    block_B.reset(total_elements_B);
    block_C.reset(total_elements_C);
    block_D.reset(total_elements_D);
    block_scatter_scale.reset(contract.dispatch_tokens);
    block_expert_scale.reset(static_cast<int64_t>(contract.num_experts_per_rank) * options.new_hidden_size);
    block_experts_token_count.reset(contract.num_experts_per_rank);
    block_experts_token_offset.reset(contract.num_experts_per_rank);
    block_token_to_expert_index.reset(contract.dispatch_tokens);
  }

  void initialize(const MoeQuantOptions &options, const MoeQuantInterfaceContract &contract, const MoeQuantTensors &tensors) {
    prepare_group_metadata(options, contract);

    problem_sizes.reset(contract.num_experts_per_rank);
    problem_sizes.copy_from_host(problem_sizes_host.data());

    std::vector<const ElementInputA *> ptr_A_host(contract.num_experts_per_rank);
    std::vector<const ElementInputB *> ptr_B_host(contract.num_experts_per_rank);
    std::vector<const ElementSource *> ptr_C_host(contract.num_experts_per_rank);
    std::vector<ElementEpilogueOutput *> ptr_D_host(contract.num_experts_per_rank);

    for (int expert_idx = 0; expert_idx < contract.num_experts_per_rank; ++expert_idx) {
      ptr_A_host[expert_idx] = block_A.get() + offset_A[expert_idx];
      ptr_B_host[expert_idx] = block_B.get() + offset_B[expert_idx];
      ptr_C_host[expert_idx] = block_C.get() + offset_C[expert_idx];
      ptr_D_host[expert_idx] = block_D.get() + offset_D[expert_idx];
    }

    ptr_A.reset(contract.num_experts_per_rank);
    ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.reset(contract.num_experts_per_rank);
    ptr_B.copy_from_host(ptr_B_host.data());
    ptr_C.reset(contract.num_experts_per_rank);
    ptr_C.copy_from_host(ptr_C_host.data());
    ptr_D.reset(contract.num_experts_per_rank);
    ptr_D.copy_from_host(ptr_D_host.data());

    stride_A.reset(contract.num_experts_per_rank);
    stride_A.copy_from_host(stride_A_host.data());
    stride_B.reset(contract.num_experts_per_rank);
    stride_B.copy_from_host(stride_B_host.data());
    stride_C.reset(contract.num_experts_per_rank);
    stride_C.copy_from_host(stride_C_host.data());
    stride_D.reset(contract.num_experts_per_rank);
    stride_D.copy_from_host(stride_D_host.data());

    block_A.copy_from_host(tensors.scatter_tokens.data());
    block_B.copy_from_host(tensors.experts_weight_int8.data());
    block_scatter_scale.copy_from_host(tensors.scatter_per_token_scale.data());
    block_expert_scale.copy_from_host(tensors.experts_scale.data());
    block_experts_token_count.copy_from_host(tensors.experts_token_count.data());
    block_experts_token_offset.copy_from_host(tensors.experts_token_offset.data());
    block_token_to_expert_index.copy_from_host(contract.token_to_expert_index.data());

    // Pre-compute combined scale matrix in block_C: per-group [M_g, N] float
    // combined_scale[token][col] = scatter_scale[token] * expert_scale[expert * N + col]
    {
      int N = options.new_hidden_size;
      std::vector<ElementSource> combined_scale(block_C.size(), ElementSource(0));
      for (int expert_idx = 0; expert_idx < contract.num_experts_per_rank; ++expert_idx) {
        int m_group = contract.experts_token_count[expert_idx];
        int token_start = contract.experts_token_offset[expert_idx];
        for (int local_m = 0; local_m < m_group; ++local_m) {
          int global_token = token_start + local_m;
          float tok_scale = tensors.scatter_per_token_scale[global_token];
          for (int col = 0; col < N; ++col) {
            float exp_scale = tensors.experts_scale[static_cast<size_t>(expert_idx) * N + col];
            combined_scale[offset_C[expert_idx] + static_cast<int64_t>(local_m) * N + col] =
                tok_scale * exp_scale;
          }
        }
      }
      block_C.copy_from_host(combined_scale.data());
    }

    std::vector<ElementEpilogueOutput> zero_d(block_D.size(), ElementEpilogueOutput(0.0f));
    block_D.copy_from_host(zero_d.data());
  }

  typename Gemm::Arguments args_from_options(
      const MoeQuantInterfaceContract &contract,
      const cutlass::KernelHardwareInfo &hw_info) {
    typename Gemm::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args{};

    using RasterOrderOptions =
        typename cutlass::gemm::kernel::detail::PersistentTileSchedulerXeGroup<ProblemShape>::RasterOrderOptions;

    arguments = typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {contract.num_experts_per_rank, problem_sizes.get(), problem_sizes_host.data()},
        {ptr_A.get(), stride_A.get(), ptr_B.get(), stride_B.get()},
        {fusion_args, ptr_C.get(), stride_C.get(), ptr_D.get(), stride_D.get()},
        hw_info,
        {1, RasterOrderOptions::AlongN}};

    return arguments;
  }

  bool verify(const MoeQuantOptions &options, const MoeQuantInterfaceContract &contract) {
    auto queue = compat::get_default_queue();
    size_t total_outputs = static_cast<size_t>(contract.dispatch_tokens) * options.new_hidden_size;
    constexpr size_t kLocalSize = 128;
    size_t global_size = ((total_outputs + kLocalSize - 1) / kLocalSize) * kLocalSize;

    int hidden_size = options.hidden_size;
    int new_hidden_size = options.new_hidden_size;
    const ElementInputA *activations = block_A.get();
    const ElementInputB *weights = block_B.get();
    const float *scatter_scale = block_scatter_scale.get();
    const float *expert_scale = block_expert_scale.get();
    const int *token_to_expert_index = block_token_to_expert_index.get();
    const ElementEpilogueOutput *output = block_D.get();

    int *mismatch_flag = sycl::malloc_shared<int>(1, queue);
    if (mismatch_flag == nullptr) {
      throw std::runtime_error("malloc_shared failed for mismatch_flag");
    }
    *mismatch_flag = 0;

    queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kLocalSize)),
        [=](sycl::nd_item<1> item) {
          size_t linear_idx = item.get_global_linear_id();
          if (linear_idx >= total_outputs || *mismatch_flag != 0) {
            return;
          }

          int token_idx = static_cast<int>(linear_idx / new_hidden_size);
          int output_idx = static_cast<int>(linear_idx % new_hidden_size);
          int expert_idx = token_to_expert_index[token_idx];
          if (expert_idx < 0) {
            return;
          }

          int32_t accum = 0;
          size_t activation_base = static_cast<size_t>(token_idx) * hidden_size;
          size_t expert_weight_base = static_cast<size_t>(expert_idx) * hidden_size * new_hidden_size;
          for (int k_idx = 0; k_idx < hidden_size; ++k_idx) {
            accum += static_cast<int32_t>(activations[activation_base + k_idx]) *
                     static_cast<int32_t>(weights[expert_weight_base + static_cast<size_t>(k_idx) * new_hidden_size + output_idx]);
          }

          float combined_scale = scatter_scale[token_idx] *
                                 expert_scale[static_cast<size_t>(expert_idx) * new_hidden_size + output_idx];
          float expected = static_cast<float>(accum) * combined_scale;
          ElementFinalOutput expected_bf16(expected);

          if (static_cast<float>(expected_bf16) != static_cast<float>(output[linear_idx])) {
            sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                mismatch(*mismatch_flag);
            mismatch.store(1);
          }
        });

    queue.wait();
    bool passed = (*mismatch_flag == 0);
    sycl::free(mismatch_flag, queue);
    return passed;
  }

  cutlass::Status run(
      const MoeQuantOptions &options,
      const MoeQuantInterfaceContract &contract,
      const MoeQuantTensors &tensors,
      const cutlass::KernelHardwareInfo &hw_info) {
    initialize(options, contract, tensors);

    Gemm gemm_op;
    auto arguments = args_from_options(contract, hw_info);

    size_t workspace_size = Gemm::get_workspace_size(arguments);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    CUTLASS_CHECK(gemm_op.can_implement(arguments));
    CUTLASS_CHECK(gemm_op.initialize(arguments, workspace.get()));
    CUTLASS_CHECK(gemm_op.run());
    compat::wait();

    if (options.verify != 0) {
      bool passed = verify(options, contract);
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
      for (int iter = 0; iter < options.iterations; ++iter) {
        CUTLASS_CHECK(gemm_op.run());
        compat::wait();
      }

      float total_time_ms = timer.seconds() * 1000.0f;
      double avg_runtime_ms = static_cast<double>(total_time_ms) / options.iterations;
      double gflops = options.gflops(contract.dispatch_tokens) / (avg_runtime_ms / 1000.0);

      std::cout << "  Groups      : " << contract.num_experts_per_rank << std::endl;
      std::cout << "  Avg runtime : " << avg_runtime_ms << " ms" << std::endl;
      std::cout << "  GFLOPS      : " << gflops << std::endl;
    }

    return cutlass::Status::kSuccess;
  }
};

} // namespace

template <int TileM, int TileN, int TileK, int WarpM, int WarpN>
static cutlass::Status run_with_tile(
  const MoeQuantOptions &options,
  const MoeQuantInterfaceContract &contract,
  const MoeQuantTensors &tensors,
  const cutlass::KernelHardwareInfo &hw_info) {
  using TileShape = Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
  // WarpLayout: WarpM subgroups in M, WarpN subgroups in N, 1 in K.
  // Stride<WarpN, 1, 0>: N direction is contiguous (stride=1), M stride = WarpN.
  using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, int32_t, int8_t, int8_t>>,
    Layout<TileShape>,
    Layout<Shape<cute::Int<WarpM>, cute::Int<WarpN>, _1>,
           Stride<cute::Int<WarpN>, _1, _0>>>::TiledMMA;

  constexpr int PipelineStages = 3;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopXeL1StagedGroup<PipelineStages,
      cutlass::gemm::KernelXePtrArrayCooperative>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGenericGroup;

  // Custom EVT: D_bf16 = bf16(float(s32_acc) * float_combined_scale)
  // Combined scale is pre-computed and passed via the per-group C matrix.
  using FusionCallbacks = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, ElementEpilogueOutput, ElementComputeEpilogue,
      cutlass::FloatRoundStyle::round_to_nearest>,
    cutlass::epilogue::fusion::Sm90AccFetch,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementSource>
  >;

  using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy,
    TileShape,
    void,
    ElementSource,
    cutlass::gemm::TagToStrideC_t<LayoutC *>,
    ElementEpilogueOutput,
    cutlass::gemm::TagToStrideC_t<LayoutD *>,
    FusionCallbacks,
    void,
    void>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    GEMMDispatchPolicy,
    TileShape,
    ElementInputA,
    cutlass::gemm::TagToStrideA_t<LayoutA *>,
    ElementInputB,
    cutlass::gemm::TagToStrideB_t<LayoutB *>,
    TiledMma,
    void, void, void, cute::identity,
    void, void, void, cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue,
    cutlass::gemm::GroupScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  std::cout << "  TileShape            : [" << TileM << ", " << TileN << ", " << TileK << "]" << std::endl;
  std::cout << "  WarpLayout           : [" << WarpM << ", " << WarpN << "]" << std::endl;
  ExampleRunner<Gemm> runner;
  return runner.run(options, contract, tensors, hw_info);
}

int run_moe_quant_grouped_gemm_int8(
    int num_tokens,
    int hidden_size,
    int new_hidden_size,
    int num_experts,
    int topk,
    int ep_size,
    int ep_rank,
    int quant_group_size,
    int iterations,
    int verify,
    const std::string& dtype,
    const std::string& w_dtype,
    const std::string& compute_dtype,
    const std::string& dst_dtype) {
  MoeQuantOptions options;
  options.num_tokens = num_tokens;
  options.hidden_size = hidden_size;
  options.new_hidden_size = new_hidden_size;
  options.num_experts = num_experts;
  options.topk = topk;
  options.ep_size = ep_size;
  options.ep_rank = ep_rank;
  options.quant_group_size = quant_group_size;
  options.iterations = iterations;
  options.verify = verify;
  options.dtype = dtype;
  options.w_dtype = w_dtype;
  options.compute_dtype = compute_dtype;
  options.dst_dtype = dst_dtype;

  if (options.dtype != "int8") {
    throw std::invalid_argument("dtype must be int8");
  }
  if (options.w_dtype != "int8" && options.w_dtype != "int4") {
    throw std::invalid_argument("w_dtype must be one of: int8, int4");
  }
  if (options.dst_dtype != "bfloat16") {
    throw std::invalid_argument("dst_dtype must be bfloat16");
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

  try {
    MoeQuantInterfaceContract contract = build_interface_contract(options);
    MoeQuantTensors tensors = build_tensors(options, contract);

    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    cutlass::Status status = cutlass::Status::kSuccess;
    const int max_m = contract.max_group_tokens;
    if (max_m <= 8) {
      status = run_with_tile<8, 256, 64, 1, 8>(
          options, contract, tensors, hw_info);
    } else if (max_m <= 16) {
      status = run_with_tile<16, 128, 64, 2, 4>(
          options, contract, tensors, hw_info);
    } else if (max_m <= 32) {
      status = run_with_tile<32, 128, 64, 4, 4>(
          options, contract, tensors, hw_info);
    } else if (max_m <= 64) {
      status = run_with_tile<64, 128, 64, 8, 4>(
          options, contract, tensors, hw_info);
    } else if (max_m <= 128) {
      status = run_with_tile<128, 128, 64, 8, 4>(
          options, contract, tensors, hw_info);
    } else {
      status = run_with_tile<256, 256, 32, 8, 4>(
          options, contract, tensors, hw_info);
    }

    if (status != cutlass::Status::kSuccess) {
      throw std::runtime_error(std::string("run_moe_quant_grouped_gemm_int8 failed: ") + cutlassGetStatusString(status));
    }
  } catch (std::exception const &error) {
    throw std::runtime_error(error.what());
  }

  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "run_moe_quant_grouped_gemm_int8",
      &run_moe_quant_grouped_gemm_int8,
      pybind11::arg("num_tokens") = 48,
      pybind11::arg("hidden_size") = 8192,
      pybind11::arg("new_hidden_size") = 8192,
      pybind11::arg("num_experts") = 128,
      pybind11::arg("topk") = 8,
      pybind11::arg("ep_size") = 1,
      pybind11::arg("ep_rank") = 0,
      pybind11::arg("quant_group_size") = -1,
      pybind11::arg("iterations") = 100,
      pybind11::arg("verify") = 1,
      pybind11::arg("dtype") = "int8",
      pybind11::arg("w_dtype") = "int8",
      pybind11::arg("compute_dtype") = "int8",
      pybind11::arg("dst_dtype") = "bfloat16",
      "Run BMG MoE quant grouped GEMM int8 with CUTLASS SYCL backend.");
}