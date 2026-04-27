// RMS Norm SYCL extension for xpu-perf.
// Ported from dbg/A/src/ATen/native/xpu/sycl/LayerNormKernels.cpp (rms_norm path).

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

constexpr int SIMD = 16;

struct SyclKerConfigBase {};

#define __SYCL_KER_CONFIG_CONVENTION__ SyclKerConfigBase
#define SYCL_REQD_SUB_GROUP_SIZE(SIZE) [[sycl::reqd_sub_group_size(SIZE)]]

template <typename T>
using sycl_local_acc_t = sycl::local_accessor<T, 1>;

template <typename T, int N>
struct alignas(16) aligned_vector {
  T val[N];
};

template <typename T>
bool can_vectorize(const T* ptr, int alignment) {
  uint64_t addr = reinterpret_cast<uint64_t>(ptr);
  return addr % alignment == 0;
}

template <typename KernelFn>
inline void sycl_kernel_submit(
    const sycl::range<1>& global_range,
    const sycl::range<1>& local_range,
    sycl::queue& queue,
    KernelFn kfn) {
  queue.submit([&](sycl::handler& cgh) {
    kfn.sycl_ker_config_convention(cgh);
    cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), kfn);
  });
}

template <typename KernelFn>
inline void sycl_kernel_submit(
    const sycl::range<2>& global_range,
    const sycl::range<2>& local_range,
    sycl::queue& queue,
    KernelFn kfn) {
  queue.submit([&](sycl::handler& cgh) {
    kfn.sycl_ker_config_convention(cgh);
    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range), kfn);
  });
}

namespace rms_norm_impl_detail {

constexpr int granularity = 16;

inline int next_pow2(int val) {
  int result = 1;
  while (result < val) {
    result <<= 1;
  }
  return result;
}

} // namespace rms_norm_impl_detail

template <
    typename T,
    typename T_ACC,
    int UNROLL,
    int threadsPerGroup,
    int maxThreads>
struct RmsNormKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int T_per_load =
      rms_norm_impl_detail::granularity / sizeof(T);

  SYCL_REQD_SUB_GROUP_SIZE(SIMD)
  void operator()(sycl::nd_item<2> item_id) const {
    constexpr int groups_per_block = maxThreads / threadsPerGroup;
    const int row_in_block = item_id.get_local_id(0);
    const int tid = item_id.get_local_id(1);
    const int row_idx =
        item_id.get_group(1) * groups_per_block + row_in_block;

    if (row_idx >= M_) {
      return;
    }

    const int thread_offset = tid * T_per_load;
    const int stride = threadsPerGroup * T_per_load;

    float var_sum = 0.f;
    const T* input_base = X_ + static_cast<int64_t>(row_idx) * N_;

    T local_buffer[UNROLL * T_per_load];

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + i * T_per_load;
      const int iter_offset = i * stride + thread_offset;

      if (aligned_mode_) {
        const bool do_loads = (iter_offset < N_);
        if (do_loads) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(iteration_buffer) =
              *reinterpret_cast<const vec_t*>(input_base + iter_offset);
        } else {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            iteration_buffer[j] = T(0);
          }
        }
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          float up_cast = static_cast<float>(iteration_buffer[j]);
          var_sum += up_cast * up_cast;
        }
      } else {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          const int idx = iter_offset + j;
          T v = (idx < N_) ? input_base[idx] : T(0);
          iteration_buffer[j] = v;
          float up_cast = static_cast<float>(v);
          var_sum += up_cast * up_cast;
        }
      }
    }

    auto sg = item_id.get_sub_group();

    if constexpr (threadsPerGroup <= SIMD) {
      int sg_local_id = sg.get_local_linear_id();
      int local_tid = sg_local_id % threadsPerGroup;

#pragma unroll
      for (int offset = threadsPerGroup / 2; offset > 0; offset >>= 1) {
        float shifted = sycl::shift_group_left(sg, var_sum, offset);
        if (local_tid < offset) {
          var_sum += shifted;
        }
      }

      int partition_in_sg = sg_local_id / threadsPerGroup;
      var_sum = sycl::select_from_group(
          sg, var_sum, partition_in_sg * threadsPerGroup);
    } else {
#pragma unroll
      for (int offset = SIMD / 2; offset > 0; offset >>= 1) {
        var_sum += sycl::shift_group_left(sg, var_sum, offset);
      }

      constexpr int num_warps = threadsPerGroup / SIMD;
      int sg_local_id = sg.get_local_linear_id();
      int warp_id = tid / SIMD;

      if (sg_local_id == 0) {
        shared_[warp_id] = var_sum;
      }
      sycl::group_barrier(item_id.get_group());

      if (warp_id == 0) {
        var_sum =
            (sg_local_id < num_warps) ? shared_[sg_local_id] : 0.f;
#pragma unroll
        for (int offset = SIMD / 2; offset > 0; offset >>= 1) {
          var_sum += sycl::shift_group_left(sg, var_sum, offset);
        }
        if (sg_local_id == 0) {
          shared_[0] = var_sum;
        }
      }
      sycl::group_barrier(item_id.get_group());
      var_sum = shared_[0];
    }

    const float var = var_sum / static_cast<float>(N_);
    const float denom = sycl::rsqrt(var + epsilon_);

    T* block_output = Y_ + static_cast<int64_t>(row_idx) * N_;

#pragma unroll
    for (int i = 0; i < UNROLL; i++) {
      T* iteration_buffer = local_buffer + i * T_per_load;
      const int iter_offset = i * stride + thread_offset;

      T gamma_local[T_per_load];
      if (aligned_mode_) {
        if (gamma_ != nullptr && iter_offset < N_) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(gamma_local) =
              *reinterpret_cast<const vec_t*>(gamma_ + iter_offset);
        } else {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            gamma_local[j] = T(0);
          }
        }
      } else {
        if (gamma_ != nullptr) {
#pragma unroll
          for (int j = 0; j < T_per_load; j++) {
            const int idx = iter_offset + j;
            gamma_local[j] = (idx < N_) ? gamma_[idx] : T(0);
          }
        }
      }

#pragma unroll
      for (int j = 0; j < T_per_load; j++) {
        float val = static_cast<float>(iteration_buffer[j]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_local[j]);
        }
        iteration_buffer[j] = static_cast<T>(val);
      }

      if (aligned_mode_) {
        if (iter_offset < N_) {
          using vec_t = aligned_vector<T, T_per_load>;
          *reinterpret_cast<vec_t*>(block_output + iter_offset) =
              *reinterpret_cast<const vec_t*>(iteration_buffer);
        }
      } else {
#pragma unroll
        for (int j = 0; j < T_per_load; j++) {
          const int idx = iter_offset + j;
          if (idx < N_) {
            block_output[idx] = iteration_buffer[j];
          }
        }
      }
    }

    if (tid == 0 && rstd_ != nullptr) {
      rstd_[row_idx] = static_cast<T_ACC>(denom);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int shared_size =
        (threadsPerGroup > SIMD) ? (threadsPerGroup / SIMD) : 1;
    shared_ = sycl_local_acc_t<float>(shared_size, cgh);
  }

  RmsNormKernelFunctor(
      int N,
      int M,
      float epsilon,
      const T* X,
      const T* gamma,
      T* Y,
      T_ACC* rstd,
      bool aligned_mode)
      : N_(N),
        M_(M),
        epsilon_(epsilon),
        X_(X),
        gamma_(gamma),
        Y_(Y),
        rstd_(rstd),
        aligned_mode_(aligned_mode) {}

 private:
  int N_;
  int M_;
  float epsilon_;
  const T* X_;
  const T* gamma_;
  T* Y_;
  T_ACC* rstd_;
  bool aligned_mode_;
  sycl_local_acc_t<float> shared_;
};

template <typename T, typename T_ACC, int TPB>
struct RmsNormLargeNKernelFunctor
    : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int T_per_load =
      rms_norm_impl_detail::granularity / sizeof(T);
  static constexpr int NWARPS = TPB / SIMD;

  SYCL_REQD_SUB_GROUP_SIZE(SIMD)
  void operator()(sycl::nd_item<1> item_id) const {
    const int row = static_cast<int>(item_id.get_group(0));
    if (row >= M_) {
      return;
    }
    const int tid = static_cast<int>(item_id.get_local_id(0));

    const T* row_in = X_ + static_cast<int64_t>(row) * N_;
    T* row_out = Y_ + static_cast<int64_t>(row) * N_;

    float var_sum = 0.f;
    if (can_vec_) {
      using vec_t = aligned_vector<T, T_per_load>;
      const int n_vec = N_ / T_per_load;
      const int tail_start = n_vec * T_per_load;
      const vec_t* X_vec = reinterpret_cast<const vec_t*>(row_in);
      for (int i = tid; i < n_vec; i += TPB) {
        vec_t v = X_vec[i];
#pragma unroll
        for (int j = 0; j < T_per_load; ++j) {
          float f = static_cast<float>(v.val[j]);
          var_sum += f * f;
        }
      }
      for (int i = tail_start + tid; i < N_; i += TPB) {
        float f = static_cast<float>(row_in[i]);
        var_sum += f * f;
      }
    } else {
      for (int i = tid; i < N_; i += TPB) {
        float f = static_cast<float>(row_in[i]);
        var_sum += f * f;
      }
    }

    auto sg = item_id.get_sub_group();
#pragma unroll
    for (int off = SIMD / 2; off > 0; off >>= 1) {
      var_sum += sycl::shift_group_left(sg, var_sum, off);
    }
    const int sg_lane = static_cast<int>(sg.get_local_linear_id());
    const int wid = tid / SIMD;
    if (sg_lane == 0) {
      shared_[wid] = var_sum;
    }
    sycl::group_barrier(item_id.get_group());
    if (wid == 0) {
      float v = (sg_lane < NWARPS) ? shared_[sg_lane] : 0.f;
#pragma unroll
      for (int off = SIMD / 2; off > 0; off >>= 1) {
        v += sycl::shift_group_left(sg, v, off);
      }
      if (sg_lane == 0) {
        shared_[0] = v;
      }
    }
    sycl::group_barrier(item_id.get_group());
    const float total = shared_[0];
    const float denom =
        sycl::rsqrt(total / static_cast<float>(N_) + epsilon_);

    if (can_vec_) {
      using vec_t = aligned_vector<T, T_per_load>;
      const int n_vec = N_ / T_per_load;
      const int tail_start = n_vec * T_per_load;
      const vec_t* X_vec = reinterpret_cast<const vec_t*>(row_in);
      const vec_t* G_vec = (gamma_ != nullptr)
          ? reinterpret_cast<const vec_t*>(gamma_)
          : nullptr;
      vec_t* Y_vec = reinterpret_cast<vec_t*>(row_out);
      for (int i = tid; i < n_vec; i += TPB) {
        vec_t v = X_vec[i];
        vec_t g;
        if (G_vec != nullptr) {
          g = G_vec[i];
        }
        vec_t out;
#pragma unroll
        for (int j = 0; j < T_per_load; ++j) {
          float val = static_cast<float>(v.val[j]) * denom;
          if (G_vec != nullptr) {
            val *= static_cast<float>(g.val[j]);
          }
          out.val[j] = static_cast<T>(val);
        }
        Y_vec[i] = out;
      }
      for (int i = tail_start + tid; i < N_; i += TPB) {
        float val = static_cast<float>(row_in[i]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_[i]);
        }
        row_out[i] = static_cast<T>(val);
      }
    } else {
      for (int i = tid; i < N_; i += TPB) {
        float val = static_cast<float>(row_in[i]) * denom;
        if (gamma_ != nullptr) {
          val *= static_cast<float>(gamma_[i]);
        }
        row_out[i] = static_cast<T>(val);
      }
    }

    if (tid == 0 && rstd_ != nullptr) {
      rstd_[row] = static_cast<T_ACC>(denom);
    }
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    shared_ = sycl_local_acc_t<float>(NWARPS, cgh);
  }

  RmsNormLargeNKernelFunctor(
      int N,
      int M,
      float epsilon,
      const T* X,
      const T* gamma,
      T* Y,
      T_ACC* rstd,
      bool can_vec)
      : N_(N),
        M_(M),
        epsilon_(epsilon),
        X_(X),
        gamma_(gamma),
        Y_(Y),
        rstd_(rstd),
        can_vec_(can_vec) {}

 private:
  int N_;
  int M_;
  float epsilon_;
  const T* X_;
  const T* gamma_;
  T* Y_;
  T_ACC* rstd_;
  bool can_vec_;
  sycl_local_acc_t<float> shared_;
};

template <typename T, typename T_ACC>
void launch_rms_norm_large_n_kernel(
    int N,
    int M,
    float eps,
    const T* X,
    const T* gamma,
    T* Y,
    T_ACC* rstd) {
  constexpr int TPB = 256;
  constexpr int T_per_load =
      rms_norm_impl_detail::granularity / sizeof(T);
  constexpr int alignment = T_per_load * sizeof(T);

  const bool can_vec = (N % T_per_load == 0) && (N >= T_per_load) &&
      can_vectorize(X, alignment) && can_vectorize(Y, alignment) &&
      (gamma == nullptr || can_vectorize(gamma, alignment));

  using KernelClass = RmsNormLargeNKernelFunctor<T, T_ACC, TPB>;
  KernelClass kfn(N, M, eps, X, gamma, Y, rstd, can_vec);
  sycl::range<1> local_range(static_cast<size_t>(TPB));
  sycl::range<1> global_range(static_cast<size_t>(M) * TPB);
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  sycl_kernel_submit(global_range, local_range, queue, kfn);
}

#define LAUNCH_RMS_NORM_IPEX(UNROLL_VAL, TPG, MAXT)                         \
  do {                                                                       \
    using KernelClass =                                                      \
        RmsNormKernelFunctor<T, T_ACC, UNROLL_VAL, TPG, MAXT>;              \
    KernelClass kfn(                                                         \
        N_int,                                                               \
        M_int,                                                               \
        eps_f,                                                               \
        X_data,                                                              \
        gamma_data,                                                          \
        Y_data,                                                              \
        rstd_data,                                                           \
        aligned_mode);                                                       \
    sycl::range<2> local_range{                                              \
        static_cast<size_t>(groups_per_block), static_cast<size_t>(TPG)};    \
    sycl::range<2> global_range{                                             \
        static_cast<size_t>(groups_per_block),                               \
        static_cast<size_t>(groups_launch) * static_cast<size_t>(TPG)};      \
    sycl_kernel_submit(global_range, local_range, queue, kfn);               \
  } while (0)

template <typename T, typename T_ACC>
void rms_norm_kernel_impl(
    const at::Tensor& X,
    const at::Tensor& gamma,
    int64_t M,
    int64_t N,
    T_ACC eps,
    at::Tensor* Y,
    at::Tensor* rstd) {
  constexpr int T_per_load =
      rms_norm_impl_detail::granularity / sizeof(T);

  const T* X_data = X.const_data_ptr<T>();
  const T* gamma_data =
      gamma.defined() ? gamma.const_data_ptr<T>() : nullptr;
  T* Y_data = Y->data_ptr<T>();
  T_ACC* rstd_data = rstd->data_ptr<T_ACC>();

  constexpr int kIpexMaxN = 16384;

  if (N > kIpexMaxN) {
    launch_rms_norm_large_n_kernel<T, T_ACC>(
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<float>(eps),
        X_data,
        gamma_data,
        Y_data,
        rstd_data);
    return;
  }

  const int N_int = static_cast<int>(N);
  const int M_int = static_cast<int>(M);
  const float eps_f = static_cast<float>(eps);

  constexpr int maxThreads = 256;
  constexpr int internalUnroll = sizeof(T) == 4 ? 4 : 2;

  const bool is_subblock_schedule = (N_int <= 128);
  const int h_per_step =
      is_subblock_schedule ? T_per_load : T_per_load * internalUnroll;

  const int one_step_threads =
      rms_norm_impl_detail::next_pow2((N_int + h_per_step - 1) / h_per_step);
  const int threads_per_group =
      (one_step_threads < maxThreads) ? one_step_threads : maxThreads;

  const int groups_per_block_max = is_subblock_schedule
      ? (maxThreads + threads_per_group - 1) / threads_per_group
      : 1;
  const int groups_per_block =
      (M_int < groups_per_block_max) ? M_int : groups_per_block_max;
  const int groups_launch =
      (M_int + groups_per_block - 1) / groups_per_block;

  const int elems_per_step = threads_per_group * h_per_step;
  const int external_unroll =
      (N_int + elems_per_step - 1) / elems_per_step;

  auto& queue = c10::xpu::getCurrentXPUStream().queue();

  constexpr int kAlignBytes = rms_norm_impl_detail::granularity;
  const bool aligned_mode = (N_int % T_per_load == 0) &&
      can_vectorize(X_data, kAlignBytes) &&
      can_vectorize(Y_data, kAlignBytes) &&
      (gamma_data == nullptr || can_vectorize(gamma_data, kAlignBytes));

  if (is_subblock_schedule) {
    if (threads_per_group == 1) {
      LAUNCH_RMS_NORM_IPEX(1, 1, maxThreads);
    } else if (threads_per_group == 2) {
      LAUNCH_RMS_NORM_IPEX(1, 2, maxThreads);
    } else if (threads_per_group == 4) {
      LAUNCH_RMS_NORM_IPEX(1, 4, maxThreads);
    } else if (threads_per_group == 8) {
      LAUNCH_RMS_NORM_IPEX(1, 8, maxThreads);
    } else if (threads_per_group == 16) {
      LAUNCH_RMS_NORM_IPEX(1, 16, maxThreads);
    } else if (threads_per_group == 32) {
      LAUNCH_RMS_NORM_IPEX(1, 32, maxThreads);
    }
  } else if (external_unroll == 1) {
    LAUNCH_RMS_NORM_IPEX(1 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 2) {
    LAUNCH_RMS_NORM_IPEX(2 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 3) {
    LAUNCH_RMS_NORM_IPEX(3 * internalUnroll, maxThreads, maxThreads);
  } else if (external_unroll == 4) {
    LAUNCH_RMS_NORM_IPEX(4 * internalUnroll, maxThreads, maxThreads);
  }
}

#undef LAUNCH_RMS_NORM_IPEX

torch::Tensor rms_norm_forward(
    const torch::Tensor& X,
    const torch::Tensor& gamma,
    double eps) {
  TORCH_CHECK(X.is_xpu(), "X must be an XPU tensor");
  TORCH_CHECK(gamma.is_xpu(), "gamma must be an XPU tensor");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
  TORCH_CHECK(X.scalar_type() == gamma.scalar_type(), "X and gamma must have same dtype");
  TORCH_CHECK(X.dim() == 2 || X.dim() == 3, "X must be 2D or 3D");
  TORCH_CHECK(gamma.dim() == 1, "gamma must be 1D");

  int64_t M = 0;
  int64_t N = 0;
  if (X.dim() == 2) {
    M = X.size(0);
    N = X.size(1);
  } else {
    M = X.size(0) * X.size(1);
    N = X.size(2);
  }
  TORCH_CHECK(gamma.size(0) == N, "gamma size must match last dimension of X");

  auto X_2d = (X.dim() == 2) ? X : X.view({M, N});
  auto Y_2d = torch::empty_like(X_2d);
  auto rstd = torch::empty({M}, X.options().dtype(torch::kFloat));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      X.scalar_type(),
      "rms_norm_sycl_ext",
      [&]() {
        using acc_t = typename std::conditional<std::is_same<scalar_t, double>::value, double, float>::type;
        rms_norm_kernel_impl<scalar_t, acc_t>(
            X_2d,
            gamma,
            M,
            N,
            static_cast<acc_t>(eps),
            &Y_2d,
            &rstd);
      });

  if (X.dim() == 2) {
    return Y_2d;
  }
  return Y_2d.view_as(X);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "rms_norm_forward",
      &rms_norm_forward,
      "RMSNorm forward (SYCL extension)");
}
