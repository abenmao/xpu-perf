#include <torch/extension.h>

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace py = pybind11;

namespace {

inline sycl::queue& get_cached_queue() {
  static thread_local sycl::queue q{sycl::gpu_selector_v};
  return q;
}

template <typename scalar_t>
inline float to_float(scalar_t x) {
  return static_cast<float>(x);
}

template <typename scalar_t>
inline scalar_t from_float(float x) {
  return static_cast<scalar_t>(x);
}

inline bool keep_lhs_min_or_nan(float lhs_v, int lhs_i, float rhs_v, int rhs_i) {
  bool lhs_nan = std::isnan(lhs_v);
  bool rhs_nan = std::isnan(rhs_v);

  if (lhs_nan) {
    if (rhs_nan) {
      return lhs_i < rhs_i;
    }
    return true;
  }

  if (lhs_v == rhs_v) {
    return lhs_i < rhs_i;
  }

  return lhs_v < rhs_v;
}

inline int normalize_rows_per_group(int rows_per_group) {
  if (rows_per_group <= 0) {
    return 1;
  }
  if (rows_per_group > 256) {
    rows_per_group = 256;
  }
  while (rows_per_group > 1 && (256 % rows_per_group != 0)) {
    --rows_per_group;
  }
  return rows_per_group;
}

template <typename scalar_t>
void reduce_min_compute_into_impl(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor indices,
    const std::string& smalldim_mode,
    int64_t rows_per_group_i64) {
  if (input.dim() != 2) {
    throw std::runtime_error("input must be 2D [batch, dim]");
  }
  if (values.dim() != 2 || indices.dim() != 2) {
    throw std::runtime_error("values/indices must be 2D [batch, 1]");
  }

  const int batch = static_cast<int>(input.size(0));
  const int dim_size = static_cast<int>(input.size(1));
  const int rows_per_group = normalize_rows_per_group(static_cast<int>(rows_per_group_i64));

  if (batch <= 0 || dim_size <= 0) {
    throw std::runtime_error("batch/dim_size must be positive");
  }
  if (values.size(0) != batch || values.size(1) != 1) {
    throw std::runtime_error("values shape must be [batch, 1]");
  }
  if (indices.size(0) != batch || indices.size(1) != 1) {
    throw std::runtime_error("indices shape must be [batch, 1]");
  }
  if (!input.is_contiguous() || !values.is_contiguous() || !indices.is_contiguous()) {
    throw std::runtime_error("input/values/indices must be contiguous");
  }

  auto* d_in = reinterpret_cast<scalar_t*>(input.data_ptr());
  auto* d_min = reinterpret_cast<scalar_t*>(values.data_ptr());
  auto* d_idx = reinterpret_cast<std::int32_t*>(indices.data_ptr());

  sycl::queue& q = get_cached_queue();

  auto run_kernel_baseline = [&]() {
    int wg_size = 256;
    if (dim_size < wg_size) {
      wg_size = 1;
      while (wg_size * 2 <= dim_size) {
        wg_size *= 2;
      }
    }

    sycl::range<1> local_range(static_cast<std::size_t>(wg_size));
    sycl::range<1> global_range(static_cast<std::size_t>(batch) * local_range[0]);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_min(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int row = static_cast<int>(item.get_group(0));
        std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);

          if constexpr (
              std::is_same_v<scalar_t, sycl::half> ||
              std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
            float thread_min = std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            if (dim_size <= 1024) {
              for (int i = lid; i < dim_size; i += (wg_size << 1)) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_min_or_nan(thread_min, thread_idx, v, i)) {
                  thread_min = v;
                  thread_idx = i;
                }

                int j = i + wg_size;
                if (j < dim_size) {
                  float v2 = to_float(d_in[base + static_cast<std::size_t>(j)]);
                  if (!keep_lhs_min_or_nan(thread_min, thread_idx, v2, j)) {
                    thread_min = v2;
                    thread_idx = j;
                  }
                }
              }
            } else {
              constexpr int kInputVec = 4;
              float acc_v[kInputVec];
              int acc_i[kInputVec];

              for (int k = 0; k < kInputVec; ++k) {
                acc_v[k] = std::numeric_limits<float>::infinity();
                acc_i[k] = 0;
              }

              int vec_start = lid * kInputVec;
              int vec_step = wg_size * kInputVec;
              int vec_tail = (dim_size / kInputVec) * kInputVec;

              for (int i = vec_start; i < vec_tail; i += vec_step) {
                float v0 = to_float(d_in[base + static_cast<std::size_t>(i + 0)]);
                float v1 = to_float(d_in[base + static_cast<std::size_t>(i + 1)]);
                float v2 = to_float(d_in[base + static_cast<std::size_t>(i + 2)]);
                float v3 = to_float(d_in[base + static_cast<std::size_t>(i + 3)]);

                if (!keep_lhs_min_or_nan(acc_v[0], acc_i[0], v0, i + 0)) {
                  acc_v[0] = v0;
                  acc_i[0] = i + 0;
                }
                if (!keep_lhs_min_or_nan(acc_v[1], acc_i[1], v1, i + 1)) {
                  acc_v[1] = v1;
                  acc_i[1] = i + 1;
                }
                if (!keep_lhs_min_or_nan(acc_v[2], acc_i[2], v2, i + 2)) {
                  acc_v[2] = v2;
                  acc_i[2] = i + 2;
                }
                if (!keep_lhs_min_or_nan(acc_v[3], acc_i[3], v3, i + 3)) {
                  acc_v[3] = v3;
                  acc_i[3] = i + 3;
                }
              }

              for (int i = vec_tail + lid; i < dim_size; i += wg_size) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_min_or_nan(acc_v[0], acc_i[0], v, i)) {
                  acc_v[0] = v;
                  acc_i[0] = i;
                }
              }

              thread_min = acc_v[0];
              thread_idx = acc_i[0];
              for (int k = 1; k < kInputVec; ++k) {
                if (!keep_lhs_min_or_nan(thread_min, thread_idx, acc_v[k], acc_i[k])) {
                  thread_min = acc_v[k];
                  thread_idx = acc_i[k];
                }
              }
            }
            local_min[lid] = thread_min;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_min[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_min_or_nan(local_min[lid], local_idx[lid], cand_v, cand_i)) {
                  local_min[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_min[0]);
              d_idx[row] = local_idx[0];
            }
          } else {
            float thread_min = std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            for (int i = lid; i < dim_size; i += wg_size) {
              float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
              if (!keep_lhs_min_or_nan(thread_min, thread_idx, v, i)) {
                thread_min = v;
                thread_idx = i;
              }
            }

            local_min[lid] = thread_min;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_min[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_min_or_nan(local_min[lid], local_idx[lid], cand_v, cand_i)) {
                  local_min[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_min[0]);
              d_idx[row] = local_idx[0];
            }
          }
      });
    });
  };

  auto run_kernel_multirow = [&](int rows_per_group_local) {
    const int wg_size = 256;
    const int threads_per_row = wg_size / rows_per_group_local;
    const int group_count = (batch + rows_per_group_local - 1) / rows_per_group_local;

    sycl::range<1> local_range(static_cast<std::size_t>(wg_size));
    sycl::range<1> global_range(static_cast<std::size_t>(group_count) * local_range[0]);

    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_min(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int group_id = static_cast<int>(item.get_group(0));
        int row_in_group = lid / threads_per_row;
        int lane = lid % threads_per_row;
        int row = group_id * rows_per_group_local + row_in_group;

        float thread_min = std::numeric_limits<float>::infinity();
        int thread_idx = 0;

        if (row < batch) {
          std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);
          for (int i = lane; i < dim_size; i += threads_per_row) {
            float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
            if (!keep_lhs_min_or_nan(thread_min, thread_idx, v, i)) {
              thread_min = v;
              thread_idx = i;
            }
          }
        }

        local_min[lid] = thread_min;
        local_idx[lid] = thread_idx;
        item.barrier(sycl::access::fence_space::local_space);

        int seg_base = row_in_group * threads_per_row;
        for (int stride = threads_per_row / 2; stride > 0; stride >>= 1) {
          if (lane < stride) {
            int left = seg_base + lane;
            int right = left + stride;
            float cand_v = local_min[right];
            int cand_i = local_idx[right];
            if (!keep_lhs_min_or_nan(local_min[left], local_idx[left], cand_v, cand_i)) {
              local_min[left] = cand_v;
              local_idx[left] = cand_i;
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
        }

        if (lane == 0 && row < batch) {
          int out_idx = seg_base;
          d_min[row] = from_float<scalar_t>(local_min[out_idx]);
          d_idx[row] = local_idx[out_idx];
        }
      });
    });
  };

  bool use_multirow =
      (smalldim_mode == "multirow" && dim_size <= 2048 && rows_per_group > 1);
  if (use_multirow) {
    run_kernel_multirow(rows_per_group);
  } else {
    run_kernel_baseline();
  }
}

void reduce_min_compute_into(
    torch::Tensor input,
    torch::Tensor values,
    torch::Tensor indices,
    const std::string& smalldim_mode,
    int64_t rows_per_group) {
  // Entry 3: production path. Writes results into caller-provided output tensors.
  if (indices.scalar_type() != torch::kInt32) {
    throw std::runtime_error("indices dtype must be int32");
  }

  const auto st = input.scalar_type();
  if (st == torch::kFloat32) {
    reduce_min_compute_into_impl<float>(input, values, indices, smalldim_mode, rows_per_group);
    return;
  }
  if (st == torch::kFloat16) {
    reduce_min_compute_into_impl<sycl::half>(input, values, indices, smalldim_mode, rows_per_group);
    return;
  }
  if (st == torch::kBFloat16) {
    reduce_min_compute_into_impl<sycl::ext::oneapi::bfloat16>(input, values, indices, smalldim_mode, rows_per_group);
    return;
  }
  throw std::runtime_error("Unsupported input dtype for reduce_min_compute_into");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Python export for the production entry point used by the sycl_ext provider.
  m.def(
      "reduce_min_compute_into",
      &reduce_min_compute_into,
      "Compute reduce_min into provided output tensors",
      py::arg("input"),
      py::arg("values"),
      py::arg("indices"),
      py::arg("smalldim_mode") = "multirow",
      py::arg("rows_per_group") = -1);
}