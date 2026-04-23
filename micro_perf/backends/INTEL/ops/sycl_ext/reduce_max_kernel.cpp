#include <torch/extension.h>

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

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

inline bool keep_lhs_max_or_nan(float lhs_v, int lhs_i, float rhs_v, int rhs_i) {
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

  return lhs_v > rhs_v;
}

template <typename scalar_t>
void fill_input(
    std::vector<scalar_t>& h_in,
    int batch,
    int dim_size,
    const std::string& input_mode) {
  for (int b = 0; b < batch; ++b) {
    for (int d = 0; d < dim_size; ++d) {
      float val = 0.0f;
      if (input_mode == "sin") {
        val = std::sin(0.001f * static_cast<float>(b * dim_size + d));
      } else if (input_mode == "tie") {
        val = static_cast<float>((d % 17) - 8);
        if (d == 7 || d == 19) {
          val = 1234.0f;
        }
      } else if (input_mode == "nan_head") {
        val = (d == 0) ? std::numeric_limits<float>::quiet_NaN()
                       : std::sin(0.01f * static_cast<float>(b + d));
      } else if (input_mode == "nan_mix") {
        if (d == 3 || d == 9) {
          val = std::numeric_limits<float>::quiet_NaN();
        } else {
          val = std::cos(0.01f * static_cast<float>(b * dim_size + d));
        }
      } else {
        throw std::runtime_error("Unsupported input_mode for reduce_max_sycl.so: " + input_mode);
      }
      h_in[static_cast<std::size_t>(b) * dim_size + d] = from_float<scalar_t>(val);
    }
  }
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
py::dict reduce_max_benchmark_impl(
    int64_t batch_i64,
    int64_t dim_size_i64,
    int64_t iterations_i64,
    bool verify,
    const std::string& input_mode,
    int64_t warmup_i64,
    const std::string& smalldim_mode,
    int64_t rows_per_group_i64) {
  int batch = static_cast<int>(batch_i64);
  int dim_size = static_cast<int>(dim_size_i64);
  int iterations = static_cast<int>(iterations_i64);
  int warmup = static_cast<int>(warmup_i64);
  int rows_per_group = normalize_rows_per_group(static_cast<int>(rows_per_group_i64));

  if (batch <= 0 || dim_size <= 0 || iterations <= 0) {
    throw std::runtime_error("batch/dim_size/iterations must be positive");
  }
  if (warmup < 0) {
    throw std::runtime_error("warmup must be >= 0");
  }
  if (warmup != 0) {
    throw std::runtime_error("warmup is not supported in binary-equivalent mode for reduce_max_sycl.so");
  }

  const std::size_t elems = static_cast<std::size_t>(batch) * static_cast<std::size_t>(dim_size);
  std::vector<scalar_t> h_in(elems);
  std::vector<scalar_t> h_min(batch);
  std::vector<std::int32_t> h_idx(batch);
  fill_input<scalar_t>(h_in, batch, dim_size, input_mode);

  sycl::queue& q = get_cached_queue();

  scalar_t* d_in = sycl::malloc_device<scalar_t>(elems, q);
  scalar_t* d_min = sycl::malloc_device<scalar_t>(batch, q);
  std::int32_t* d_idx = sycl::malloc_device<std::int32_t>(batch, q);
  if (!d_in || !d_min || !d_idx) {
    throw std::runtime_error("Failed to allocate SYCL buffers in reduce_max_sycl.so");
  }

  q.memcpy(d_in, h_in.data(), elems * sizeof(scalar_t)).wait();

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
      sycl::local_accessor<float, 1> local_max(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int row = static_cast<int>(item.get_group(0));
        std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);

          if constexpr (
              std::is_same_v<scalar_t, sycl::half> ||
              std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
            float thread_max = -std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            if (dim_size <= 1024) {
              for (int i = lid; i < dim_size; i += (wg_size << 1)) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
                  thread_max = v;
                  thread_idx = i;
                }

                int j = i + wg_size;
                if (j < dim_size) {
                  float v2 = to_float(d_in[base + static_cast<std::size_t>(j)]);
                  if (!keep_lhs_max_or_nan(thread_max, thread_idx, v2, j)) {
                    thread_max = v2;
                    thread_idx = j;
                  }
                }
              }
            } else {
              constexpr int kInputVec = 4;
              float acc_v[kInputVec];
              int acc_i[kInputVec];

              for (int k = 0; k < kInputVec; ++k) {
                acc_v[k] = -std::numeric_limits<float>::infinity();
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

                if (!keep_lhs_max_or_nan(acc_v[0], acc_i[0], v0, i + 0)) {
                  acc_v[0] = v0;
                  acc_i[0] = i + 0;
                }
                if (!keep_lhs_max_or_nan(acc_v[1], acc_i[1], v1, i + 1)) {
                  acc_v[1] = v1;
                  acc_i[1] = i + 1;
                }
                if (!keep_lhs_max_or_nan(acc_v[2], acc_i[2], v2, i + 2)) {
                  acc_v[2] = v2;
                  acc_i[2] = i + 2;
                }
                if (!keep_lhs_max_or_nan(acc_v[3], acc_i[3], v3, i + 3)) {
                  acc_v[3] = v3;
                  acc_i[3] = i + 3;
                }
              }

              for (int i = vec_tail + lid; i < dim_size; i += wg_size) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_max_or_nan(acc_v[0], acc_i[0], v, i)) {
                  acc_v[0] = v;
                  acc_i[0] = i;
                }
              }

              thread_max = acc_v[0];
              thread_idx = acc_i[0];
              for (int k = 1; k < kInputVec; ++k) {
                if (!keep_lhs_max_or_nan(thread_max, thread_idx, acc_v[k], acc_i[k])) {
                  thread_max = acc_v[k];
                  thread_idx = acc_i[k];
                }
              }
            }
            local_max[lid] = thread_max;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_max[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_max_or_nan(local_max[lid], local_idx[lid], cand_v, cand_i)) {
                  local_max[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_max[0]);
              d_idx[row] = local_idx[0];
            }
          } else {
            float thread_max = -std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            for (int i = lid; i < dim_size; i += wg_size) {
              float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
              if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
                thread_max = v;
                thread_idx = i;
              }
            }

            local_max[lid] = thread_max;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_max[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_max_or_nan(local_max[lid], local_idx[lid], cand_v, cand_i)) {
                  local_max[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_max[0]);
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
      sycl::local_accessor<float, 1> local_max(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int group_id = static_cast<int>(item.get_group(0));
        int row_in_group = lid / threads_per_row;
        int lane = lid % threads_per_row;
        int row = group_id * rows_per_group_local + row_in_group;

        float thread_max = -std::numeric_limits<float>::infinity();
        int thread_idx = 0;

        if (row < batch) {
          std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);
          for (int i = lane; i < dim_size; i += threads_per_row) {
            float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
            if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
              thread_max = v;
              thread_idx = i;
            }
          }
        }

        local_max[lid] = thread_max;
        local_idx[lid] = thread_idx;
        item.barrier(sycl::access::fence_space::local_space);

        int seg_base = row_in_group * threads_per_row;
        for (int stride = threads_per_row / 2; stride > 0; stride >>= 1) {
          if (lane < stride) {
            int left = seg_base + lane;
            int right = left + stride;
            float cand_v = local_max[right];
            int cand_i = local_idx[right];
            if (!keep_lhs_max_or_nan(local_max[left], local_idx[left], cand_v, cand_i)) {
              local_max[left] = cand_v;
              local_idx[left] = cand_i;
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
        }

        if (lane == 0 && row < batch) {
          int out_idx = seg_base;
          d_min[row] = from_float<scalar_t>(local_max[out_idx]);
          d_idx[row] = local_idx[out_idx];
        }
      });
    });
  };

  auto run_kernel = [&]() {
    bool use_multirow =
        (smalldim_mode == "multirow" && dim_size <= 2048 && rows_per_group > 1);
    if (use_multirow) {
      run_kernel_multirow(rows_per_group);
    } else {
      run_kernel_baseline();
    }
  };

  const int fixed_warmup = 5;
  for (int i = 0; i < fixed_warmup; ++i) {
    run_kernel();
  }
  q.wait();

  double total_ms = 0.0;
  for (int i = 0; i < iterations; ++i) {
    auto t0 = std::chrono::high_resolution_clock::now();
    run_kernel();
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  bool verify_pass = true;
  if (verify) {
    q.memcpy(h_min.data(), d_min, static_cast<std::size_t>(batch) * sizeof(scalar_t)).wait();
    q.memcpy(h_idx.data(), d_idx, static_cast<std::size_t>(batch) * sizeof(std::int32_t)).wait();

    double max_abs_err = 0.0;
    int mismatch = 0;
    for (int b = 0; b < batch; ++b) {
      float ref_max = -std::numeric_limits<float>::infinity();
      int ref_idx = 0;
      for (int d = 0; d < dim_size; ++d) {
        float v = to_float(h_in[static_cast<std::size_t>(b) * static_cast<std::size_t>(dim_size) + d]);
        if (!keep_lhs_max_or_nan(ref_max, ref_idx, v, d)) {
          ref_max = v;
          ref_idx = d;
        }
      }

      float got_max = to_float(h_min[b]);
      max_abs_err = std::max(
          max_abs_err,
          std::abs(static_cast<double>(got_max) - static_cast<double>(ref_max)));
      if (h_idx[b] != ref_idx) {
        ++mismatch;
      }
    }
    verify_pass = (max_abs_err == 0.0 && mismatch == 0);
  }

  const double avg_ms = total_ms / static_cast<double>(iterations);
  const double io_bytes =
      static_cast<double>(elems) * sizeof(scalar_t) +
      static_cast<double>(batch) * (sizeof(scalar_t) + sizeof(std::int32_t));
  const double gb_s = io_bytes / (avg_ms * 1e6);

  sycl::free(d_in, q);
  sycl::free(d_min, q);
  sycl::free(d_idx, q);

  py::dict ret;
  ret["latency_ms"] = avg_ms;
  ret["gb_s"] = gb_s;
  ret["tflops"] = 0.0;
  ret["verify_pass"] = verify_pass;
  ret["smalldim_mode"] = smalldim_mode;
  if (rows_per_group <= 0) {
    ret["rows_per_group"] = py::none();
  } else {
    ret["rows_per_group"] = py::int_(rows_per_group);
  }
  return ret;
}

py::dict reduce_max_benchmark(
    int64_t batch,
    int64_t dim_size,
    const std::string& dtype_str,
    int64_t iterations,
    bool verify,
    const std::string& input_mode,
    int64_t warmup,
    const std::string& smalldim_mode,
    int64_t rows_per_group) {
  // Entry 1: benchmark path for perf runs. Returns latency/bandwidth/verify metadata.
  if (dtype_str == "float32") {
    return reduce_max_benchmark_impl<float>(
        batch, dim_size, iterations, verify, input_mode, warmup, smalldim_mode, rows_per_group);
  }
  if (dtype_str == "float16") {
    return reduce_max_benchmark_impl<sycl::half>(
        batch, dim_size, iterations, verify, input_mode, warmup, smalldim_mode, rows_per_group);
  }
  if (dtype_str == "bfloat16") {
    return reduce_max_benchmark_impl<sycl::ext::oneapi::bfloat16>(
        batch, dim_size, iterations, verify, input_mode, warmup, smalldim_mode, rows_per_group);
  }
  throw std::runtime_error("Unsupported dtype for reduce_max_sycl.so: " + dtype_str);
}

template <typename scalar_t>
py::dict reduce_max_compute_impl(
    int64_t batch_i64,
    int64_t dim_size_i64,
    const std::string& input_mode,
    const std::string& smalldim_mode,
    int64_t rows_per_group_i64) {
  int batch = static_cast<int>(batch_i64);
  int dim_size = static_cast<int>(dim_size_i64);
  int rows_per_group = normalize_rows_per_group(static_cast<int>(rows_per_group_i64));

  if (batch <= 0 || dim_size <= 0) {
    throw std::runtime_error("batch/dim_size must be positive");
  }

  const std::size_t elems = static_cast<std::size_t>(batch) * static_cast<std::size_t>(dim_size);
  std::vector<scalar_t> h_in(elems);
  std::vector<scalar_t> h_min(batch);
  std::vector<std::int32_t> h_idx(batch);
  fill_input<scalar_t>(h_in, batch, dim_size, input_mode);

  sycl::queue& q = get_cached_queue();

  scalar_t* d_in = sycl::malloc_device<scalar_t>(elems, q);
  scalar_t* d_min = sycl::malloc_device<scalar_t>(batch, q);
  std::int32_t* d_idx = sycl::malloc_device<std::int32_t>(batch, q);
  if (!d_in || !d_min || !d_idx) {
    throw std::runtime_error("Failed to allocate SYCL buffers in reduce_max_sycl.so");
  }

  q.memcpy(d_in, h_in.data(), elems * sizeof(scalar_t)).wait();

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
      sycl::local_accessor<float, 1> local_max(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int row = static_cast<int>(item.get_group(0));
        std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);

          if constexpr (
              std::is_same_v<scalar_t, sycl::half> ||
              std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
            float thread_max = -std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            if (dim_size <= 1024) {
              for (int i = lid; i < dim_size; i += (wg_size << 1)) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
                  thread_max = v;
                  thread_idx = i;
                }

                int j = i + wg_size;
                if (j < dim_size) {
                  float v2 = to_float(d_in[base + static_cast<std::size_t>(j)]);
                  if (!keep_lhs_max_or_nan(thread_max, thread_idx, v2, j)) {
                    thread_max = v2;
                    thread_idx = j;
                  }
                }
              }
            } else {
              constexpr int kInputVec = 4;
              float acc_v[kInputVec];
              int acc_i[kInputVec];

              for (int k = 0; k < kInputVec; ++k) {
                acc_v[k] = -std::numeric_limits<float>::infinity();
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

                if (!keep_lhs_max_or_nan(acc_v[0], acc_i[0], v0, i + 0)) {
                  acc_v[0] = v0;
                  acc_i[0] = i + 0;
                }
                if (!keep_lhs_max_or_nan(acc_v[1], acc_i[1], v1, i + 1)) {
                  acc_v[1] = v1;
                  acc_i[1] = i + 1;
                }
                if (!keep_lhs_max_or_nan(acc_v[2], acc_i[2], v2, i + 2)) {
                  acc_v[2] = v2;
                  acc_i[2] = i + 2;
                }
                if (!keep_lhs_max_or_nan(acc_v[3], acc_i[3], v3, i + 3)) {
                  acc_v[3] = v3;
                  acc_i[3] = i + 3;
                }
              }

              for (int i = vec_tail + lid; i < dim_size; i += wg_size) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_max_or_nan(acc_v[0], acc_i[0], v, i)) {
                  acc_v[0] = v;
                  acc_i[0] = i;
                }
              }

              thread_max = acc_v[0];
              thread_idx = acc_i[0];
              for (int k = 1; k < kInputVec; ++k) {
                if (!keep_lhs_max_or_nan(thread_max, thread_idx, acc_v[k], acc_i[k])) {
                  thread_max = acc_v[k];
                  thread_idx = acc_i[k];
                }
              }
            }
            local_max[lid] = thread_max;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_max[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_max_or_nan(local_max[lid], local_idx[lid], cand_v, cand_i)) {
                  local_max[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_max[0]);
              d_idx[row] = local_idx[0];
            }
          } else {
            float thread_max = -std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            for (int i = lid; i < dim_size; i += wg_size) {
              float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
              if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
                thread_max = v;
                thread_idx = i;
              }
            }

            local_max[lid] = thread_max;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_max[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_max_or_nan(local_max[lid], local_idx[lid], cand_v, cand_i)) {
                  local_max[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_max[0]);
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
      sycl::local_accessor<float, 1> local_max(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int group_id = static_cast<int>(item.get_group(0));
        int row_in_group = lid / threads_per_row;
        int lane = lid % threads_per_row;
        int row = group_id * rows_per_group_local + row_in_group;

        float thread_max = -std::numeric_limits<float>::infinity();
        int thread_idx = 0;

        if (row < batch) {
          std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);
          for (int i = lane; i < dim_size; i += threads_per_row) {
            float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
            if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
              thread_max = v;
              thread_idx = i;
            }
          }
        }

        local_max[lid] = thread_max;
        local_idx[lid] = thread_idx;
        item.barrier(sycl::access::fence_space::local_space);

        int seg_base = row_in_group * threads_per_row;
        for (int stride = threads_per_row / 2; stride > 0; stride >>= 1) {
          if (lane < stride) {
            int left = seg_base + lane;
            int right = left + stride;
            float cand_v = local_max[right];
            int cand_i = local_idx[right];
            if (!keep_lhs_max_or_nan(local_max[left], local_idx[left], cand_v, cand_i)) {
              local_max[left] = cand_v;
              local_idx[left] = cand_i;
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
        }

        if (lane == 0 && row < batch) {
          int out_idx = seg_base;
          d_min[row] = from_float<scalar_t>(local_max[out_idx]);
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
  q.wait();

  q.memcpy(h_min.data(), d_min, static_cast<std::size_t>(batch) * sizeof(scalar_t)).wait();
  q.memcpy(h_idx.data(), d_idx, static_cast<std::size_t>(batch) * sizeof(std::int32_t)).wait();

  sycl::free(d_in, q);
  sycl::free(d_min, q);
  sycl::free(d_idx, q);

  std::vector<float> h_in_f(elems);
  std::vector<float> h_min_f(batch);
  for (std::size_t i = 0; i < elems; ++i) {
    h_in_f[i] = to_float(h_in[i]);
  }
  for (int i = 0; i < batch; ++i) {
    h_min_f[i] = to_float(h_min[i]);
  }

  auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  torch::Tensor input = torch::from_blob(h_in_f.data(), {batch, dim_size}, opts_f).clone();
  torch::Tensor values = torch::from_blob(h_min_f.data(), {batch, 1}, opts_f).clone();
  torch::Tensor indices =
      torch::from_blob(h_idx.data(), {batch, 1}, opts_i).clone();

  py::dict ret;
  ret["input"] = input;
  ret["values"] = values;
  ret["indices"] = indices;
  ret["smalldim_mode"] = smalldim_mode;
  if (rows_per_group <= 0) {
    ret["rows_per_group"] = py::none();
  } else {
    ret["rows_per_group"] = py::int_(rows_per_group);
  }
  return ret;
}

py::dict reduce_max_compute(
    int64_t batch,
    int64_t dim_size,
    const std::string& dtype_str,
    const std::string& input_mode,
    const std::string& smalldim_mode,
    int64_t rows_per_group) {
  // Entry 2: debug/correctness path. Returns materialized input/values/indices tensors.
  if (dtype_str == "float32") {
    return reduce_max_compute_impl<float>(
        batch, dim_size, input_mode, smalldim_mode, rows_per_group);
  }
  if (dtype_str == "float16") {
    return reduce_max_compute_impl<sycl::half>(
        batch, dim_size, input_mode, smalldim_mode, rows_per_group);
  }
  if (dtype_str == "bfloat16") {
    return reduce_max_compute_impl<sycl::ext::oneapi::bfloat16>(
        batch, dim_size, input_mode, smalldim_mode, rows_per_group);
  }
  throw std::runtime_error("Unsupported dtype for reduce_max_sycl.so: " + dtype_str);
}

template <typename scalar_t>
void reduce_max_compute_into_impl(
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
      sycl::local_accessor<float, 1> local_max(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int row = static_cast<int>(item.get_group(0));
        std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);

          if constexpr (
              std::is_same_v<scalar_t, sycl::half> ||
              std::is_same_v<scalar_t, sycl::ext::oneapi::bfloat16>) {
            float thread_max = -std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            if (dim_size <= 1024) {
              for (int i = lid; i < dim_size; i += (wg_size << 1)) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
                  thread_max = v;
                  thread_idx = i;
                }

                int j = i + wg_size;
                if (j < dim_size) {
                  float v2 = to_float(d_in[base + static_cast<std::size_t>(j)]);
                  if (!keep_lhs_max_or_nan(thread_max, thread_idx, v2, j)) {
                    thread_max = v2;
                    thread_idx = j;
                  }
                }
              }
            } else {
              constexpr int kInputVec = 4;
              float acc_v[kInputVec];
              int acc_i[kInputVec];

              for (int k = 0; k < kInputVec; ++k) {
                acc_v[k] = -std::numeric_limits<float>::infinity();
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

                if (!keep_lhs_max_or_nan(acc_v[0], acc_i[0], v0, i + 0)) {
                  acc_v[0] = v0;
                  acc_i[0] = i + 0;
                }
                if (!keep_lhs_max_or_nan(acc_v[1], acc_i[1], v1, i + 1)) {
                  acc_v[1] = v1;
                  acc_i[1] = i + 1;
                }
                if (!keep_lhs_max_or_nan(acc_v[2], acc_i[2], v2, i + 2)) {
                  acc_v[2] = v2;
                  acc_i[2] = i + 2;
                }
                if (!keep_lhs_max_or_nan(acc_v[3], acc_i[3], v3, i + 3)) {
                  acc_v[3] = v3;
                  acc_i[3] = i + 3;
                }
              }

              for (int i = vec_tail + lid; i < dim_size; i += wg_size) {
                float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
                if (!keep_lhs_max_or_nan(acc_v[0], acc_i[0], v, i)) {
                  acc_v[0] = v;
                  acc_i[0] = i;
                }
              }

              thread_max = acc_v[0];
              thread_idx = acc_i[0];
              for (int k = 1; k < kInputVec; ++k) {
                if (!keep_lhs_max_or_nan(thread_max, thread_idx, acc_v[k], acc_i[k])) {
                  thread_max = acc_v[k];
                  thread_idx = acc_i[k];
                }
              }
            }
            local_max[lid] = thread_max;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_max[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_max_or_nan(local_max[lid], local_idx[lid], cand_v, cand_i)) {
                  local_max[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_max[0]);
              d_idx[row] = local_idx[0];
            }
          } else {
            float thread_max = -std::numeric_limits<float>::infinity();
            int thread_idx = 0;

            for (int i = lid; i < dim_size; i += wg_size) {
              float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
              if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
                thread_max = v;
                thread_idx = i;
              }
            }

            local_max[lid] = thread_max;
            local_idx[lid] = thread_idx;
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = wg_size / 2; stride > 0; stride >>= 1) {
              if (lid < stride) {
                float cand_v = local_max[lid + stride];
                int cand_i = local_idx[lid + stride];
                if (!keep_lhs_max_or_nan(local_max[lid], local_idx[lid], cand_v, cand_i)) {
                  local_max[lid] = cand_v;
                  local_idx[lid] = cand_i;
                }
              }
              item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
              d_min[row] = from_float<scalar_t>(local_max[0]);
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
      sycl::local_accessor<float, 1> local_max(local_range, cgh);
      sycl::local_accessor<int, 1> local_idx(local_range, cgh);

      cgh.parallel_for(sycl::nd_range<1>(global_range, local_range), [=](sycl::nd_item<1> item) {
        int lid = static_cast<int>(item.get_local_id(0));
        int group_id = static_cast<int>(item.get_group(0));
        int row_in_group = lid / threads_per_row;
        int lane = lid % threads_per_row;
        int row = group_id * rows_per_group_local + row_in_group;

        float thread_max = -std::numeric_limits<float>::infinity();
        int thread_idx = 0;

        if (row < batch) {
          std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim_size);
          for (int i = lane; i < dim_size; i += threads_per_row) {
            float v = to_float(d_in[base + static_cast<std::size_t>(i)]);
            if (!keep_lhs_max_or_nan(thread_max, thread_idx, v, i)) {
              thread_max = v;
              thread_idx = i;
            }
          }
        }

        local_max[lid] = thread_max;
        local_idx[lid] = thread_idx;
        item.barrier(sycl::access::fence_space::local_space);

        int seg_base = row_in_group * threads_per_row;
        for (int stride = threads_per_row / 2; stride > 0; stride >>= 1) {
          if (lane < stride) {
            int left = seg_base + lane;
            int right = left + stride;
            float cand_v = local_max[right];
            int cand_i = local_idx[right];
            if (!keep_lhs_max_or_nan(local_max[left], local_idx[left], cand_v, cand_i)) {
              local_max[left] = cand_v;
              local_idx[left] = cand_i;
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
        }

        if (lane == 0 && row < batch) {
          int out_idx = seg_base;
          d_min[row] = from_float<scalar_t>(local_max[out_idx]);
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

void reduce_max_compute_into(
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
    reduce_max_compute_into_impl<float>(input, values, indices, smalldim_mode, rows_per_group);
    return;
  }
  if (st == torch::kFloat16) {
    reduce_max_compute_into_impl<sycl::half>(input, values, indices, smalldim_mode, rows_per_group);
    return;
  }
  if (st == torch::kBFloat16) {
    reduce_max_compute_into_impl<sycl::ext::oneapi::bfloat16>(input, values, indices, smalldim_mode, rows_per_group);
    return;
  }
  throw std::runtime_error("Unsupported input dtype for reduce_max_compute_into");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Python exports map 1:1 to benchmark/debug/production entry points above.
  m.def(
      "reduce_max_benchmark",
      &reduce_max_benchmark,
      "ReduceMax benchmark via in-process SYCL extension",
      py::arg("batch"),
      py::arg("dim_size"),
      py::arg("dtype"),
      py::arg("iterations") = 100,
      py::arg("verify") = false,
      py::arg("input_mode") = "sin",
      py::arg("warmup") = 0,
      py::arg("smalldim_mode") = "multirow",
      py::arg("rows_per_group") = -1);

  m.def(
      "reduce_max_compute",
      &reduce_max_compute,
      "Compute reduce_max values/indices once and return tensors for strict correctness checks",
      py::arg("batch"),
      py::arg("dim_size"),
      py::arg("dtype"),
      py::arg("input_mode") = "sin",
      py::arg("smalldim_mode") = "multirow",
      py::arg("rows_per_group") = -1);

    m.def(
      "reduce_max_compute_into",
      &reduce_max_compute_into,
      "Compute reduce_max into provided output tensors",
      py::arg("input"),
      py::arg("values"),
      py::arg("indices"),
      py::arg("smalldim_mode") = "multirow",
      py::arg("rows_per_group") = -1);
}