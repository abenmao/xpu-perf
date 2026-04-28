// scatter_kernel.cpp — Vectorized SYCL scatter kernel for micro_perf.
//
// Uses wide ld_vec/st_vec (16-byte) per slice to avoid narrow d16 stores
// for bf16/fp16, matching fp32 store throughput on Xe2.
//
// Build:
//   icpx -fsycl -shared -fPIC -O3 -std=c++17 \
//       -DTORCH_EXTENSION_NAME=scatter_sycl \
//       $TORCH_INCLUDES -I"$PYTHON_INCLUDE" \
//       scatter_kernel.cpp -o scatter_sycl.so \
//       $TORCH_LIBS -ltorch -ltorch_python -lc10

#include <torch/extension.h>

#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

// ============================================================
// Vec types and ld_vec/st_vec — mirrors MemoryAccess.h
// ============================================================

struct uint4 {
  uint32_t x, y, z, w;
};

template <int N>
struct VecType {};

template <>
struct VecType<16> {
  using type = uint4; // 128-bit
};

template <>
struct VecType<8> {
  using type = uint64_t; // 64-bit
};

template <>
struct VecType<4> {
  using type = uint32_t; // 32-bit
};

template <int N>
inline typename VecType<N>::type ld_vec(const char* addr) {
  using vec_t = typename VecType<N>::type;
  vec_t v;
  std::memcpy(&v, addr, sizeof(vec_t));
  return v;
}

template <int N>
inline void st_vec(char* addr, typename VecType<N>::type v) {
  // For 16-byte (uint4), split into 2x uint64_t for better codegen on Xe2.
  if constexpr (N == 16) {
    uint64_t lo, hi;
    std::memcpy(&lo, &v, 8);
    std::memcpy(&hi, reinterpret_cast<const char*>(&v) + 8, 8);
    *reinterpret_cast<uint64_t*>(addr) = lo;
    *reinterpret_cast<uint64_t*>(addr + 8) = hi;
  } else {
    std::memcpy(addr, &v, sizeof(v));
  }
}

// ============================================================
// Vectorized scatter kernel (16-byte wide ops)
// ============================================================

constexpr int SIMD = 32;
constexpr int ALIGNMENT = 16;

struct VectorizedScatterKernel {
  [[sycl::reqd_sub_group_size(SIMD)]] void operator()(
      sycl::nd_item<2> item) const {
    int64_t ind = idx_[item.get_group(1)];

    int32_t off =
        static_cast<int32_t>(
            item.get_local_range(1) * item.get_group(0) +
            item.get_local_id(1)) *
        ALIGNMENT;
    if (off >= slice_size_)
      return;

    int64_t inp_offset =
        static_cast<int64_t>(item.get_group(1)) * inp_stride_;
    auto vec = ld_vec<ALIGNMENT>(inp_ + inp_offset + off);
    st_vec<ALIGNMENT>(out_ + ind * out_stride_ + off, vec);
  }

  char* out_;
  const char* inp_;
  const int64_t* idx_;
  int64_t slice_size_;
  int64_t inp_stride_;
  int64_t out_stride_;
};

void scatter_vectorized(
    char* dst,
    const char* src,
    const int64_t* idx,
    int64_t num_indices,
    int64_t slice_size_bytes,
    int64_t src_stride_bytes,
    int64_t dst_stride_bytes,
    sycl::queue& q) {
  int64_t max_threads = 512; // conservative
  auto num_threads_needed =
      (slice_size_bytes + ALIGNMENT - 1) / ALIGNMENT;
  // round up to SIMD
  num_threads_needed =
      ((num_threads_needed + SIMD - 1) / SIMD) * SIMD;
  auto wg_size = std::min(max_threads, num_threads_needed);

  auto num_groups_dim0 = static_cast<uint32_t>(
      (slice_size_bytes + wg_size * ALIGNMENT - 1) /
      (wg_size * ALIGNMENT));

  sycl::range<2> local_range(1, wg_size);
  sycl::range<2> global_range(
      num_groups_dim0,
      static_cast<uint32_t>(num_indices) * wg_size);

  auto kernel = VectorizedScatterKernel{
      dst, src, idx, slice_size_bytes, src_stride_bytes, dst_stride_bytes};

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range), kernel);
  });
}

// ============================================================
// Check vectorized eligibility
// ============================================================

bool vectorized_eligible(
    const char* dst_ptr,
    const char* src_ptr,
    int64_t slice_size_bytes,
    int64_t src_stride_bytes,
    int64_t dst_stride_bytes) {
  auto check_align = [](uintptr_t v) { return (v % ALIGNMENT) == 0; };
  return check_align(reinterpret_cast<uintptr_t>(dst_ptr)) &&
      check_align(reinterpret_cast<uintptr_t>(src_ptr)) &&
      check_align(static_cast<uintptr_t>(slice_size_bytes)) &&
      check_align(static_cast<uintptr_t>(src_stride_bytes)) &&
      check_align(static_cast<uintptr_t>(dst_stride_bytes));
}

// ============================================================
// Top-level dispatch
// ============================================================

inline sycl::queue& get_current_queue() {
  return c10::xpu::getCurrentXPUStream().queue();
}

void scatter_compute(
    torch::Tensor dst,
    torch::Tensor src,
    torch::Tensor index) {
  // dst: [dst_batch, dim_size]
  // src: [src_batch, dim_size]
  // index: [src_batch] int64 — dst_row = index[i], dst[dst_row, :] = src[i, :]
  if (dst.dim() != 2 || src.dim() != 2 || index.dim() != 1)
    throw std::runtime_error("Expected dst[M,D], src[N,D], index[N]");
  if (!dst.is_contiguous() || !src.is_contiguous() || !index.is_contiguous())
    throw std::runtime_error("All tensors must be contiguous");
  if (index.scalar_type() != torch::kInt64)
    throw std::runtime_error("index must be int64");
  if (dst.scalar_type() != src.scalar_type())
    throw std::runtime_error("dst and src must have same dtype");

  int64_t num_indices = index.size(0);
  int64_t dim_size = src.size(1);
  if (dst.size(1) != dim_size)
    throw std::runtime_error("dst and src must have same dim_size");

  sycl::queue& q = get_current_queue();

  auto element_size = dst.element_size();
  int64_t slice_bytes = dim_size * element_size;
  int64_t src_stride = slice_bytes;
  int64_t dst_stride = slice_bytes;

  char* dst_ptr = reinterpret_cast<char*>(dst.data_ptr());
  const char* src_ptr = reinterpret_cast<const char*>(src.data_ptr());
  const int64_t* idx_ptr = index.data_ptr<int64_t>();

  if (!vectorized_eligible(dst_ptr, src_ptr, slice_bytes, src_stride, dst_stride))
    throw std::runtime_error(
        "alignment check failed (slice_bytes=" +
        std::to_string(slice_bytes) +
        "). dim_size * element_size must be a multiple of 16.");

  scatter_vectorized(
      dst_ptr, src_ptr, idx_ptr,
      num_indices, slice_bytes, src_stride, dst_stride, q);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "scatter_compute",
      &scatter_compute,
      "Vectorized scatter: dst[index[i], :] = src[i, :]",
      py::arg("dst"),
      py::arg("src"),
      py::arg("index"));
  m.def(
      "sync",
      []() { get_current_queue().wait(); },
      "Wait for all previously submitted kernels to finish");
}
