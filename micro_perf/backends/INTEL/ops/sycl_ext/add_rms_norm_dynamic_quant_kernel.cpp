// Fused add-residual + RMS-norm + smooth-scale dynamic int8 quantization.
// SYCL extension for xpu-perf.
//
// Per token (row m, hidden dim k):
//   after_res[m,k]       = hidden_states[m,k] + residual[m,k]   (if has_residual)
//                        = hidden_states[m,k]                    (otherwise)
//   rms                  = sqrt( mean_k( after_res[m,k]^2 ) + eps )
//   after_norm[m,k]      = after_res[m,k] / rms * norm_weight[k]
//   scaled               = after_norm[m,k] * smooth_scale[k]
//   amax[m]              = max_k |scaled|
//   per_token_scale[m]   = amax[m] / 127                         (clamped >= 1e-10)
//   quant_tokens[m,k]    = round(scaled / per_token_scale[m]).clamp(-128,127)

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {

// ---------- tiny helpers for vectorised loads / stores ----------
template <typename scalar_t>
struct vec4_t {
  scalar_t val[4];
};

struct alignas(4) i8x4_t {
  int8_t v[4];
};

// ---------- kernel functor ----------
//
// Optimisation vs v1: keep intermediate float values in private memory
// across passes so that after_res and after_norm are never *re-read* from
// global memory.  This cuts global memory traffic from ~25 B/elem to
// ~17 B/elem (−32 %).
//
// Inspired by the ESIMD kernel in IPEX (esimd/src/norm.cpp) which holds
// all intermediates in SIMD registers across all three fused stages.

template <typename scalar_t>
class add_rms_norm_dynamic_quant_kernel {
 private:
  // outputs
  int8_t* quant_out_;
  float* per_token_scale_;
  scalar_t* after_res_;
  scalar_t* after_norm_;
  // inputs
  scalar_t const* hidden_states_;
  scalar_t const* residual_;       // may be nullptr
  float const* norm_weight_;
  float const* smooth_scale_;
  int const hidden_size_;
  float const eps_;

  // Max scalar elements any single work-item will process.
  // With wg_size = 512 this supports hidden_size up to 16 × 512 = 8 192,
  // covering all common LLM hidden sizes.
  static constexpr int MAX_PRIV = 16;

  static inline int8_t quant_one(float v, float inv_scale) {
    float r = v * inv_scale;
    r = sycl::fmin(sycl::fmax(r, -128.0f), 127.0f);
    return static_cast<int8_t>(sycl::rint(r));
  }

 public:
  add_rms_norm_dynamic_quant_kernel(
      int8_t* quant_out,
      float* per_token_scale,
      scalar_t* after_res,
      scalar_t* after_norm,
      scalar_t const* hidden_states,
      scalar_t const* residual,
      float const* norm_weight,
      float const* smooth_scale,
      int hidden_size,
      float eps)
      : quant_out_(quant_out),
        per_token_scale_(per_token_scale),
        after_res_(after_res),
        after_norm_(after_norm),
        hidden_states_(hidden_states),
        residual_(residual),
        norm_weight_(norm_weight),
        smooth_scale_(smooth_scale),
        hidden_size_(hidden_size),
        eps_(eps) {}

  void operator()(sycl::nd_item<1> item) const {
    int const tid = item.get_local_id(0);
    int const wg = item.get_local_range(0);
    int const token_idx = item.get_group(0);

    int64_t const row_off =
        static_cast<int64_t>(token_idx) * hidden_size_;

    scalar_t const* hs_row = hidden_states_ + row_off;
    scalar_t const* res_row =
        residual_ ? (residual_ + row_off) : nullptr;
    scalar_t* ares_row = after_res_ + row_off;
    scalar_t* anorm_row = after_norm_ + row_off;
    int8_t* qout_row = quant_out_ + row_off;

    // Private buffer: holds float intermediates across passes so we
    // never re-read after_res / after_norm from global memory.
    float priv[MAX_PRIV];

    // Check vectorisation feasibility (4-element vectors).
    bool const can_vec = (hidden_size_ % 4 == 0) &&
        ((reinterpret_cast<uintptr_t>(hs_row) & 7u) == 0u) &&
        (res_row == nullptr ||
         (reinterpret_cast<uintptr_t>(res_row) & 7u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(ares_row) & 7u) == 0u);

    using xvec_t = vec4_t<scalar_t>;
    using svec_t = vec4_t<float>;
    int const num_vec = hidden_size_ >> 2;

    // ================================================================
    // Pass 1: add residual  +  accumulate sum-of-squares
    //         → write after_res to global, keep float values in priv[]
    // ================================================================
    float thread_sum_sq = 0.0f;
    int n_priv = 0;

    if (can_vec) {
      auto const* hs_v = reinterpret_cast<xvec_t const*>(hs_row);
      auto const* res_v =
          res_row ? reinterpret_cast<xvec_t const*>(res_row) : nullptr;
      auto* ares_v = reinterpret_cast<xvec_t*>(ares_row);

#pragma unroll 4
      for (int i = tid; i < num_vec; i += wg) {
        xvec_t hv = hs_v[i];
        xvec_t rv;
        if (res_v) rv = res_v[i];
        xvec_t av;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float val = static_cast<float>(hv.val[j]);
          if (res_v) val += static_cast<float>(rv.val[j]);
          av.val[j] = static_cast<scalar_t>(val);
          priv[n_priv++] = val;
          thread_sum_sq += val * val;
        }
        ares_v[i] = av;
      }
    } else {
      for (int i = tid; i < hidden_size_; i += wg) {
        float val = static_cast<float>(hs_row[i]);
        if (res_row) val += static_cast<float>(res_row[i]);
        ares_row[i] = static_cast<scalar_t>(val);
        priv[n_priv++] = val;
        thread_sum_sq += val * val;
      }
    }

    // Work-group reduction for sum_sq → variance → rstd.
    float const sum_sq = sycl::reduce_over_group(
        item.get_group(), thread_sum_sq, sycl::plus<float>());

    auto& sh1 = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float>(item.get_group());
    if (tid == 0) {
      float variance = sum_sq / static_cast<float>(hidden_size_);
      sh1 = sycl::rsqrt(variance + eps_);
    }
    sycl::group_barrier(item.get_group());
    float const rstd = sh1;

    // ================================================================
    // Pass 2: RMS-norm × weight → write after_norm to global
    //         then  × smooth_scale → absmax,  keep scaled in priv[]
    //         (reads from priv[] instead of re-reading after_res)
    // ================================================================
    float thread_amax = 0.0f;
    int pi = 0;

    if (can_vec) {
      auto const* nw_v = reinterpret_cast<svec_t const*>(norm_weight_);
      auto const* ss_v = reinterpret_cast<svec_t const*>(smooth_scale_);
      auto* anorm_v = reinterpret_cast<xvec_t*>(anorm_row);

#pragma unroll 4
      for (int i = tid; i < num_vec; i += wg) {
        svec_t nw = nw_v[i];
        svec_t ss = ss_v[i];
        xvec_t nv;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          float normed = priv[pi] * rstd * nw.val[j];
          nv.val[j] = static_cast<scalar_t>(normed);
          float scaled = normed * ss.val[j];
          priv[pi] = scaled;   // reuse slot for pass 3
          pi++;
          thread_amax = sycl::max(thread_amax, sycl::fabs(scaled));
        }
        anorm_v[i] = nv;
      }
    } else {
      for (int i = tid; i < hidden_size_; i += wg) {
        float normed = priv[pi] * rstd * norm_weight_[i];
        anorm_row[i] = static_cast<scalar_t>(normed);
        float scaled = normed * smooth_scale_[i];
        priv[pi] = scaled;
        pi++;
        thread_amax = sycl::max(thread_amax, sycl::fabs(scaled));
      }
    }

    // Work-group reduction for absmax.
    float const row_amax = sycl::reduce_over_group(
        item.get_group(), thread_amax, sycl::maximum<float>());

    auto& sh2 = *sycl::ext::oneapi::group_local_memory_for_overwrite<
        float[2]>(item.get_group());
    if (tid == 0) {
      constexpr float kMinScale = 1e-10f;
      float scale = row_amax / 127.0f;
      if (scale < kMinScale) scale = kMinScale;
      sh2[0] = scale;
      sh2[1] = (row_amax > 0.0f) ? (1.0f / scale) : 0.0f;
      per_token_scale_[token_idx] = scale;
    }
    sycl::group_barrier(item.get_group());
    float const inv_scale = sh2[1];

    // ================================================================
    // Pass 3: quantise directly from priv[] — zero global reads
    // ================================================================
    pi = 0;
    if (can_vec) {
      auto* qout_v = reinterpret_cast<i8x4_t*>(qout_row);

#pragma unroll 4
      for (int i = tid; i < num_vec; i += wg) {
        i8x4_t ov;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          ov.v[j] = quant_one(priv[pi++], inv_scale);
        }
        qout_v[i] = ov;
      }
    } else {
      for (int i = tid; i < hidden_size_; i += wg) {
        qout_row[i] = quant_one(priv[pi++], inv_scale);
      }
    }
  }
};

// ---------- host launcher ----------
void add_rms_norm_dynamic_quant_forward(
    at::Tensor const& hidden_states,
    at::Tensor const& residual,       // may be empty (0-dim) when no residual
    at::Tensor const& norm_weight,
    at::Tensor const& smooth_scale,
    at::Tensor& quant_out,
    at::Tensor& per_token_scale,
    at::Tensor& after_res,
    at::Tensor& after_norm,
    double eps) {
  TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
  TORCH_CHECK(quant_out.is_contiguous(), "quant_out must be contiguous");
  TORCH_CHECK(after_res.is_contiguous(), "after_res must be contiguous");
  TORCH_CHECK(after_norm.is_contiguous(), "after_norm must be contiguous");
  TORCH_CHECK(norm_weight.is_contiguous(), "norm_weight must be contiguous");
  TORCH_CHECK(smooth_scale.is_contiguous(), "smooth_scale must be contiguous");
  TORCH_CHECK(hidden_states.dim() == 2, "hidden_states must be 2-D");
  TORCH_CHECK(quant_out.scalar_type() == at::kChar, "quant_out must be int8");
  TORCH_CHECK(
      norm_weight.scalar_type() == at::kFloat, "norm_weight must be fp32");
  TORCH_CHECK(
      smooth_scale.scalar_type() == at::kFloat, "smooth_scale must be fp32");
  TORCH_CHECK(
      per_token_scale.scalar_type() == at::kFloat,
      "per_token_scale must be fp32");

  int const hidden_size = hidden_states.size(1);
  int const num_tokens = hidden_states.size(0);

  bool const has_residual = residual.numel() > 0;
  if (has_residual) {
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(
        residual.sizes() == hidden_states.sizes(),
        "residual shape must match hidden_states");
  }

  TORCH_CHECK(
      norm_weight.numel() == hidden_size,
      "norm_weight numel must equal hidden_size");
  TORCH_CHECK(
      smooth_scale.numel() == hidden_size,
      "smooth_scale numel must equal hidden_size");
  TORCH_CHECK(
      per_token_scale.numel() == num_tokens,
      "per_token_scale numel must equal num_tokens");

  if (num_tokens == 0) return;

  // Each work-item stores at most ceil(hidden_size / wg_size) elements
  // in a private buffer.  Guard against exceeding the compiled limit.
  constexpr int kMaxPriv = 16;   // must match MAX_PRIV in kernel
  int local_size = std::min(hidden_size, 512);
  if (local_size > 32) {
    local_size = (local_size / 32) * 32;
  }
  int elems_per_wi = (hidden_size + local_size - 1) / local_size;
  TORCH_CHECK(
      elems_per_wi <= kMaxPriv,
      "hidden_size / wg_size exceeds compiled MAX_PRIV (",
      kMaxPriv, "); got ", elems_per_wi);
  sycl::range<1> grid(static_cast<size_t>(num_tokens));
  sycl::range<1> block(static_cast<size_t>(local_size));

  auto& queue =
      c10::xpu::getCurrentXPUStream(hidden_states.device().index()).queue();

  float const eps_f = static_cast<float>(eps);

  AT_DISPATCH_SWITCH(
      hidden_states.scalar_type(),
      "add_rms_norm_dynamic_quant_sycl_ext",
      AT_DISPATCH_CASE(at::ScalarType::Half, [&] {
        auto* res_ptr =
            has_residual ? residual.data_ptr<scalar_t>() : nullptr;
        queue.submit([&](sycl::handler& cgh) {
          auto kernel = add_rms_norm_dynamic_quant_kernel<scalar_t>(
              quant_out.data_ptr<int8_t>(),
              per_token_scale.data_ptr<float>(),
              after_res.data_ptr<scalar_t>(),
              after_norm.data_ptr<scalar_t>(),
              hidden_states.data_ptr<scalar_t>(),
              res_ptr,
              norm_weight.data_ptr<float>(),
              smooth_scale.data_ptr<float>(),
              hidden_size,
              eps_f);
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
        });
      })
      AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] {
        auto* res_ptr =
            has_residual ? residual.data_ptr<scalar_t>() : nullptr;
        queue.submit([&](sycl::handler& cgh) {
          auto kernel = add_rms_norm_dynamic_quant_kernel<scalar_t>(
              quant_out.data_ptr<int8_t>(),
              per_token_scale.data_ptr<float>(),
              after_res.data_ptr<scalar_t>(),
              after_norm.data_ptr<scalar_t>(),
              hidden_states.data_ptr<scalar_t>(),
              res_ptr,
              norm_weight.data_ptr<float>(),
              smooth_scale.data_ptr<float>(),
              hidden_size,
              eps_f);
          cgh.parallel_for(sycl::nd_range<1>(grid * block, block), kernel);
        });
      }));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "add_rms_norm_dynamic_quant_forward",
      &add_rms_norm_dynamic_quant_forward,
      "Fused add-residual + RMS-norm + smooth-scale dynamic int8 quant "
      "(SYCL extension)");
}
