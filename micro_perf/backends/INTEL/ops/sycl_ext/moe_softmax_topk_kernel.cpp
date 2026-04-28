// fused_moe_softmax_topk.cpp
// Fused moe_softmax_topk for Intel XPU (SYCL/ESIMD)
// Removed token_expert_indices / source_rows

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

#include <limits>
#include <cfloat>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
static constexpr int WARP_SIZE = 32;

// 0 = pre-softmax : softmax(gating) -> topk(+bias) -> optional renormalize
// 1 = post-softmax: topk(+bias) on raw gating -> softmax over the k winners
enum class SoftmaxMode : int { Pre = 0, Post = 1 };

static inline float clamp_nan_inf(float v) {
  return (sycl::isnan(v) || sycl::isinf(v)) ? 0.f : v;
}

// ============================================================
// ESIMD fast path
// ============================================================
namespace esimd_fast {
using namespace sycl::ext::intel::esimd;

template <int NUM_EXPERTS>
class EsimdTopkPost {
 public:
  EsimdTopkPost(
      const float* gating, float* weights, int* indices,
      int num_tokens, int topk)
      : gating(gating),
        weights(weights),
        indices(indices),
        num_tokens(num_tokens),
        topk(topk) {}

  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int nid = item.get_global_id(0);

    const float* logits_base = gating + nid * NUM_EXPERTS;
    int* idx_base = indices + nid * topk;
    float* w_base = weights + nid * topk;

    simd<float, NUM_EXPERTS> logits =
        block_load<float, NUM_EXPERTS>(logits_base);

    simd<int,   32> topk_idxv;
    simd<float, 32> topk_weightv = -FLT_MAX;

    for (int i = 0; i < topk; ++i) {
      float max_logit =
          sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits);
      uint32_t index = 0;
#pragma unroll
      for (int n = 0; n < NUM_EXPERTS; n += 32) {
        simd_mask<32> mask = logits.template select<32, 1>(n) == max_logit;
        uint32_t m = pack_mask(mask);
        if (m != 0) {
          index = n + fbl(m);
          logits[index] = -FLT_MAX;
          break;
        }
      }
      topk_idxv[i] = static_cast<int>(index);
      topk_weightv[i] = max_logit;
    }

    simd<float, 32> logitsv =
        exp(topk_weightv -
            sycl::ext::intel::esimd::hmax<float, float, 32>(topk_weightv));
    logitsv = logitsv /
        sycl::ext::intel::esimd::detail::sum<float, float, 32>(logitsv);

    for (int i = 0; i < topk; ++i) {
      idx_base[i] = topk_idxv[i];
      w_base[i] = logitsv[i];
    }
  }

 private:
  const float* gating;
  float* weights;
  int* indices;
  int num_tokens;
  int topk;
};

template <int NUM_EXPERTS>
class EsimdTopk8x2Post {
 public:
  EsimdTopk8x2Post(
      const float* gating, float* weights, int* indices,
      int num_tokens)
      : gating(gating),
        weights(weights),
        indices(indices),
        num_tokens(num_tokens) {}

  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int nid = item.get_global_id(0) << 1;

    const float* logits_base = gating + nid * NUM_EXPERTS;
    int* idx_base = indices + nid * 8;
    float* w_base = weights + nid * 8;

    simd<float, NUM_EXPERTS> logits0 =
        block_load<float, NUM_EXPERTS>(logits_base);
    simd<float, NUM_EXPERTS> logits1 =
        block_load<float, NUM_EXPERTS>(logits_base + NUM_EXPERTS);

    simd<int,   16> topk_idxv;
    simd<float, 16> topk_weightv;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float mx0 =
          sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits0);
      float mx1 =
          sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits1);

#pragma unroll
      for (int n = 0; n < NUM_EXPERTS; n += 32) {
        simd_mask<32> mask = logits0.template select<32, 1>(n) == mx0;
        uint32_t m = pack_mask(mask);
        if (m != 0) {
          uint32_t idx = n + fbl(m);
          logits0[idx] = -FLT_MAX;
          topk_idxv[i]    = static_cast<int>(idx);
          topk_weightv[i] = mx0;
          break;
        }
      }
#pragma unroll
      for (int n = 0; n < NUM_EXPERTS; n += 32) {
        simd_mask<32> mask = logits1.template select<32, 1>(n) == mx1;
        uint32_t m = pack_mask(mask);
        if (m != 0) {
          uint32_t idx = n + fbl(m);
          logits1[idx] = -FLT_MAX;
          topk_idxv[i + 8]    = static_cast<int>(idx);
          topk_weightv[i + 8] = mx1;
          break;
        }
      }
    }

    simd<float, 16> w;
    w.select<8, 1>(0) =
        exp(topk_weightv.select<8, 1>(0) -
            sycl::ext::intel::esimd::hmax<float, float, 8>(
                topk_weightv.select<8, 1>(0)));
    w.select<8, 1>(8) =
        exp(topk_weightv.select<8, 1>(8) -
            sycl::ext::intel::esimd::hmax<float, float, 8>(
                topk_weightv.select<8, 1>(8)));
    w.select<8, 1>(0) = w.select<8, 1>(0) /
        sycl::ext::intel::esimd::detail::sum<float, float, 8>(
            w.select<8, 1>(0));
    w.select<8, 1>(8) = w.select<8, 1>(8) /
        sycl::ext::intel::esimd::detail::sum<float, float, 8>(
            w.select<8, 1>(8));

    __ESIMD_ENS::lsc_block_store<
        int, 16, __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(idx_base, topk_idxv);
    __ESIMD_ENS::lsc_block_store<
        float, 16, __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(w_base, w);
  }

 private:
  const float* gating;
  float* weights;
  int* indices;
  int num_tokens;
};

template <int NUM_EXPERTS>
class EsimdTopkPre {
 public:
  EsimdTopkPre(const float* gating, float* weights, int* indices,
               int num_tokens, int topk, bool renormalize)
      : gating(gating), weights(weights), indices(indices),
        num_tokens(num_tokens), topk(topk), renormalize(renormalize) {}

  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int nid = item.get_global_id(0);
    const float* logits_base = gating + nid * NUM_EXPERTS;
    int*   idx_base = indices + nid * topk;
    float* w_base   = weights + nid * topk;

    simd<float, NUM_EXPERTS> logits =
        block_load<float, NUM_EXPERTS>(logits_base);

    float row_max =
        sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits);
    simd<float, NUM_EXPERTS> e = exp(logits - row_max);
    float row_sum =
        sycl::ext::intel::esimd::detail::sum<float, float, NUM_EXPERTS>(e);
    simd<float, NUM_EXPERTS> probs = e / row_sum;

    simd<int,   32> topk_idxv;
    simd<float, 32> topk_weightv = -FLT_MAX;

    for (int i = 0; i < topk; ++i) {
      float max_p =
          sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(probs);
      uint32_t index = 0;
#pragma unroll
      for (int n = 0; n < NUM_EXPERTS; n += 32) {
        simd_mask<32> mask = probs.template select<32, 1>(n) == max_p;
        uint32_t m = pack_mask(mask);
        if (m != 0) {
          index = n + fbl(m);
          probs[index] = -FLT_MAX;
          break;
        }
      }
      topk_idxv[i]    = static_cast<int>(index);
      topk_weightv[i] = max_p;
    }

    if (renormalize) {
      float s = 0.f;
      for (int i = 0; i < topk; ++i) s += topk_weightv[i];
      const float denom = s > 0.f ? s : 1.f;
      for (int i = 0; i < topk; ++i) topk_weightv[i] /= denom;
    }

    for (int i = 0; i < topk; ++i) {
      idx_base[i] = topk_idxv[i];
      w_base[i]   = topk_weightv[i];
    }
  }

 private:
  const float* gating;
  float* weights;
  int* indices;
  int num_tokens;
  int topk;
  bool renormalize;
};

template <int NUM_EXPERTS>
class EsimdTopk8x2Pre {
 public:
  EsimdTopk8x2Pre(const float* gating, float* weights, int* indices,
                  int num_tokens, bool renormalize)
      : gating(gating), weights(weights), indices(indices),
        num_tokens(num_tokens), renormalize(renormalize) {}

  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL {
    const int nid = item.get_global_id(0) << 1;

    const float* logits_base = gating + nid * NUM_EXPERTS;
    int*   idx_base = indices + nid * 8;
    float* w_base   = weights + nid * 8;

    simd<float, NUM_EXPERTS> logits0 =
        block_load<float, NUM_EXPERTS>(logits_base);
    simd<float, NUM_EXPERTS> logits1 =
        block_load<float, NUM_EXPERTS>(logits_base + NUM_EXPERTS);

    {
      float m0 = sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits0);
      simd<float, NUM_EXPERTS> e0 = exp(logits0 - m0);
      float s0 = sycl::ext::intel::esimd::detail::sum<float, float, NUM_EXPERTS>(e0);
      logits0 = e0 / s0;
    }
    {
      float m1 = sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits1);
      simd<float, NUM_EXPERTS> e1 = exp(logits1 - m1);
      float s1 = sycl::ext::intel::esimd::detail::sum<float, float, NUM_EXPERTS>(e1);
      logits1 = e1 / s1;
    }

    simd<int,   16> topk_idxv;
    simd<float, 16> topk_weightv;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float mx0 =
          sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits0);
      float mx1 =
          sycl::ext::intel::esimd::hmax<float, float, NUM_EXPERTS>(logits1);
#pragma unroll
      for (int n = 0; n < NUM_EXPERTS; n += 32) {
        simd_mask<32> mask = logits0.template select<32, 1>(n) == mx0;
        uint32_t m = pack_mask(mask);
        if (m != 0) {
          uint32_t idx = n + fbl(m);
          logits0[idx] = -FLT_MAX;
          topk_idxv[i]    = static_cast<int>(idx);
          topk_weightv[i] = mx0;
          break;
        }
      }
#pragma unroll
      for (int n = 0; n < NUM_EXPERTS; n += 32) {
        simd_mask<32> mask = logits1.template select<32, 1>(n) == mx1;
        uint32_t m = pack_mask(mask);
        if (m != 0) {
          uint32_t idx = n + fbl(m);
          logits1[idx] = -FLT_MAX;
          topk_idxv[i + 8]    = static_cast<int>(idx);
          topk_weightv[i + 8] = mx1;
          break;
        }
      }
    }

    if (renormalize) {
      float s0 =
          sycl::ext::intel::esimd::detail::sum<float, float, 8>(
              topk_weightv.select<8, 1>(0));
      float s1 =
          sycl::ext::intel::esimd::detail::sum<float, float, 8>(
              topk_weightv.select<8, 1>(8));
      const float d0 = s0 > 0.f ? s0 : 1.f;
      const float d1 = s1 > 0.f ? s1 : 1.f;
      topk_weightv.select<8, 1>(0) = topk_weightv.select<8, 1>(0) / d0;
      topk_weightv.select<8, 1>(8) = topk_weightv.select<8, 1>(8) / d1;
    }

    __ESIMD_ENS::lsc_block_store<
        int, 16, __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(idx_base, topk_idxv);
    __ESIMD_ENS::lsc_block_store<
        float, 16, __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>(w_base, topk_weightv);
  }

 private:
  const float* gating;
  float* weights;
  int* indices;
  int num_tokens;
  bool renormalize;
};

template <int NUM_EXPERTS>
static inline void launch_esimd(
    sycl::queue& queue, const float* gating, float* weights, int* indices,
    int num_tokens, int topk,
    SoftmaxMode mode, bool renormalize) {
  const bool use_x2 =
      (topk == 8) && (num_tokens >= 1024) && ((num_tokens & 1) == 0);

  if (use_x2) {
    sycl::range<1> g(num_tokens / 2), l(1);
    if (mode == SoftmaxMode::Pre) {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(g * l, l),
            EsimdTopk8x2Pre<NUM_EXPERTS>(
                gating, weights, indices,
                num_tokens, renormalize));
      });
    } else {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(g * l, l),
            EsimdTopk8x2Post<NUM_EXPERTS>(
                gating, weights, indices, num_tokens));
      });
    }
  } else {
    sycl::range<1> g(num_tokens), l(1);
    if (mode == SoftmaxMode::Pre) {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(g * l, l),
            EsimdTopkPre<NUM_EXPERTS>(
                gating, weights, indices,
                num_tokens, topk, renormalize));
      });
    } else {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(g * l, l),
            EsimdTopkPost<NUM_EXPERTS>(
                gating, weights, indices, num_tokens, topk));
      });
    }
  }
}

static inline bool maybe_launch(
    sycl::queue& queue, const float* gating, float* weights, int* indices,
    int num_tokens, int num_experts, int topk,
    SoftmaxMode mode, bool renormalize) {
  switch (num_experts) {
    case 32:
      launch_esimd<32>(queue, gating, weights, indices,
                       num_tokens, topk, mode, renormalize);
      return true;
    case 64:
      launch_esimd<64>(queue, gating, weights, indices,
                       num_tokens, topk, mode, renormalize);
      return true;
    case 128:
      launch_esimd<128>(queue, gating, weights, indices,
                        num_tokens, topk, mode, renormalize);
      return true;
    case 256:
      launch_esimd<256>(queue, gating, weights, indices,
                        num_tokens, topk, mode, renormalize);
      return true;
    default:
      return false;
  }
}
}  // namespace esimd_fast

// ====================== Softmax kernel (SLM fallback) =====================
template <int TPB>
class MoeSoftmax {
 public:
  MoeSoftmax(const float* input, const bool* finished, float* output,
             const int num_cols)
      : input(input), finished(finished), output(output), num_cols(num_cols) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);
    const int row_off = group_id_x * num_cols;

    if ((finished != nullptr) && finished[group_id_x]) return;

    float threadData = -std::numeric_limits<float>::infinity();
    for (int ii = local_id_x; ii < num_cols; ii += TPB)
      threadData = MAX(input[row_off + ii], threadData);
    const float maxElem =
        sycl::reduce_over_group(group, threadData, sycl::maximum<float>());

    threadData = 0.f;
    for (int ii = local_id_x; ii < num_cols; ii += TPB)
      threadData += sycl::native::exp(input[row_off + ii] - maxElem);
    const float Z = sycl::reduce_over_group(group, threadData, sycl::plus<>());
    const float inv = 1.f / Z;

    for (int ii = local_id_x; ii < num_cols; ii += TPB) {
      const int idx = row_off + ii;
      float v = sycl::native::exp(input[idx] - maxElem) * inv;
      output[idx] = clamp_nan_inf(v);
    }
  }

 private:
  const float* input;
  const bool* finished;
  float* output;
  const int num_cols;
};

// ====================== TopK kernel (SLM fallback) ========================
template <int TPB, typename IndType>
class MoeTopK {
 public:
  MoeTopK(sycl::local_accessor<int, 1>& prev_winners_slm,
          const float* inputs, const bool* finished, float* output,
          IndType* indices,
          const int num_experts, const int k,
          const int start_expert, const int end_expert,
          const bool renormalize, const float* bias)
      : prev_winners_slm(prev_winners_slm),
        inputs(inputs), finished(finished), output(output),
        indices(indices),
        num_experts(num_experts), k(k),
        start_expert(start_expert), end_expert(end_expert),
        renormalize(renormalize), bias(bias) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<1> item) const {
    auto group = item.get_group();
    auto local_id_x = item.get_local_id(0);
    auto group_id_x = item.get_group(0);

    int* prev_winners = prev_winners_slm
        .template get_multi_ptr<sycl::access::decorated::no>().get();

    const int num_rows = item.get_group_range(0);
    const int block_row = group_id_x;
    const bool row_is_active = finished ? !finished[block_row] : true;
    const int read_off = group_id_x * num_experts;

    float sum_val = 0.0f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
      int   kIdx = 0;
      float kVal = -std::numeric_limits<float>::infinity();

      for (int expert = local_id_x; expert < num_experts; expert += TPB) {
        int   inpIdx = expert;
        float inpVal =
            inputs[read_off + expert] + (bias != nullptr ? bias[expert] : 0.f);

        for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
          if (prev_winners[prior_k] == expert) {
            inpIdx = kIdx;
            inpVal = kVal;
          }
        }
        if (inpVal > kVal) { kIdx = inpIdx; kVal = inpVal; }
      }

      const float resultVal =
          sycl::reduce_over_group(group, kVal, sycl::maximum<float>());
      const int resultIdx = sycl::reduce_over_group(
          group, resultVal == kVal ? kIdx : 0x7FFFFFFF, sycl::minimum<int>());

      if (local_id_x == 0) {
        const int expert = resultIdx;
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;
        const int idx = k * block_row + k_idx;
        const float w = inputs[read_off + expert];
        output[idx]  = w;
        indices[idx] =
            should_process_row ? (expert - start_expert) : num_experts;
        prev_winners[k_idx] = expert;
        if (renormalize) sum_val += w;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (renormalize) {
      sum_val = sycl::group_broadcast(group, sum_val, 0);
      const float denom = sum_val > 0.f ? sum_val : 1.f;
      auto lr = item.get_local_range(0);
      for (int k_idx = local_id_x; k_idx < k; k_idx += lr) {
        output[k * block_row + k_idx] /= denom;
      }
    }
  }

 private:
  sycl::local_accessor<int, 1> prev_winners_slm;
  const float* inputs;
  const bool* finished;
  float* output;
  IndType* indices;
  const int num_experts;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
  const float* bias;
};

// ====================== Fused SYCL fast-path kernel =======================
template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG,
          int WARP_SIZE_PARAM, typename IndType, SoftmaxMode Mode>
class TopKGating {
 public:
  TopKGating(const float* input, const bool* finished, float* output,
             const int num_rows, IndType* indices,
             const int k, const int start_expert, const int end_expert,
             const bool renormalize, const float* bias)
      : input(input), finished(finished), output(output),
        num_rows(num_rows), indices(indices),
        k(k), start_expert(start_expert), end_expert(end_expert),
        renormalize(renormalize), bias(bias) {}

  void operator()
      [[sycl::reqd_sub_group_size(WARP_SIZE)]] (sycl::nd_item<2> item) const {
    auto sg = item.get_sub_group();
    auto local_id_x = item.get_local_id(1);
    auto local_id_y = item.get_local_id(0);
    auto group_id_x = item.get_group(1);

    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                  "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    static constexpr int ELTS_PER_LDG    = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW    = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD    = VPT / ELTS_PER_LDG;

    static_assert(VPT % ELTS_PER_LDG == 0, "");
    static_assert(WARP_SIZE_PARAM % THREADS_PER_ROW == 0, "");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "");
    static_assert(THREADS_PER_ROW <= WARP_SIZE_PARAM, "");

    static constexpr int ELTS_PER_WARP = WARP_SIZE_PARAM * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA  = WARPS_PER_CTA * ROWS_PER_WARP;
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "");

    const int cta_base_row  = group_id_x * ROWS_PER_CTA;
    const int warp_base_row = cta_base_row + local_id_y * ROWS_PER_WARP;
    const int thread_row    = warp_base_row + (local_id_x / THREADS_PER_ROW);
    if (thread_row >= num_rows) return;

    const bool row_is_active = finished ? !finished[thread_row] : true;
    const float* thread_row_ptr = input + thread_row * ELTS_PER_ROW;
    const int tg_idx   = local_id_x % THREADS_PER_ROW;
    const int first_el = tg_idx * ELTS_PER_LDG;
    const float* thread_read_ptr = thread_row_ptr + first_el;

    float row_chunk[VPT];
    if constexpr (ELTS_PER_LDG == 1) {
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
        row_chunk[ii] = thread_read_ptr[ii * THREADS_PER_ROW];
    } else {
      using LoadVec = sycl::vec<float, ELTS_PER_LDG>;
      const LoadVec* vp = reinterpret_cast<const LoadVec*>(thread_read_ptr);
#pragma unroll
      for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
        LoadVec v = vp[ii * THREADS_PER_ROW];
#pragma unroll
        for (int jj = 0; jj < ELTS_PER_LDG; ++jj)
          row_chunk[ii * ELTS_PER_LDG + jj] = v[jj];
      }
    }

    if constexpr (Mode == SoftmaxMode::Pre) {
      float thread_max = row_chunk[0];
#pragma unroll
      for (int ii = 1; ii < VPT; ++ii)
        thread_max = MAX(thread_max, row_chunk[ii]);
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float o = sycl::permute_group_by_xor(sg, thread_max, mask);
        thread_max = MAX(thread_max, o);
      }
      float row_sum = 0.f;
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = sycl::native::exp(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
      }
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
      const float inv = 1.f / row_sum;
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) row_chunk[ii] *= inv;
    }

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
      if (sycl::isnan(row_chunk[ii]) || sycl::isinf(row_chunk[ii]))
        row_chunk[ii] = (Mode == SoftmaxMode::Pre) ? 0.f : -INFINITY;
    }

    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float row_chunk_with_bias[VPT];
    if (bias != nullptr) {
#pragma unroll
      for (int ldg = 0; ldg < LDG_PER_THREAD; ++ldg) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          const int expert = first_el + ldg * COLS_PER_GROUP_LDG + ii;
          float bv = expert < NUM_EXPERTS ? bias[expert] : 0.f;
          row_chunk_with_bias[ldg * ELTS_PER_LDG + ii] =
              row_chunk[ldg * ELTS_PER_LDG + ii] + bv;
        }
      }
    } else {
#pragma unroll
      for (int ii = 0; ii < VPT; ++ii) row_chunk_with_bias[ii] = row_chunk[ii];
    }

    float selected_sum = 0.f;
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      float max_vb = row_chunk_with_bias[0];
      float max_v  = row_chunk[0];
      int   expert_local = first_el;
#pragma unroll
      for (int ldg = 0, col = first_el; ldg < LDG_PER_THREAD;
           ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
        for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
          float vb = row_chunk_with_bias[ldg * ELTS_PER_LDG + ii];
          float v  = row_chunk[ldg * ELTS_PER_LDG + ii];
          if (vb > max_vb) {
            max_vb = vb; max_v = v;
            expert_local = col + ii;
          }
        }
      }

      int expert = expert_local;
#pragma unroll
      for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
        float ovb = sycl::permute_group_by_xor(sg, max_vb, mask);
        float ov  = sycl::permute_group_by_xor(sg, max_v,  mask);
        int   oe  = sycl::permute_group_by_xor(sg, expert, mask);
        if (ovb > max_vb || (ovb == max_vb && oe < expert)) {
          max_vb = ovb; max_v = ov; expert = oe;
        }
      }

      if (tg_idx == 0) {
        const bool node_uses_expert =
            expert >= start_expert && expert < end_expert;
        const bool should_process_row = row_is_active && node_uses_expert;
        const int idx = k * thread_row + k_idx;
        output[idx]  = max_v;
        indices[idx] =
            should_process_row ? (expert - start_expert) : NUM_EXPERTS;
        if constexpr (Mode == SoftmaxMode::Pre) selected_sum += max_v;
      }

      if (k_idx + 1 < k) {
        const int ldg_group   = expert / COLS_PER_GROUP_LDG;
        const int thread_clr  = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;
        if (tg_idx == thread_clr) {
          const int off = expert % ELTS_PER_LDG;
          row_chunk_with_bias[ldg_group * ELTS_PER_LDG + off] = -10000.f;
        }
      }
    }

    if (tg_idx == 0) {
      if constexpr (Mode == SoftmaxMode::Pre) {
        if (renormalize) {
          const float denom = selected_sum > 0.f ? selected_sum : 1.f;
          for (int k_idx = 0; k_idx < k; ++k_idx)
            output[k * thread_row + k_idx] /= denom;
        }
      } else {
        float mx = -INFINITY;
        for (int k_idx = 0; k_idx < k; ++k_idx)
          mx = MAX(mx, output[k * thread_row + k_idx]);
        float sm = 0.f;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
          float e = sycl::native::exp(output[k * thread_row + k_idx] - mx);
          output[k * thread_row + k_idx] = e;
          sm += e;
        }
        const float inv = 1.f / sm;
        for (int k_idx = 0; k_idx < k; ++k_idx)
          output[k * thread_row + k_idx] *= inv;
      }
    }
  }

 private:
  const float* input;
  const bool* finished;
  float* output;
  const int num_rows;
  IndType* indices;
  const int k;
  const int start_expert;
  const int end_expert;
  const bool renormalize;
  const float* bias;
};

namespace detail {
template <int EXPERTS, int BYTES_PER_LDG, int WARP_SIZE_PARAM>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
  static_assert(
      EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0 ||
          EXPERTS % (ELTS_PER_LDG * WARP_SIZE_PARAM) == 0, "");
  static constexpr int VECs_PER_THREAD =
      MAX(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE_PARAM));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE_PARAM / THREADS_PER_ROW;
};
}  // namespace detail

template <int EXPERTS, int WARPS_PER_TB, int WARP_SIZE_PARAM,
          int MAX_BYTES_PER_LDG, typename IndType, SoftmaxMode Mode>
void topk_gating_launcher_helper(
    const float* input, const bool* finished, float* output,
    IndType* indices, const int num_rows, const int k,
    const int start_expert, const int end_expert, bool renormalize,
    const float* bias, sycl::queue& queue) {
  static constexpr int BYTES_PER_LDG =
      MIN(MAX_BYTES_PER_LDG, (int)sizeof(float) * EXPERTS);
  using C = detail::TopkConstants<EXPERTS, BYTES_PER_LDG, WARP_SIZE_PARAM>;
  const int num_warps  = (num_rows + C::ROWS_PER_WARP - 1) / C::ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  sycl::range<2> grid(1, num_blocks);
  sycl::range<2> block(WARPS_PER_TB, WARP_SIZE_PARAM);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(grid * block, block),
        TopKGating<C::VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG,
                   WARP_SIZE_PARAM, IndType, Mode>(
            input, finished, output, num_rows, indices,
            k, start_expert, end_expert, renormalize, bias));
  });
}

#define LAUNCH_TOPK_KERNEL(NUM_EXPERTS, WARPS_PER_TB, MAX_BYTES)             \
  static_assert(WARP_SIZE == 32,                                             \
                "Unsupported warp size. Only 32 is supported for XPU");      \
  topk_gating_launcher_helper<NUM_EXPERTS, WARPS_PER_TB, WARP_SIZE,          \
                              MAX_BYTES, IndType, Mode>(                     \
      gating_output, nullptr, topk_weights, topk_indices,                    \
      num_tokens, topk, 0, num_experts,                \
      renormalize, bias, queue);

template <typename IndType, SoftmaxMode Mode>
void topk_gating_kernel_launcher(
    const float* gating_output, float* topk_weights, IndType* topk_indices,
    float* scoring_workspace,
    const int num_tokens, const int num_experts, const int topk,
    const bool renormalize, const float* bias, sycl::queue& queue) {
  static constexpr int WARPS_PER_TB = 4;
  static constexpr int BYTES_PER_LDG_POWER_OF_2 = 16;
  static constexpr int BYTES_PER_LDG_MULTIPLE_64 = 2 * (int)sizeof(float);

  // ---- ESIMD fast path (IndType=int, no bias) ------------------------------
  if constexpr (std::is_same_v<IndType, int>) {
    const bool post_ok = (Mode == SoftmaxMode::Post) && (bias == nullptr);
    const bool pre_ok  = (Mode == SoftmaxMode::Pre)  && (bias == nullptr) && renormalize;
    if (post_ok || pre_ok) {
      if (esimd_fast::maybe_launch(
              queue, gating_output, topk_weights, topk_indices,
              num_tokens, num_experts, topk, Mode, renormalize)) {
        return;
      }
    }
  }

  switch (num_experts) {
    case 1:   LAUNCH_TOPK_KERNEL(1,   WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 2:   LAUNCH_TOPK_KERNEL(2,   WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 4:   LAUNCH_TOPK_KERNEL(4,   WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 8:   LAUNCH_TOPK_KERNEL(8,   WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 16:  LAUNCH_TOPK_KERNEL(16,  WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 32:  LAUNCH_TOPK_KERNEL(32,  WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 64:  LAUNCH_TOPK_KERNEL(64,  WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 128: LAUNCH_TOPK_KERNEL(128, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 256: LAUNCH_TOPK_KERNEL(256, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 512: LAUNCH_TOPK_KERNEL(512, WARPS_PER_TB, BYTES_PER_LDG_POWER_OF_2); break;
    case 192: LAUNCH_TOPK_KERNEL(192, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
    case 320: LAUNCH_TOPK_KERNEL(320, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
    case 384: LAUNCH_TOPK_KERNEL(384, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
    case 448: LAUNCH_TOPK_KERNEL(448, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
    case 576: LAUNCH_TOPK_KERNEL(576, WARPS_PER_TB, BYTES_PER_LDG_MULTIPLE_64); break;
    default: {
      TORCH_CHECK(
          scoring_workspace != nullptr,
          "scoring_workspace required for irregular num_experts.");
      static constexpr int TPB = 256;
      sycl::range<1> grid(num_tokens), block(TPB);

      if constexpr (Mode == SoftmaxMode::Pre) {
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(grid * block, block),
              MoeSoftmax<TPB>(
                  gating_output, nullptr, scoring_workspace, num_experts));
        });
        queue.submit([&](sycl::handler& cgh) {
          sycl::local_accessor<int, 1> prev_winners_slm(
              sycl::range<1>(topk > 0 ? topk : 1), cgh);
          cgh.parallel_for(
              sycl::nd_range<1>(grid * block, block),
              MoeTopK<TPB, IndType>(
                  prev_winners_slm, scoring_workspace, nullptr,
                  topk_weights, topk_indices,
                  num_experts, topk, 0, num_experts, renormalize, bias));
        });
      } else {
        queue.submit([&](sycl::handler& cgh) {
          sycl::local_accessor<int, 1> prev_winners_slm(
              sycl::range<1>(topk > 0 ? topk : 1), cgh);
          cgh.parallel_for(
              sycl::nd_range<1>(grid * block, block),
              MoeTopK<TPB, IndType>(
                  prev_winners_slm, gating_output, nullptr,
                  scoring_workspace, topk_indices,
                  num_experts, topk, 0, num_experts,
                  /*renormalize=*/false, bias));
        });
        queue.submit([&](sycl::handler& cgh) {
          cgh.parallel_for(
              sycl::nd_range<1>(grid * block, block),
              MoeSoftmax<TPB>(
                  scoring_workspace, nullptr, topk_weights, topk));
        });
      }
    }
  }
}
#undef LAUNCH_TOPK_KERNEL

// ====================== Host entry point ==================================
void moe_softmax_topk(
    torch::Tensor& topk_weights,          // [num_tokens, topk]
    torch::Tensor& topk_indices,          // [num_tokens, topk]
    torch::Tensor& gating_output,         // [num_tokens, num_experts] float
    int SoftmaxFunc,                      // 0 = pre, 1 = post
    const bool renormalize,
    std::optional<torch::Tensor> bias) {
  TORCH_CHECK(gating_output.scalar_type() == torch::kFloat,
              "gating_output must be float32");
  TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32,
              "topk_indices must be int32");
  TORCH_CHECK(SoftmaxFunc == 0 || SoftmaxFunc == 1,
              "SoftmaxFunc must be 0 (pre) or 1 (post), got ", SoftmaxFunc);

  const int num_experts = gating_output.size(-1);
  const auto num_tokens = gating_output.numel() / num_experts;
  const int topk = topk_weights.size(-1);

  const bool is_pow_2 =
      (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
  const bool needs_workspace = !is_pow_2 || num_experts > 256;
  const int64_t workspace_size =
      needs_workspace ? num_tokens * num_experts : 0;

  const at::DeviceGuard device_guard(gating_output.device());
  auto& queue = c10::xpu::getCurrentXPUStream().queue();
  torch::Tensor scoring_workspace = torch::empty(
      {workspace_size}, gating_output.options().dtype(torch::kFloat));

  const float* gp = gating_output.const_data_ptr<float>();
  float*       tw = topk_weights.data_ptr<float>();
  int*         ti = topk_indices.data_ptr<int>();
  float*       ws = workspace_size > 0 ? scoring_workspace.data_ptr<float>()
                                       : nullptr;
  const float* bp = bias.has_value() ? bias->data_ptr<float>() : nullptr;

  if (SoftmaxFunc == 0) {
    topk_gating_kernel_launcher<int, SoftmaxMode::Pre>(
        gp, tw, ti, ws, num_tokens, num_experts, topk, renormalize, bp,
        queue);
  } else {
    topk_gating_kernel_launcher<int, SoftmaxMode::Post>(
        gp, tw, ti, ws, num_tokens, num_experts, topk, renormalize, bp,
        queue);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_softmax_topk", &moe_softmax_topk,
          "Fused moe_softmax_topk (SYCL)");
}
