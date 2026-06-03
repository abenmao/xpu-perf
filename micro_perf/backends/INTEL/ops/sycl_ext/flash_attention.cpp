#include <ATen/ATen.h>
#include <torch/extension.h>

#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cutlass/util/packed_stride.hpp"
#include "benchmarks/flash_attention/fmha_configuration.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "sycl_common.hpp"

#include <c10/xpu/XPUStream.h>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace {

using namespace cute;

int cached_sm_count(int device_id) {
  static std::mutex mtx;
  static std::unordered_map<int, int> cache;
  std::lock_guard<std::mutex> lk(mtx);
  auto it = cache.find(device_id);
  if (it != cache.end()) {
    return it->second;
  }
  int sm = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device_id);
  cache.emplace(device_id, sm);
  return sm;
}

// Monotonically growing per-device workspace buffer reused across launches
// to avoid device_memory allocation on every call.
uint8_t* reusable_workspace(int device_id, size_t bytes) {
  static std::mutex mtx;
  static std::unordered_map<int, cutlass::device_memory::allocation<uint8_t>> cache;
  std::lock_guard<std::mutex> lk(mtx);
  auto& slot = cache[device_id];
  if (slot.size() < bytes) {
    slot.reset(bytes);
  }
  return slot.get();
}

struct SyclTlaExecutionPlan {
  bool use_sycl_tla = false;
  bool is_causal = false;
  bool is_decode = false;
  int64_t cache_len = 0;
  int64_t kv_new_len = 0;
};

bool has_supported_head_dim(int64_t head_dim);

bool is_sycl_tla_candidate(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    double dropout_p) {
  if (!query.is_xpu() || !key.is_xpu() || !value.is_xpu()) {
    return false;
  }
  if (query.scalar_type() != torch::kBFloat16 ||
      key.scalar_type() != torch::kBFloat16 ||
      value.scalar_type() != torch::kBFloat16) {
    return false;
  }
  if (!has_supported_head_dim(query.size(3))) {
    return false;
  }
  if (dropout_p != 0.0) {
    return false;
  }
  return true;
}

bool has_execution_plan_override(
    const std::optional<int64_t>& plan_cache_len,
    const std::optional<int64_t>& plan_kv_new_len,
    const std::optional<bool>& plan_is_causal,
    const std::optional<bool>& plan_is_decode,
    const std::optional<bool>& plan_use_sycl_tla) {
  return plan_cache_len.has_value() ||
      plan_kv_new_len.has_value() ||
      plan_is_causal.has_value() ||
      plan_is_decode.has_value() ||
      plan_use_sycl_tla.has_value();
}

std::optional<SyclTlaExecutionPlan> build_execution_plan_override(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    double dropout_p,
    const std::optional<int64_t>& plan_cache_len,
    const std::optional<int64_t>& plan_kv_new_len,
    const std::optional<bool>& plan_is_causal,
    const std::optional<bool>& plan_is_decode,
    const std::optional<bool>& plan_use_sycl_tla) {
  if (!has_execution_plan_override(
          plan_cache_len,
          plan_kv_new_len,
          plan_is_causal,
          plan_is_decode,
          plan_use_sycl_tla)) {
    return std::nullopt;
  }

  const int64_t q_len = query.size(2);
  const int64_t kv_total_len = key.size(2);

  SyclTlaExecutionPlan plan;
  plan.cache_len = plan_cache_len.value_or(0);
  plan.kv_new_len = plan_kv_new_len.value_or(kv_total_len - plan.cache_len);
  plan.is_causal = plan_is_causal.value_or(false);
  plan.is_decode = plan_is_decode.value_or(q_len <= 16);
  plan.use_sycl_tla = plan_use_sycl_tla.value_or(false);

  TORCH_CHECK(plan.cache_len >= 0, "plan_cache_len must be non-negative");
  TORCH_CHECK(plan.kv_new_len > 0, "plan_kv_new_len must be positive");
  TORCH_CHECK(
      plan.cache_len + plan.kv_new_len <= kv_total_len,
      "execution plan expects cache_len + kv_new_len <= total kv length, got cache_len=",
      plan.cache_len,
      ", kv_new_len=",
      plan.kv_new_len,
      ", kv_total_len=",
      kv_total_len);

  if (plan.use_sycl_tla) {
    TORCH_CHECK(
        is_sycl_tla_candidate(query, key, value, dropout_p),
        "provided execution plan requests sycl-tla for unsupported input configuration");
  }

  return plan;
}

struct ExternalFMHAParams {
  cutlass::bfloat16_t* query = nullptr;
  cutlass::bfloat16_t* key = nullptr;
  cutlass::bfloat16_t* value = nullptr;
  cutlass::bfloat16_t* key_cache = nullptr;
  cutlass::bfloat16_t* value_cache = nullptr;
  // Output buffer; the kernel-side cast in make_kernel_arguments picks the
  // matching element type based on the FMHAConfig selected by the wrapper.
  void* output = nullptr;
  int batch = 0;
  int num_heads_q = 0;
  int num_heads_kv = 0;
  int seq_len_q = 0;
  int seq_len_kv = 0;
  int seq_len_kv_cache = 0;
  // Total kv length of the parent K/V tensor backing key/value/key_cache/
  // value_cache. Used to derive head/batch strides without forcing a
  // .contiguous() copy on the cached path. When 0, fall back to seq_len_kv.
  int parent_kv_total = 0;
  int head_dim = 0;
  float softmax_scale = 0.0f;
  int device_id = 0;
};

struct SyclTlaLaunchContext {
  torch::Tensor query_contig;
  torch::Tensor key_contig;
  torch::Tensor value_contig;
  torch::Tensor output_tensor;
  ExternalFMHAParams ext;
};

template <cutlass::flash_attention::FMHAMode Mode,
          class ElementQ,
          class ElementK,
          class ElementV,
          class ElementO,
          bool Causal,
          bool CachedKV,
          int HeadDim>
struct FMHAConfigSelector {
  using type = typename cutlass::flash_attention::FMHAConfigGen<
      Mode,
      ElementQ,
      ElementK,
      ElementV,
      ElementO,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      float,
      Causal,
      false,
      CachedKV,
      false,
      false,
      false,
      HeadDim>::type;
};
// NOTE: For HeadDim=128 prefill on BMG, the sycl-tla 06 example binary uses a
// hand-tuned tile config ShapeQK<256,32,32> with PipelineStages=2 and an
// auto-derived SubgroupLayoutPV, which is ~36% faster than the default
// ShapeConfig<Prefill,128> = <128,64,32> picked by FMHAConfigGen here.
//
// We specialize FMHAConfigSelector for HeadDim=128 prefill to directly
// instantiate cutlass::flash_attention::FMHAConfig with 06's hand-tuned
// tiles and SubgroupLayoutPV_ = void (auto-derive). This avoids
// FMHAConfigGenWithTileShape, which explicitly constructs SubgroupLayoutPV
// from the tile ratios and ends up with a slower kernel.
template <class ElementQ, class ElementK, class ElementV, class ElementO,
          bool Causal, bool CachedKV>
struct FMHAConfigSelector<cutlass::flash_attention::FMHAMode::Prefill,
                          ElementQ, ElementK, ElementV, ElementO,
                          Causal, CachedKV, 128> {
  using type = cutlass::flash_attention::FMHAConfig<
      ElementQ, ElementK, ElementV, ElementO,
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor,
      /*ElementScale=*/float,
      /*TileShapeQK=*/cute::Shape<cute::_256, cute::_32, cute::_32>,
      /*TileShapePV=*/cute::Shape<cute::_256, cute::_32, cute::_32>,
      /*TileShapeOutput=*/cute::Shape<cute::_256, cute::_128>,
      /*SubgroupLayoutQK=*/cute::Layout<cute::Shape<cute::_16, cute::_1, cute::_1>>,
      /*SubgroupLayoutPV_=*/void,
      Causal,
      /*VarLen=*/false,
      CachedKV,
      /*PagedKV=*/false,
      /*Persistent=*/false,
      /*UseScale=*/false,
      /*PipelineStages=*/2>;
};

template <class KernelArguments, class Enable = void>
struct KernelArgumentScaleSetter {
  static void apply(KernelArguments&) {}
};

template <class KernelArguments>
struct KernelArgumentScaleSetter<
    KernelArguments,
    std::void_t<
        decltype(std::declval<KernelArguments&>().scaleQ),
        decltype(std::declval<KernelArguments&>().dScaleQ),
        decltype(std::declval<KernelArguments&>().scaleK),
        decltype(std::declval<KernelArguments&>().dScaleK),
        decltype(std::declval<KernelArguments&>().scaleV),
        decltype(std::declval<KernelArguments&>().dScaleV),
        decltype(std::declval<KernelArguments&>().scale_k),
        decltype(std::declval<KernelArguments&>().scale_v),
        decltype(std::declval<KernelArguments&>().group_size)>> {
  static void apply(KernelArguments& args) {
    args.scaleQ = nullptr;
    args.dScaleQ = {};
    args.scaleK = nullptr;
    args.dScaleK = {};
    args.scaleV = nullptr;
    args.dScaleV = {};
    args.scale_k = 1.0f;
    args.scale_v = 1.0f;
    args.group_size = 32;
  }
};

template <class ProblemShapeType>
ProblemShapeType make_problem_shape(const ExternalFMHAParams& ext) {
  ProblemShapeType problem_shape{};
  problem_shape.batch = ext.batch;
  problem_shape.num_heads_q = ext.num_heads_q;
  problem_shape.num_heads_kv = ext.num_heads_kv;
  problem_shape.seq_len_qo = ext.seq_len_q;
  problem_shape.seq_len_kv = ext.seq_len_kv;
  problem_shape.seq_len_kv_cache = ext.seq_len_kv_cache;
  problem_shape.head_size_qk = ext.head_dim;
  problem_shape.head_size_vo = ext.head_dim;
  return problem_shape;
}

template <class FMHAKernel, class ProblemShapeType, class StrideQ, class StrideK, class StrideV, class StrideO>
typename FMHAKernel::KernelArguments make_kernel_arguments(
    const ExternalFMHAParams& ext,
    const ProblemShapeType& problem_shape,
    const StrideQ& stride_q,
    const StrideK& stride_k,
    const StrideV& stride_v,
    const StrideO& stride_o,
    const StrideK& stride_k_cache,
    const StrideV& stride_v_cache) {
  typename FMHAKernel::KernelArguments kernel_args{};
  kernel_args.shape = problem_shape;
  kernel_args.Q = ext.query;
  kernel_args.dQ = stride_q;
  kernel_args.K = ext.key;
  kernel_args.dK = stride_k;
  kernel_args.V = ext.value;
  kernel_args.dV = stride_v;
  kernel_args.O = static_cast<decltype(kernel_args.O)>(ext.output);
  kernel_args.dO = stride_o;
  KernelArgumentScaleSetter<typename FMHAKernel::KernelArguments>::apply(kernel_args);
  kernel_args.K_cache = ext.key_cache;
  kernel_args.dK_cache = stride_k_cache;
  kernel_args.V_cache = ext.value_cache;
  kernel_args.dV_cache = stride_v_cache;
  return kernel_args;
}

template <class FMHAKernel>
typename FMHAKernel::Arguments make_fmha_arguments(
    const typename FMHAKernel::KernelArguments& kernel_args,
    float softmax_scale,
    const cutlass::KernelHardwareInfo& hw_info) {
  return typename FMHAKernel::Arguments{
      kernel_args,
      {softmax_scale, nullptr, 0, nullptr},
      {},
      hw_info,
  };
}

void check_attention_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  TORCH_CHECK(query.dim() == 4, "query must be a 4D tensor [batch, heads, q_len, head_dim]");
  TORCH_CHECK(key.dim() == 4, "key must be a 4D tensor [batch, heads, kv_len, head_dim]");
  TORCH_CHECK(value.dim() == 4, "value must be a 4D tensor [batch, heads, kv_len, head_dim]");

  TORCH_CHECK(query.device() == key.device(), "query and key must be on the same device");
  TORCH_CHECK(query.device() == value.device(), "query and value must be on the same device");
  TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query and key must have the same dtype");
  TORCH_CHECK(key.scalar_type() == value.scalar_type(), "key and value must have the same dtype");

  TORCH_CHECK(query.size(0) == key.size(0), "query and key batch size must match");
  TORCH_CHECK(query.size(0) == value.size(0), "query and value batch size must match");
  TORCH_CHECK(key.size(2) == value.size(2), "key/value sequence length must match");
  TORCH_CHECK(key.size(3) == value.size(3), "key/value head_dim must match");
  TORCH_CHECK(query.size(3) == key.size(3), "query/key head_dim must match");
}

torch::Tensor expand_gqa_heads(const torch::Tensor& tensor, int64_t target_heads) {
  const int64_t source_heads = tensor.size(1);
  TORCH_CHECK(source_heads > 0, "source head count must be positive");
  TORCH_CHECK(
      target_heads % source_heads == 0,
      "enable_gqa requires query heads to be divisible by key/value heads, got query_heads=",
      target_heads,
      ", kv_heads=",
      source_heads);

  if (source_heads == target_heads) {
    return tensor;
  }

  return tensor.repeat_interleave(target_heads / source_heads, 1);
}

torch::Tensor build_causal_mask(
    int64_t q_len,
    int64_t kv_len,
    const c10::Device& device) {
  auto options = torch::TensorOptions().dtype(torch::kLong).device(device);
  auto q_idx = torch::arange(q_len, options).unsqueeze(1);
  auto kv_idx = torch::arange(kv_len, options).unsqueeze(0);

  const int64_t diagonal = kv_len - q_len;
  return kv_idx <= (q_idx + diagonal);
}

torch::Tensor apply_attention_mask(
    const torch::Tensor& scores,
    const torch::Tensor& attn_mask) {
  auto mask = attn_mask;
  if (mask.device() != scores.device()) {
    mask = mask.to(scores.device());
  }

  if (mask.scalar_type() == torch::kBool) {
    const auto neg_inf = -std::numeric_limits<float>::infinity();
    return scores.masked_fill(mask.logical_not(), neg_inf);
  }

  return scores + mask.to(scores.scalar_type());
}

torch::Tensor scaled_dot_product_attention_fallback(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    std::optional<torch::Tensor> attn_mask,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  const double actual_scale = scale.has_value()
      ? *scale
      : 1.0 / std::sqrt(static_cast<double>(query.size(3)));

  auto scores = torch::matmul(query, key.transpose(-2, -1)) * actual_scale;

  if (is_causal) {
    auto causal_mask = build_causal_mask(query.size(2), key.size(2), scores.device());
    const auto neg_inf = -std::numeric_limits<float>::infinity();
    scores = scores.masked_fill(causal_mask.logical_not(), neg_inf);
  }

  if (attn_mask.has_value()) {
    scores = apply_attention_mask(scores, *attn_mask);
  }

  auto attn = torch::softmax(scores, -1);
  if (dropout_p > 0.0) {
    attn = torch::dropout(attn, dropout_p, true);
  }

  return torch::matmul(attn, value);
}

bool has_supported_head_dim(int64_t head_dim) {
  return head_dim == 64 || head_dim == 96 || head_dim == 128 || head_dim == 192;
}

torch::Tensor materialize_2d_mask(const torch::Tensor& attn_mask, int64_t q_len, int64_t kv_len) {
  TORCH_CHECK(attn_mask.dim() >= 2, "attn_mask must have at least 2 dimensions");
  TORCH_CHECK(attn_mask.size(-2) == q_len, "attn_mask q_len mismatch");
  TORCH_CHECK(attn_mask.size(-1) == kv_len, "attn_mask kv_len mismatch");

  for (int64_t dim = 0; dim < attn_mask.dim() - 2; ++dim) {
    TORCH_CHECK(
        attn_mask.size(dim) == 1,
        "Only broadcastable singleton leading attn_mask dimensions are supported in sycl_tla path");
  }

  auto mask_2d = attn_mask;
  while (mask_2d.dim() > 2) {
    mask_2d = mask_2d.select(0, 0);
  }
  return mask_2d;
}

bool is_lower_right_causal_mask(const torch::Tensor& attn_mask, int64_t q_len, int64_t kv_len) {
  if (attn_mask.scalar_type() != torch::kBool) {
    return false;
  }

  torch::Tensor mask_2d;
  try {
    mask_2d = materialize_2d_mask(attn_mask, q_len, kv_len);
  } catch (const c10::Error&) {
    return false;
  }

  auto expected = build_causal_mask(q_len, kv_len, c10::Device(torch::kCPU));
  auto host_mask = mask_2d.to(torch::kCPU);
  return host_mask.equal(expected);
}

SyclTlaExecutionPlan build_execution_plan(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal) {
  SyclTlaExecutionPlan plan;

  if (!is_sycl_tla_candidate(query, key, value, dropout_p)) {
    return plan;
  }

  const int64_t q_len = query.size(2);
  const int64_t kv_total_len = key.size(2);

  plan.kv_new_len = kv_total_len;
  plan.is_causal = false;

  if (attn_mask.has_value()) {
    if (!is_lower_right_causal_mask(*attn_mask, q_len, kv_total_len)) {
      return plan;
    }
    TORCH_CHECK(kv_total_len >= q_len, "lower-right causal mask requires kv_len >= q_len");
    plan.cache_len = kv_total_len - q_len;
    plan.kv_new_len = q_len;
    plan.is_causal = true;
  } else if (q_len == 1) {
    plan.cache_len = std::max<int64_t>(0, kv_total_len - 1);
    plan.kv_new_len = kv_total_len - plan.cache_len;
    plan.is_causal = false;
  } else if (is_causal) {
    if (q_len != kv_total_len) {
      return plan;
    }
    plan.cache_len = 0;
    plan.kv_new_len = kv_total_len;
    plan.is_causal = true;
  } else {
    plan.cache_len = 0;
    plan.kv_new_len = kv_total_len;
    plan.is_causal = false;
  }

  if (plan.kv_new_len <= 0) {
    return plan;
  }

  // Route through the Decode kernel whenever the q dimension is small. The
  // Decode kernel's q-tile is 1 (loops over q rows) with a large kv-tile (512),
  // which is the right shape for short-q + large-cache cases. The Prefill
  // kernel's 256-row q-tile wastes most of its q lanes when q_len is tiny
  // (e.g. q=4 over a 10k-row cache is ~64x over-compute in the q dim). The
  // threshold of 16 is conservative: below it, Decode reliably wins on BMG
  // for the head_dim=128 shapes we benchmark; above it, Prefill's wider tile
  // amortizes K/V loads better.
  plan.is_decode = (q_len <= 16);
  // The CachedKV dispatch path is wired in run_sycl_tla_bf16 (K_cache/V_cache
  // kernel args, dedicated FMHAConfigSelector instantiations). Enable it for
  // both prefill and decode shapes.
  plan.use_sycl_tla = true;
  return plan;
}

torch::Tensor prepare_output_tensor(
    const torch::Tensor& query,
    const std::optional<torch::Tensor>& output,
    bool fp32_output) {
  const auto expected_dtype = fp32_output ? torch::kFloat32 : query.scalar_type();

  if (output.has_value()) {
    auto out = *output;
    TORCH_CHECK(out.device() == query.device(), "output must be on the same device as query");
    TORCH_CHECK(out.scalar_type() == expected_dtype, "output dtype mismatch");
    TORCH_CHECK(out.sizes().vec() == query.sizes().vec(), "output shape mismatch");
    TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
    return out;
  }

  return fp32_output
      ? torch::empty(query.sizes(), query.options().dtype(torch::kFloat32))
      : torch::empty_like(query);
}

void check_scaled_dot_product_attention_arguments(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    bool enable_gqa) {
  check_attention_inputs(query, key, value);
  TORCH_CHECK(dropout_p >= 0.0 && dropout_p < 1.0, "dropout_p must be in [0, 1)");
  TORCH_CHECK(
      !(attn_mask.has_value() && is_causal),
      "attn_mask and is_causal cannot both be set");

  if (!enable_gqa) {
    TORCH_CHECK(
        query.size(1) == key.size(1),
        "query and key head count must match unless enable_gqa=True");
    TORCH_CHECK(
        query.size(1) == value.size(1),
        "query and value head count must match unless enable_gqa=True");
  } else {
    TORCH_CHECK(
        query.size(1) % key.size(1) == 0,
        "enable_gqa requires query heads to be divisible by key/value heads");
    TORCH_CHECK(
        key.size(1) == value.size(1),
        "key and value head count must match for enable_gqa=True");
  }
}

SyclTlaExecutionPlan resolve_execution_plan(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& attn_mask,
    double dropout_p,
    bool is_causal,
    const std::optional<int64_t>& plan_cache_len,
    const std::optional<int64_t>& plan_kv_new_len,
    const std::optional<bool>& plan_is_causal,
    const std::optional<bool>& plan_is_decode,
    const std::optional<bool>& plan_use_sycl_tla) {
  auto plan_override = build_execution_plan_override(
      query,
      key,
      value,
      dropout_p,
      plan_cache_len,
      plan_kv_new_len,
      plan_is_causal,
      plan_is_decode,
      plan_use_sycl_tla);
  return plan_override.has_value()
      ? *plan_override
      : build_execution_plan(query, key, value, attn_mask, dropout_p, is_causal);
}

bool resolve_fp32_output(
    const SyclTlaExecutionPlan& plan,
    int64_t head_dim,
    const std::optional<std::string>& output_dtype) {
  const bool fp32_output_supported =
      plan.use_sycl_tla && !plan.is_decode && head_dim == 128;

  bool fp32_output = fp32_output_supported;
  if (output_dtype.has_value()) {
    const auto& d = *output_dtype;
    if (d == "float32" || d == "fp32") {
      TORCH_CHECK(
          fp32_output_supported,
          "output_dtype='float32' is currently only supported for prefill with head_dim=128 on the sycl-tla path");
      fp32_output = true;
    } else if (d == "bfloat16" || d == "bf16") {
      fp32_output = false;
    } else {
      TORCH_CHECK(
          false,
          "sycl_ext flash_attention output_dtype must be one of {bfloat16, float32}, got: ",
          d);
    }
  }

  return fp32_output;
}

float resolve_softmax_scale(
    const torch::Tensor& query,
    const std::optional<double>& scale) {
  return static_cast<float>(
      scale.has_value()
          ? *scale
          : 1.0 / std::sqrt(static_cast<double>(query.size(3))));
}

SyclTlaLaunchContext build_sycl_tla_launch_context(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const SyclTlaExecutionPlan& plan,
    const std::optional<double>& scale,
    bool fp32_output,
    const std::optional<torch::Tensor>& output) {
  SyclTlaLaunchContext context;
  context.query_contig = query.is_contiguous() ? query : query.contiguous();
  context.key_contig = key.is_contiguous() ? key : key.contiguous();
  context.value_contig = value.is_contiguous() ? value : value.contiguous();
  context.output_tensor = prepare_output_tensor(context.query_contig, output, fp32_output);

  const int64_t parent_kv_total = context.key_contig.size(2);
  const int64_t head_dim = context.query_contig.size(3);
  auto* key_base = reinterpret_cast<cutlass::bfloat16_t*>(
      context.key_contig.data_ptr<at::BFloat16>());
  auto* value_base = reinterpret_cast<cutlass::bfloat16_t*>(
      context.value_contig.data_ptr<at::BFloat16>());
  auto* key_new_ptr = key_base;
  auto* value_new_ptr = value_base;
  cutlass::bfloat16_t* key_cache_ptr = nullptr;
  cutlass::bfloat16_t* value_cache_ptr = nullptr;

  if (plan.cache_len > 0) {
    key_cache_ptr = key_base;
    value_cache_ptr = value_base;
    key_new_ptr = key_base + plan.cache_len * head_dim;
    value_new_ptr = value_base + plan.cache_len * head_dim;
  }

  context.ext.query = reinterpret_cast<cutlass::bfloat16_t*>(
      context.query_contig.data_ptr<at::BFloat16>());
  context.ext.key = key_new_ptr;
  context.ext.value = value_new_ptr;
  context.ext.key_cache = key_cache_ptr;
  context.ext.value_cache = value_cache_ptr;
  context.ext.output = fp32_output
      ? static_cast<void*>(context.output_tensor.data_ptr<float>())
      : static_cast<void*>(context.output_tensor.data_ptr<at::BFloat16>());
  context.ext.batch = static_cast<int>(context.query_contig.size(0));
  context.ext.num_heads_q = static_cast<int>(context.query_contig.size(1));
  context.ext.num_heads_kv = static_cast<int>(context.key_contig.size(1));
  context.ext.seq_len_q = static_cast<int>(context.query_contig.size(2));
  context.ext.seq_len_kv = static_cast<int>(plan.kv_new_len);
  context.ext.seq_len_kv_cache = static_cast<int>(plan.cache_len);
  context.ext.parent_kv_total = static_cast<int>(parent_kv_total);
  context.ext.head_dim = static_cast<int>(head_dim);
  context.ext.softmax_scale = resolve_softmax_scale(context.query_contig, scale);
  context.ext.device_id = context.query_contig.get_device();

  return context;
}

std::vector<torch::Tensor> build_keepalive_tensors(
    const SyclTlaLaunchContext& context) {
  return {
      context.query_contig,
      context.key_contig,
      context.value_contig,
      context.output_tensor,
  };
}

template <class FMHAKernel>
void launch_fmha(typename FMHAKernel::Params params, sycl::queue q) {
  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  dim3 const block = FMHAKernel::get_block_shape();
  dim3 const grid = FMHAKernel::get_grid_shape(params);
  int smem_size = FMHAKernel::SharedStorageSize;

  const auto sycl_block = compat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

  compat::experimental::launch_properties launch_props {
    syclex::work_group_scratch_size(smem_size),
  };
  compat::experimental::kernel_properties kernel_props {
    syclex::sub_group_size<cute::intel::sg_size>,
#if (SYCL_INTEL_TARGET == 35)
    intelex::grf_size<512>
#else
    intelex::grf_size<256>
#endif
  };
  compat::experimental::launch_policy policy {sycl_grid, sycl_block, launch_props, kernel_props};
  auto event = compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, FMHAKernel>(policy, q, params);
  EventManager::getInstance().addEvent(event);
}

struct PreparedScaledDotProductAttention {
  virtual ~PreparedScaledDotProductAttention() = default;
  virtual void run(sycl::queue& q) = 0;
  virtual torch::Tensor output() const = 0;
  virtual int device_id() const = 0;
};

template <class FMHAKernel>
struct PreparedScaledDotProductAttentionImpl : PreparedScaledDotProductAttention {
  PreparedScaledDotProductAttentionImpl(
      const typename FMHAKernel::Arguments& arguments,
      std::vector<torch::Tensor> keepalive,
      torch::Tensor output_tensor,
      int device_id)
      : keepalive_(std::move(keepalive)),
        output_tensor_(std::move(output_tensor)),
        device_id_(device_id) {
    const size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    if (workspace_size > 0) {
      workspace_.reset(workspace_size);
    }
    uint8_t* workspace_ptr = workspace_size > 0 ? workspace_.get() : nullptr;

    auto status = FMHAKernel::initialize_workspace(arguments, workspace_ptr);
    TORCH_CHECK(
        status == cutlass::Status::kSuccess,
        "sycl-tla flash attention workspace initialization failed with status code ",
        static_cast<int>(status));

    params_.emplace(FMHAKernel::to_underlying_arguments(arguments, workspace_ptr));
  }

  void run(sycl::queue& q) override {
    TORCH_CHECK(params_.has_value(), "prepared flash attention params not initialized");
    launch_fmha<FMHAKernel>(*params_, q);
  }

  torch::Tensor output() const override {
    return output_tensor_;
  }

  int device_id() const override {
    return device_id_;
  }

 private:
  std::optional<typename FMHAKernel::Params> params_;
  cutlass::device_memory::allocation<uint8_t> workspace_;
  std::vector<torch::Tensor> keepalive_;
  torch::Tensor output_tensor_;
  int device_id_ = 0;
};

template <class ElementO,
          bool Causal,
          bool CachedKV,
          cutlass::flash_attention::FMHAMode Mode,
          int HeadDim>
std::shared_ptr<PreparedScaledDotProductAttention> prepare_sycl_tla_bf16(
    const ExternalFMHAParams& ext,
    const torch::Tensor& output_tensor,
    std::vector<torch::Tensor> keepalive) {
  using ElementQ = cutlass::bfloat16_t;
  using ElementK = cutlass::bfloat16_t;
  using ElementV = cutlass::bfloat16_t;
  using FMHAConfiguration = typename FMHAConfigSelector<
      Mode,
      ElementQ,
      ElementK,
      ElementV,
      ElementO,
      Causal,
      CachedKV,
      HeadDim>::type;
  using FMHAKernel = typename FMHAConfiguration::FMHAKernel;
  using ProblemShapeType = typename FMHAConfiguration::ProblemShapeType;
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = ext.device_id;
  hw_info.sm_count = cached_sm_count(ext.device_id);

  auto shape_q = cute::make_shape(ext.seq_len_q, ext.head_dim, ext.num_heads_q, ext.batch);
  const int parent_kv_total = ext.parent_kv_total > 0 ? ext.parent_kv_total : ext.seq_len_kv;
  auto shape_k_stride = cute::make_shape(parent_kv_total, ext.head_dim, ext.num_heads_kv, ext.batch);
  auto shape_v_stride = cute::make_shape(ext.head_dim, parent_kv_total, ext.num_heads_kv, ext.batch);
  auto shape_o = cute::make_shape(ext.seq_len_q, ext.head_dim, ext.num_heads_q, ext.batch);

  auto stride_q = cutlass::make_cute_packed_stride(StrideQ{}, shape_q);
  auto stride_k = cutlass::make_cute_packed_stride(StrideK{}, shape_k_stride);
  auto stride_v = cutlass::make_cute_packed_stride(StrideV{}, shape_v_stride);
  auto stride_k_cache = stride_k;
  auto stride_v_cache = stride_v;
  auto stride_o = cutlass::make_cute_packed_stride(StrideO{}, shape_o);

  auto problem_shape = make_problem_shape<ProblemShapeType>(ext);
  auto kernel_args = make_kernel_arguments<FMHAKernel>(
      ext,
      problem_shape,
      stride_q,
      stride_k,
      stride_v,
      stride_o,
      stride_k_cache,
      stride_v_cache);
  auto arguments = make_fmha_arguments<FMHAKernel>(kernel_args, ext.softmax_scale, hw_info);

  TORCH_CHECK(
      FMHAKernel::can_implement(arguments),
      "sycl-tla flash attention cannot implement the requested problem");

  return std::make_shared<PreparedScaledDotProductAttentionImpl<FMHAKernel>>(
      arguments,
      std::move(keepalive),
      output_tensor,
      ext.device_id);
}

template <class ElementO,
          bool Causal,
          bool CachedKV,
          cutlass::flash_attention::FMHAMode Mode,
          int HeadDim>
cutlass::Status run_sycl_tla_bf16(const ExternalFMHAParams& ext) {
  using ElementQ = cutlass::bfloat16_t;
  using ElementK = cutlass::bfloat16_t;
  using ElementV = cutlass::bfloat16_t;
  using FMHAConfiguration = typename FMHAConfigSelector<
      Mode,
      ElementQ,
      ElementK,
      ElementV,
      ElementO,
      Causal,
      CachedKV,
      HeadDim>::type;
  using FMHAKernel = typename FMHAConfiguration::FMHAKernel;
  using ProblemShapeType = typename FMHAConfiguration::ProblemShapeType;
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = ext.device_id;
  hw_info.sm_count = cached_sm_count(ext.device_id);

  auto shape_q = cute::make_shape(ext.seq_len_q, ext.head_dim, ext.num_heads_q, ext.batch);
  auto shape_o = cute::make_shape(ext.seq_len_q, ext.head_dim, ext.num_heads_q, ext.batch);

  // K/V layouts use the parent buffer's kv-total dim to compute head/batch
  // strides when present, so the wrapper can pass pointer offsets into a
  // shared [b, h, kv_total, d] tensor without making a .contiguous() copy of
  // each slice. parent_kv_total == 0 means "data is packed at seq_len_kv".
  const int parent_kv_total = ext.parent_kv_total > 0 ? ext.parent_kv_total : ext.seq_len_kv;
  auto shape_k_stride = cute::make_shape(parent_kv_total, ext.head_dim, ext.num_heads_kv, ext.batch);
  auto shape_v_stride = cute::make_shape(ext.head_dim, parent_kv_total, ext.num_heads_kv, ext.batch);

  auto stride_q = cutlass::make_cute_packed_stride(StrideQ{}, shape_q);
  auto stride_k = cutlass::make_cute_packed_stride(StrideK{}, shape_k_stride);
  auto stride_v = cutlass::make_cute_packed_stride(StrideV{}, shape_v_stride);
  // K_cache / V_cache share the same parent buffer (and thus the same head/
  // batch strides) as K / V. The kernel reads only seq_len_kv_cache rows.
  auto stride_k_cache = stride_k;
  auto stride_v_cache = stride_v;
  auto stride_o = cutlass::make_cute_packed_stride(StrideO{}, shape_o);

  auto problem_shape = make_problem_shape<ProblemShapeType>(ext);
  auto kernel_args = make_kernel_arguments<FMHAKernel>(
      ext,
      problem_shape,
      stride_q,
      stride_k,
      stride_v,
      stride_o,
      stride_k_cache,
      stride_v_cache);
  auto arguments = make_fmha_arguments<FMHAKernel>(kernel_args, ext.softmax_scale, hw_info);

  if (!FMHAKernel::can_implement(arguments)) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
  uint8_t* workspace_ptr = workspace_size > 0
      ? reusable_workspace(ext.device_id, workspace_size)
      : nullptr;

  auto status = FMHAKernel::initialize_workspace(arguments, workspace_ptr);
  if (status != cutlass::Status::kSuccess) {
    return status;
  }

  auto params = FMHAKernel::to_underlying_arguments(arguments, workspace_ptr);
  sycl::queue& torch_q = c10::xpu::getCurrentXPUStream(ext.device_id).queue();
  launch_fmha<FMHAKernel>(params, torch_q);
  // Submit on PyTorch's XPU stream so that torch.xpu.synchronize() and the
  // surrounding tensor lifetime tracking observe this launch. No host-side
  // wait here; the caller is responsible for synchronization.
  return cutlass::Status::kSuccess;
}

template <class ElementO, bool Causal>
cutlass::Status dispatch_prefill_bf16(int64_t head_dim, const ExternalFMHAParams& ext) {
  if (head_dim == 64) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 64>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 64>(ext);
  }
  if (head_dim == 96) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 96>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 96>(ext);
  }
  if (head_dim == 128) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 128>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 128>(ext);
  }
  if (head_dim == 192) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 192>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 192>(ext);
  }
  return cutlass::Status::kErrorInvalidProblem;
}

template <class ElementO, bool Causal>
std::shared_ptr<PreparedScaledDotProductAttention> dispatch_prepare_prefill_bf16(
    int64_t head_dim,
    const ExternalFMHAParams& ext,
    const torch::Tensor& output_tensor,
    std::vector<torch::Tensor> keepalive) {
  if (head_dim == 64) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 64>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 64>(ext, output_tensor, std::move(keepalive));
  }
  if (head_dim == 96) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 96>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 96>(ext, output_tensor, std::move(keepalive));
  }
  if (head_dim == 128) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 128>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 128>(ext, output_tensor, std::move(keepalive));
  }
  if (head_dim == 192) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Prefill, 192>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Prefill, 192>(ext, output_tensor, std::move(keepalive));
  }
  TORCH_CHECK(false, "unsupported head_dim for prefill prepare path");
  return nullptr;
}

template <class ElementO, bool Causal>
cutlass::Status dispatch_decode_bf16(int64_t head_dim, const ExternalFMHAParams& ext) {
  if (head_dim == 64) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 64>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 64>(ext);
  }
  if (head_dim == 96) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 96>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 96>(ext);
  }
  if (head_dim == 128) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 128>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 128>(ext);
  }
  if (head_dim == 192) {
    return ext.seq_len_kv_cache > 0
        ? run_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 192>(ext)
        : run_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 192>(ext);
  }
  return cutlass::Status::kErrorInvalidProblem;
}

template <class ElementO, bool Causal>
std::shared_ptr<PreparedScaledDotProductAttention> dispatch_prepare_decode_bf16(
    int64_t head_dim,
    const ExternalFMHAParams& ext,
    const torch::Tensor& output_tensor,
    std::vector<torch::Tensor> keepalive) {
  if (head_dim == 64) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 64>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 64>(ext, output_tensor, std::move(keepalive));
  }
  if (head_dim == 96) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 96>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 96>(ext, output_tensor, std::move(keepalive));
  }
  if (head_dim == 128) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 128>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 128>(ext, output_tensor, std::move(keepalive));
  }
  if (head_dim == 192) {
    return ext.seq_len_kv_cache > 0
        ? prepare_sycl_tla_bf16<ElementO, Causal, true, cutlass::flash_attention::FMHAMode::Decode, 192>(ext, output_tensor, std::move(keepalive))
        : prepare_sycl_tla_bf16<ElementO, Causal, false, cutlass::flash_attention::FMHAMode::Decode, 192>(ext, output_tensor, std::move(keepalive));
  }
  TORCH_CHECK(false, "unsupported head_dim for decode prepare path");
  return nullptr;
}

template <class ElementO>
cutlass::Status dispatch_sycl_tla_bf16(int64_t head_dim, const SyclTlaExecutionPlan& plan, const ExternalFMHAParams& ext) {
  if (plan.is_decode) {
    return plan.is_causal ? dispatch_decode_bf16<ElementO, true>(head_dim, ext) : dispatch_decode_bf16<ElementO, false>(head_dim, ext);
  }
  return plan.is_causal ? dispatch_prefill_bf16<ElementO, true>(head_dim, ext) : dispatch_prefill_bf16<ElementO, false>(head_dim, ext);
}

template <class ElementO>
std::shared_ptr<PreparedScaledDotProductAttention> dispatch_prepare_sycl_tla_bf16(
    int64_t head_dim,
    const SyclTlaExecutionPlan& plan,
    const ExternalFMHAParams& ext,
    const torch::Tensor& output_tensor,
    std::vector<torch::Tensor> keepalive) {
  if (plan.is_decode) {
    return plan.is_causal
        ? dispatch_prepare_decode_bf16<ElementO, true>(head_dim, ext, output_tensor, std::move(keepalive))
        : dispatch_prepare_decode_bf16<ElementO, false>(head_dim, ext, output_tensor, std::move(keepalive));
  }
  return plan.is_causal
      ? dispatch_prepare_prefill_bf16<ElementO, true>(head_dim, ext, output_tensor, std::move(keepalive))
      : dispatch_prepare_prefill_bf16<ElementO, false>(head_dim, ext, output_tensor, std::move(keepalive));
}

torch::Tensor scaled_dot_product_attention_sycl_tla(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const SyclTlaExecutionPlan& plan,
    std::optional<double> scale,
    bool fp32_output,
    const std::optional<torch::Tensor>& output) {
  auto context = build_sycl_tla_launch_context(
      query,
      key,
      value,
      plan,
      scale,
      fp32_output,
      output);

  auto status = fp32_output
      ? dispatch_sycl_tla_bf16<float>(context.ext.head_dim, plan, context.ext)
      : dispatch_sycl_tla_bf16<cutlass::bfloat16_t>(context.ext.head_dim, plan, context.ext);
  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "sycl-tla flash attention launch failed with status code ",
      static_cast<int>(status));
  return context.output_tensor;
}

std::shared_ptr<PreparedScaledDotProductAttention> prepare_scaled_dot_product_attention_sycl_tla(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const SyclTlaExecutionPlan& plan,
    std::optional<double> scale,
    bool fp32_output,
    const std::optional<torch::Tensor>& output) {
  auto context = build_sycl_tla_launch_context(
    query,
    key,
    value,
    plan,
    scale,
    fp32_output,
    output);
  auto keepalive = build_keepalive_tensors(context);

  return fp32_output
    ? dispatch_prepare_sycl_tla_bf16<float>(context.ext.head_dim, plan, context.ext, context.output_tensor, std::move(keepalive))
    : dispatch_prepare_sycl_tla_bf16<cutlass::bfloat16_t>(context.ext.head_dim, plan, context.ext, context.output_tensor, std::move(keepalive));
}

} // namespace

std::shared_ptr<PreparedScaledDotProductAttention> prepare_scaled_dot_product_attention_sycl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    std::optional<torch::Tensor> attn_mask = std::nullopt,
    double dropout_p = 0.0,
    bool is_causal = false,
    std::optional<double> scale = std::nullopt,
    bool enable_gqa = false,
    std::optional<std::string> output_dtype = std::nullopt,
    std::optional<torch::Tensor> output = std::nullopt,
    std::optional<int64_t> plan_cache_len = std::nullopt,
    std::optional<int64_t> plan_kv_new_len = std::nullopt,
    std::optional<bool> plan_is_causal = std::nullopt,
    std::optional<bool> plan_is_decode = std::nullopt,
    std::optional<bool> plan_use_sycl_tla = std::nullopt) {
  check_scaled_dot_product_attention_arguments(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      enable_gqa);

  auto plan = resolve_execution_plan(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      plan_cache_len,
      plan_kv_new_len,
      plan_is_causal,
      plan_is_decode,
      plan_use_sycl_tla);

  TORCH_CHECK(plan.use_sycl_tla, "prepare_scaled_dot_product_attention currently only supports the sycl-tla path");

  auto fp32_output = resolve_fp32_output(plan, query.size(3), output_dtype);

  return prepare_scaled_dot_product_attention_sycl_tla(query, key, value, plan, scale, fp32_output, output);
}

torch::Tensor run_prepared_scaled_dot_product_attention(
    const std::shared_ptr<PreparedScaledDotProductAttention>& prepared) {
  TORCH_CHECK(prepared != nullptr, "prepared flash attention handle must not be null");
  sycl::queue& torch_q = c10::xpu::getCurrentXPUStream(prepared->device_id()).queue();
  prepared->run(torch_q);
  return prepared->output();
}

torch::Tensor scaled_dot_product_attention_sycl(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    std::optional<torch::Tensor> attn_mask = std::nullopt,
    double dropout_p = 0.0,
    bool is_causal = false,
    std::optional<double> scale = std::nullopt,
    bool enable_gqa = false,
    std::optional<std::string> output_dtype = std::nullopt,
    std::optional<torch::Tensor> output = std::nullopt,
    std::optional<int64_t> plan_cache_len = std::nullopt,
    std::optional<int64_t> plan_kv_new_len = std::nullopt,
    std::optional<bool> plan_is_causal = std::nullopt,
    std::optional<bool> plan_is_decode = std::nullopt,
    std::optional<bool> plan_use_sycl_tla = std::nullopt) {
  check_scaled_dot_product_attention_arguments(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      enable_gqa);

  auto plan = resolve_execution_plan(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      plan_cache_len,
      plan_kv_new_len,
      plan_is_causal,
      plan_is_decode,
      plan_use_sycl_tla);
  auto fp32_output = resolve_fp32_output(plan, query.size(3), output_dtype);

  if (plan.use_sycl_tla) {
    return scaled_dot_product_attention_sycl_tla(query, key, value, plan, scale, fp32_output, output);
  }

  if (enable_gqa) {
    key = expand_gqa_heads(key, query.size(1));
    value = expand_gqa_heads(value, query.size(1));
  }

  auto fallback_output = scaled_dot_product_attention_fallback(
      query,
      key,
      value,
      attn_mask,
      dropout_p,
      is_causal,
      scale);

  if (output.has_value()) {
    auto out = prepare_output_tensor(query, output, false);
    out.copy_(fallback_output);
    return out;
  }

  return fallback_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<PreparedScaledDotProductAttention, std::shared_ptr<PreparedScaledDotProductAttention>>(
      m,
      "PreparedScaledDotProductAttention");

  m.def(
      "scaled_dot_product_attention",
      &scaled_dot_product_attention_sycl,
      pybind11::arg("query"),
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("attn_mask") = pybind11::none(),
      pybind11::arg("dropout_p") = 0.0,
      pybind11::arg("is_causal") = false,
      pybind11::arg("scale") = pybind11::none(),
      pybind11::arg("enable_gqa") = false,
      pybind11::arg("output_dtype") = pybind11::none(),
      pybind11::arg("output") = pybind11::none(),
      pybind11::arg("plan_cache_len") = pybind11::none(),
      pybind11::arg("plan_kv_new_len") = pybind11::none(),
      pybind11::arg("plan_is_causal") = pybind11::none(),
      pybind11::arg("plan_is_decode") = pybind11::none(),
      pybind11::arg("plan_use_sycl_tla") = pybind11::none(),
      "SDPA-compatible flash attention exposed from sycl_ext with a sycl-tla-backed bf16 XPU path and ATen fallback.");

  m.def(
      "prepare_scaled_dot_product_attention",
      &prepare_scaled_dot_product_attention_sycl,
      pybind11::arg("query"),
      pybind11::arg("key"),
      pybind11::arg("value"),
      pybind11::arg("attn_mask") = pybind11::none(),
      pybind11::arg("dropout_p") = 0.0,
      pybind11::arg("is_causal") = false,
      pybind11::arg("scale") = pybind11::none(),
      pybind11::arg("enable_gqa") = false,
      pybind11::arg("output_dtype") = pybind11::none(),
      pybind11::arg("output") = pybind11::none(),
      pybind11::arg("plan_cache_len") = pybind11::none(),
      pybind11::arg("plan_kv_new_len") = pybind11::none(),
      pybind11::arg("plan_is_causal") = pybind11::none(),
      pybind11::arg("plan_is_decode") = pybind11::none(),
      pybind11::arg("plan_use_sycl_tla") = pybind11::none(),
      "Prepare and cache the sycl-tla FMHA launch state for repeated execution.");

  m.def(
      "run_prepared_scaled_dot_product_attention",
      &run_prepared_scaled_dot_product_attention,
      pybind11::arg("prepared"),
      "Launch a previously prepared sycl-tla FMHA operation on the current XPU stream.");
}