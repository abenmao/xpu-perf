#include "csrc/utils.h"
#include "attn_interface.h"

#ifdef VLLM_XPU_ENABLE_XE2
  #include "csrc/xpu/attn/xe_2/mla_dense_decode_xe2.h"
  #include "csrc/xpu/attn/xe_2/mla_sparse_decode_xe2.h"
  #include "csrc/xpu/attn/xe_2/mla_sparse_prefill_xe2.h"
  #include "csrc/xpu/attn/xe_2/mla_dense_prefill_xe2.h"
  #include "csrc/xpu/attn/xe_2/paged_decode.hpp"
  #include "csrc/xpu/attn/xe_2/sparse_attn.hpp"
#endif

void cutlass_mla_dense_decode_interface(
    sycl::queue& queue,
    const at::Tensor& query,
    const at::Tensor& key_cache,
    const at::Tensor& value_cache,
    at::Tensor& out,
    at::Tensor& temp_out,
    at::Tensor& exp_sums,
    at::Tensor& max_logits,
    const at::Tensor& block_table,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    int max_seqlen_q,
    int max_seqlen_k,
    double sm_scale,
    bool is_causal,
    int num_kv_splits) {
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    // Build paged_decode_args_t for MLA dispatch
    // query shape: [virtual_batch, padded_h_q, head_dim_k]
    int batch_size = cu_seqlens_q.numel() - 1;
    int num_heads_q = query.size(1);   // padded_h_q (actual Q heads, padded)
    int num_heads_kv = key_cache.size(2);
    int head_size = query.size(2);     // head_dim_k = 576
    int total_seqlen_q = query.size(0);
    int num_blocks = key_cache.size(0);
    int block_size = key_cache.size(1);
    int total_seqlen_k = num_blocks * block_size;
    int max_blocks_per_seq = block_table.size(1);

    paged_decode_args_t args = {
        query.data_ptr(),
        key_cache.data_ptr(),
        value_cache.data_ptr(),
        out.data_ptr(),
        temp_out.data_ptr(),
        exp_sums.data_ptr(),
        max_logits.data_ptr(),
        block_table.data_ptr(),
        cu_seqlens_q.data_ptr(),
        cu_seqlens_k.data_ptr(),
        max_seqlen_q,
        max_seqlen_k,
        total_seqlen_q,
        total_seqlen_k,
        static_cast<float>(sm_scale),
        nullptr,  // sm_sink
        batch_size,
        num_heads_q,
        num_heads_kv,
        head_size,
        max_blocks_per_seq,
        block_size,
        -1,  // window_size_left
        -1,  // window_size_right
        true,   // is_varlen
        true,   // is_paged
        is_causal,
        false,  // is_local
        false,  // is_sink
        num_kv_splits};

    CutlassDType cuType = aten_to_dtype(query);
    cutlass_mla_dense_decode_xe2(queue, cuType, args);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2 cutlass kernel is supported currently.");
  }
}

// cutlass_mla_sparse_prefill_interface: bridges torch tensors to
// sparse_prefill_args_t and dispatches to the prefill kernel.
// Single launch: no Split-K, no GQA loop, full Q/out/lse/max_logits.
void cutlass_mla_sparse_prefill_interface(
    sycl::queue& queue,
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    at::Tensor& out,
    at::Tensor& lse,
    at::Tensor& max_logits,
    float sm_scale,
    int d_v,
    int total_tokens,
    const at::Tensor* attn_sink,
    const at::Tensor* topk_length) {
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    constexpr int kQPacked = 8;
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int gqa_ratio = h_q / h_kv;
    int num_groups = gqa_ratio / kQPacked;

    sparse_prefill_args_t args = {};
    args.q = q.data_ptr();
    args.kv = kv.data_ptr();
    args.indices = const_cast<void*>(indices.data_ptr());
    args.out = out.data_ptr();
    args.lse = lse.data_ptr();
    args.max_logits = max_logits.data_ptr();
    args.attn_sink = attn_sink ? const_cast<void*>(attn_sink->data_ptr()) : nullptr;
    args.topk_length = topk_length ? const_cast<void*>(topk_length->data_ptr()) : nullptr;
    args.s_q = q.size(0);
    args.h_q = h_q;
    args.h_kv = h_kv;
    args.d_qk = q.size(2);
    args.d_v = d_v;
    args.topk = indices.size(1);
    args.total_tokens = total_tokens;
    args.sm_scale = sm_scale;
    args.num_groups = num_groups;

    cutlass_mla_sparse_prefill_xe2(queue, args);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2 cutlass kernel is supported currently.");
  }
}

void cutlass_mla_sparse_decode_interface(
    sycl::queue& queue,
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    at::Tensor& out,
    at::Tensor& lse,
    float sm_scale,
    int d_v,
    int total_tokens,
    const at::Tensor* attn_sink,
    const at::Tensor* topk_length,
    int fp8_model,
    int fp8_bytes_per_token,
    int fp8_page_block_size,
    int num_splits,
    at::Tensor* o_accum,
    at::Tensor* lse_accum,
    at::Tensor* max_logits_accum) {
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    sparse_decode_args_t args = {};
    args.q = q.data_ptr();
    args.kv = kv.data_ptr();
    args.indices = const_cast<void*>(indices.data_ptr());
    args.out = out.data_ptr();
    args.lse = lse.data_ptr();
    args.attn_sink = attn_sink ? const_cast<void*>(attn_sink->data_ptr()) : nullptr;
    args.topk_length = topk_length ? const_cast<void*>(topk_length->data_ptr()) : nullptr;
    args.batch = q.size(0);
    args.s_q = q.size(1);
    args.h_q = q.size(2);
    args.h_kv = (fp8_model == 0) ? kv.size(1) : 1;
    args.d_qk = q.size(3);
    args.d_v = d_v;
    args.topk = indices.size(2);
    args.total_tokens = total_tokens;
    args.sm_scale = sm_scale;
    args.fp8_model = static_cast<FP8ModelType>(fp8_model);
    args.bytes_per_token = fp8_bytes_per_token;
    args.page_block_size = fp8_page_block_size;
    args.num_blocks = (fp8_model != 0 && fp8_page_block_size > 0) ?
        (total_tokens / fp8_page_block_size) : 0;
    args.num_splits = num_splits;
    args.o_accum = o_accum ? o_accum->data_ptr() : nullptr;
    args.lse_accum = lse_accum ? lse_accum->data_ptr() : nullptr;
    args.max_logits_accum = max_logits_accum ? max_logits_accum->data_ptr() : nullptr;

    cutlass_mla_sparse_decode_xe2(queue, args);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2 cutlass kernel is supported currently.");
  }
}

void cutlass_mla_dense_prefill_interface(
    sycl::queue& queue,
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& cu_seqlens_q,
    const at::Tensor& cu_seqlens_k,
    at::Tensor& out,
    at::Tensor& lse,
    at::Tensor& max_logits,
    float sm_scale,
    int d_v,
    bool is_causal) {
  if (vllm::xpu::is_xe2_arch()) {
#ifdef VLLM_XPU_ENABLE_XE2
    constexpr int kQPacked = 8;
    int h_q = q.size(1);
    int h_kv = kv.size(1);
    int gqa_ratio = h_q / h_kv;
    int num_groups = gqa_ratio / kQPacked;
    int batch_size = cu_seqlens_q.size(0) - 1;

    dense_prefill_args_t args = {};
    args.q = q.data_ptr();
    args.kv = kv.data_ptr();
    args.out = out.data_ptr();
    args.lse = lse.data_ptr();
    args.max_logits = max_logits.data_ptr();
    args.cu_seqlens_q = cu_seqlens_q.data_ptr();
    args.cu_seqlens_k = cu_seqlens_k.data_ptr();
    args.batch_size = batch_size;
    args.total_q = q.size(0);
    args.total_kv = kv.size(0);
    args.h_q = h_q;
    args.h_kv = h_kv;
    args.d_qk = q.size(2);
    args.d_v = d_v;
    args.max_seqlen_q = 0;  // not used by kernel
    args.max_seqlen_k = 0;  // not used by kernel
    args.sm_scale = sm_scale;
    args.is_causal = is_causal;
    args.num_groups = num_groups;

    cutlass_mla_dense_prefill_xe2(queue, args);
#else
    TORCH_CHECK(false, "XE2 cutlass kernel is not enabled in this build.");
#endif
  } else {
    TORCH_CHECK(false, "Only XE2 cutlass kernel is supported currently.");
  }
}
