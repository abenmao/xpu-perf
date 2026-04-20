#include "utils.h"

void cutlass_mla_dense_decode_interface(
    sycl::queue& queue,
    const at::Tensor& query,      // [total_q, h_kv, head_dim_k]
    const at::Tensor& key_cache,  // [num_blocks, page_size, h_kv, head_dim_k]
    const at::Tensor& value_cache,// same as key_cache for MLA (fused KV)
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
    int num_kv_splits);

void cutlass_mla_sparse_prefill_interface(
    sycl::queue& queue,
    const at::Tensor& q,          // [s_q, h_q, d_qk] bf16 (full h_q)
    const at::Tensor& kv,         // [total_tokens, h_kv, d_qk] bf16
    const at::Tensor& indices,    // [s_q, topk] int32 (h_kv already selected)
    at::Tensor& out,              // [s_q, h_q, d_v] bf16
    at::Tensor& lse,              // [s_q, h_q] float32
    at::Tensor& max_logits,       // [s_q, h_q] float32
    float sm_scale,
    int d_v,
    int total_tokens,
    const at::Tensor* attn_sink,     // [h_q] float32 or nullptr
    const at::Tensor* topk_length);  // [s_q] int32 or nullptr

void cutlass_mla_sparse_decode_interface(
    sycl::queue& queue,
    const at::Tensor& q,          // [b, s_q, h_q, d_qk] bf16
    const at::Tensor& kv,         // BF16: [total_tokens, h_kv, d_qk] or FP8 raw bytes
    const at::Tensor& indices,    // [b, s_q, topk] int32
    at::Tensor& out,              // [b, s_q, h_q, d_v] bf16
    at::Tensor& lse,              // [b, h_q, s_q] float32
    float sm_scale,
    int d_v,
    int total_tokens,
    const at::Tensor* attn_sink,     // [h_q] float32 or nullptr
    const at::Tensor* topk_length,   // [b] int32 or nullptr
    int fp8_model,                   // 0=NONE, 1=V32, 2=MODEL1
    int fp8_bytes_per_token,         // V32: 656, MODEL1: block_byte_stride
    int fp8_page_block_size,         // block_size for FP8 offset computation
    int num_splits,                  // Split-K: number of splits (1=no split)
    at::Tensor* o_accum,             // [b, s_q, splits, h_q, d_v] f32 or nullptr
    at::Tensor* lse_accum,           // [b, s_q, splits, h_q] f32 (exp_sums)
    at::Tensor* max_logits_accum);   // [b, s_q, splits, h_q] f32 (max_logits)

void cutlass_mla_dense_prefill_interface(
    sycl::queue& queue,
    const at::Tensor& q,          // [total_q, h_q, d_qk] bf16
    const at::Tensor& kv,         // [total_kv, h_kv, d_qk] bf16
    const at::Tensor& cu_seqlens_q, // [batch+1] int32
    const at::Tensor& cu_seqlens_k, // [batch+1] int32
    at::Tensor& out,              // [total_q, h_q, d_qk] bf16
    at::Tensor& lse,              // [total_q, h_q] float32
    at::Tensor& max_logits,       // [total_q, h_q] float32
    float sm_scale,
    int d_v,
    bool is_causal);
