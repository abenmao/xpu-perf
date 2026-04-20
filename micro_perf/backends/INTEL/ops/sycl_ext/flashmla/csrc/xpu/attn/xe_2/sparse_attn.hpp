#pragma once

#include <cstdint>

// FP8 KV cache model type identifiers
enum class FP8ModelType : int {
  NONE = 0,     // No FP8 (plain BF16 KV)
  V32 = 1,      // V3.2: d_qk=576, 656 bytes/token
  MODEL1 = 2,   // MODEL1: d_qk=512, 576 bytes/token
};

// Arguments for sparse attention prefill SYCL kernel.
// Prefill: single kernel launch handles all s_q × num_groups × h_kv work items.
// No Split-K: each WG processes one query's full topk.
// GQA groups are fused into the grid (no host-side loop).
//
// Q: [s_q, h_q, d_qk] bf16 (full h_q, not per-group)
// KV: [total_tokens, h_kv, d_qk] bf16
// indices: [s_q, topk] int32 (h_kv already selected)
// out: [s_q, h_q, d_v] bf16
// lse: [s_q, h_q] float32
// max_logits: [s_q, h_q] float32
struct sparse_prefill_args_t {
  void* q;
  void* kv;
  void* indices;
  void* out;
  void* lse;
  void* max_logits;
  void* attn_sink;      // [h_q] float32 or nullptr
  void* topk_length;    // [s_q] int32 or nullptr
  int s_q;
  int h_q;              // full h_q (e.g. 128)
  int h_kv;             // typically 1 for MLA
  int d_qk;
  int d_v;
  int topk;
  int total_tokens;
  float sm_scale;
  int num_groups;       // h_q / h_kv / kQPacked
};

// Arguments for MLA dense prefill SYCL kernel.
// Variable-length sequences via cu_seqlens_q / cu_seqlens_k.
// Contiguous KV layout: [total_kv, h_kv, d_qk] bf16.
// No Split-K: each WG processes one query token's full KV range.
// GQA groups fused into grid.
//
// Q:   [total_q, h_q, d_qk] bf16
// KV:  [total_kv, h_kv, d_qk] bf16
// out: [total_q, h_q, d_qk] bf16 (sliced to d_v on host)
// lse: [total_q, h_q] float32
// max_logits: [total_q, h_q] float32
struct dense_prefill_args_t {
  void* q;
  void* kv;
  void* out;
  void* lse;
  void* max_logits;
  void* cu_seqlens_q;    // [batch+1] int32
  void* cu_seqlens_k;    // [batch+1] int32
  int batch_size;
  int total_q;
  int total_kv;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  int max_seqlen_q;
  int max_seqlen_k;
  float sm_scale;
  bool is_causal;
  int num_groups;        // h_q / h_kv / kQPacked
};

// Arguments for sparse attention decode SYCL kernel.
// Q: [b, s_q, h_q, d_qk] bf16
// KV: [total_tokens, h_kv, d_qk] bf16 OR raw FP8 bytes
// indices: [b, s_q, topk] int32 (indices_in_kvcache)
// out: [b, s_q, h_q, d_v] bf16
// lse: [b, h_q, s_q] float32
struct sparse_decode_args_t {
  void* q;
  void* kv;
  void* indices;
  void* out;
  void* lse;
  void* attn_sink;    // [h_q] float32 or nullptr
  void* topk_length;  // [b] int32 or nullptr
  int batch;
  int s_q;
  int h_q;
  int h_kv;
  int d_qk;
  int d_v;
  int topk;
  int total_tokens;   // num_blocks * page_block_size
  float sm_scale;
  // FP8 fields
  FP8ModelType fp8_model;    // FP8 model type (NONE for BF16)
  int bytes_per_token;       // bytes per token in FP8 layout
  int page_block_size;       // block size for paged KV cache
  int num_blocks;            // total number of blocks
  // Split-K fields
  int num_splits;            // number of splits (1 = no split)
  void* o_accum;             // [b, s_q, num_splits, h_q, d_v] float32
  void* lse_accum;           // [b, s_q, num_splits, h_q] float32 (exp_sums)
  void* max_logits_accum;    // [b, s_q, num_splits, h_q] float32 (max_logits)
};
