#include "pytorch_shim.h"

#include "core/registration.h"
#include "xpu/attn/attn_interface.h"
#include "utils.h"
#include <torch/all.h>


#ifndef MLA_NAMESPACE
#define MLA_NAMESPACE flash_mla
#endif

namespace MLA_NAMESPACE {

inline int get_num_splits(
    const sycl::queue& queue,
    const int& batch_size,
    const int& num_heads_kv,
    const int& max_seqlen_k,
    const int& block_size) {
  auto device = queue.get_device();
  int num_xe_cores =
      device.get_info<sycl::ext::intel::info::device::gpu_slices>() *
      device
          .get_info<sycl::ext::intel::info::device::gpu_subslices_per_slice>();
  int parallel_ = num_xe_cores;
  int parallel_2 = num_xe_cores * 2;

  int cur_parallel_d = batch_size * num_heads_kv;

  int num_splits = (parallel_ + cur_parallel_d - 1) / cur_parallel_d;

  if (cur_parallel_d * num_splits > parallel_ && num_splits > 1) {
    num_splits = std::ceil(parallel_2 / static_cast<float>(cur_parallel_d)) - 1;
  }

  int max_splits = (max_seqlen_k + block_size - 1) / block_size;
  max_splits = std::min(max_splits, parallel_);
  return std::min(num_splits, max_splits);
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// The FMHA kernel tile processes q_packed Q positions at once via
// head_group_q = num_heads_q / num_heads_kv.  We must pad num_heads_q so
// that head_group_q is a multiple of kQPacked to avoid buffer overflows.
static constexpr int kQPacked = 8;

std::vector<at::Tensor> dense_mla_decode_fwd(
    at::Tensor& q,                                // [batch, seqlen_q, num_heads_q, head_dim_k]
    const at::Tensor& kcache,                     // [num_blocks, page_block_size, num_heads_k, head_dim_k]
    int head_dim_v,
    const at::Tensor& seqlens_k,                  // [batch]
    const at::Tensor& block_table,                // [batch, max_num_blocks_per_seq]
    float softmax_scale,
    bool is_causal,
    std::optional<at::Tensor>& tile_scheduler_metadata,
    std::optional<at::Tensor>& num_splits_opt) {
  // ---- Input validation ----
  auto q_dtype = q.scalar_type();
  TORCH_CHECK(
      q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
      "MLA decode only supports fp16 and bf16");
  TORCH_CHECK(kcache.dtype() == q_dtype, "q and kcache must have the same dtype");
  TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must be int32");
  TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must be int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(kcache);
  CHECK_DEVICE(seqlens_k);
  CHECK_DEVICE(block_table);

  TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dim");
  TORCH_CHECK(kcache.stride(-1) == 1, "kcache must have contiguous last dim");
  CHECK_CONTIGUOUS(seqlens_k);
  TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dim");

  // ---- Extract shapes ----
  const int batch_size = q.size(0);
  const int seqlen_q = q.size(1);
  const int num_heads_q = q.size(2);
  const int head_dim_k = q.size(3);
  TORCH_CHECK(head_dim_v > 0 && head_dim_v <= head_dim_k,
      "head_dim_v must be in (0, head_dim_k]");

  const int page_block_size = kcache.size(1);
  const int num_heads_kv = kcache.size(2);
  TORCH_CHECK(page_block_size == 64, "page_block_size must be 64");
  TORCH_CHECK(batch_size > 0, "batch size must be positive");
  TORCH_CHECK(num_heads_q % num_heads_kv == 0,
      "num_heads_q must be divisible by num_heads_kv");

  if (seqlen_q == 1) { is_causal = false; }

  // ---- Strategy ----
  // The kernel processes kQPacked=8 Q positions per tile and grid.y is based on
  // seq_len_qo (=1 for decode).  So head_group_q = num_heads_q/num_heads_kv
  // must be exactly kQPacked.  When gqa_ratio > kQPacked, we split into
  // ceil(gqa_ratio/kQPacked) "groups", each a separate virtual batch with
  // head_group_q = kQPacked.

  const int gqa_ratio = num_heads_q / num_heads_kv;
  const int num_groups = cdiv(gqa_ratio, kQPacked);
  const int padded_gqa = num_groups * kQPacked;
  // Per-virtual-batch head count: always kQPacked per KV head
  const int kernel_h_q = kQPacked * num_heads_kv;

  const int bs_sq = batch_size * seqlen_q;
  const int virtual_batch = bs_sq * num_groups;

  auto opts = q.options();
  auto float_opts = opts.dtype(at::kFloat);

  // ---- Q reshape ----
  // [b, s_q, h_q, d] = [b*s_q, h_kv, gqa_ratio, d]
  //  → pad gqa → [b*s_q, h_kv, num_groups, kQPacked, d]
  //  → permute  → [b*s_q, num_groups, h_kv, kQPacked, d]
  //  → reshape  → [virtual_batch, kernel_h_q, d]
  at::Tensor q_kernel = q.reshape({bs_sq, num_heads_kv, gqa_ratio, head_dim_k});
  if (padded_gqa != gqa_ratio) {
    at::Tensor q_padded = torch::zeros(
        {bs_sq, num_heads_kv, padded_gqa, head_dim_k}, opts);
    q_padded.narrow(2, 0, gqa_ratio).copy_(q_kernel);
    q_kernel = q_padded;
  }
  q_kernel = q_kernel
      .view({bs_sq, num_heads_kv, num_groups, kQPacked, head_dim_k})
      .permute({0, 2, 1, 3, 4})
      .reshape({virtual_batch, kernel_h_q, head_dim_k})
      .contiguous();

  // ---- cu_seqlens_q: [0, 1, 2, ..., virtual_batch] (1 token per VB) ----
  at::Tensor cu_seqlens_q = torch::arange(
      0, virtual_batch + 1,
      torch::TensorOptions().dtype(torch::kInt32).device(q.device()));

  // ---- Replicate seqlens_k and block_table for virtual batches ----
  at::Tensor seqlens_k_exp;
  at::Tensor block_table_exp = block_table;
  int total_replicate = seqlen_q * num_groups;
  if (total_replicate > 1) {
    block_table_exp = block_table.repeat_interleave(total_replicate, 0);
  }

  if (is_causal && seqlen_q > 1) {
    // Causal: Q position j attends to K[0 .. seqlens_k - s_q + j].
    // Encode this by adjusting seqlens_k per Q position, then disable causal.
    auto q_offsets = torch::arange(
        -(seqlen_q - 1), 1,
        torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
    // [batch, s_q]: each element = seqlens_k[b] - s_q + 1 + j
    auto sk_per_pos = (seqlens_k.unsqueeze(1) + q_offsets.unsqueeze(0))
        .clamp_min(0)
        .reshape({bs_sq});
    seqlens_k_exp = (num_groups > 1)
        ? sk_per_pos.repeat_interleave(num_groups)
        : sk_per_pos;
    is_causal = false;
  } else if (total_replicate > 1) {
    seqlens_k_exp = seqlens_k.repeat_interleave(total_replicate);
  } else {
    seqlens_k_exp = seqlens_k;
  }

  // ---- Zero-pad unused KV cache positions ----
  // The kernel masks attention weights to 0 for positions beyond seqlens_k,
  // but V at those positions may contain NaN.  Since 0*NaN = NaN (IEEE 754),
  // we must zero unused V (= K) positions in the last partial block.
  // Also sanitize block_table for zero-seqlen batches to prevent OOB reads.
  {
    auto seqlens_cpu = seqlens_k.cpu();
    auto bt_cpu = block_table.cpu();
    auto sl = seqlens_cpu.accessor<int, 1>();
    auto bt = bt_cpu.accessor<int, 2>();
    for (int i = 0; i < batch_size; i++) {
      if (sl[i] == 0) {
        // Zero-seqlen: set block_table to 0 to prevent OOB reads.
        // The kernel will read valid memory but produce empty results.
        block_table.select(0, i).zero_();
      } else {
        int rem = sl[i] % page_block_size;
        if (rem > 0) {
          int phys = bt[i][sl[i] / page_block_size];
          kcache.select(0, phys)
                .narrow(0, rem, page_block_size - rem)
                .zero_();
        }
      }
    }
  }

  // ---- Get SYCL queue and compute num_kv_splits ----
  auto& queue = vllm::xpu::vllmGetQueue(q.device().index());

  auto seqlens_cpu = seqlens_k.cpu();
  int max_seqlen_k = seqlens_cpu.max().item<int>();

  // Need at least 2 splits so the kernel writes exp_sums/max_logits
  int num_kv_splits = std::max(
      get_num_splits(
          queue, virtual_batch, num_heads_kv, max_seqlen_k, page_block_size),
      2);

  int max_splits = cdiv(max_seqlen_k, page_block_size);
  num_kv_splits = std::min(num_kv_splits, std::max(max_splits, 2));

  // ---- Allocate output and accumulation buffers ----
  // Pre-fill with sentinel values so that empty KV-splits (when
  // num_kv_splits > k_blocks) contribute nothing in ReduceSplitK:
  //   exp_sums = 0, max_logits = -inf, temp_out = 0
  at::Tensor out = torch::empty(
      {virtual_batch, kernel_h_q, head_dim_k}, opts);
  at::Tensor temp_out = torch::zeros(
      {virtual_batch, kernel_h_q * num_kv_splits, head_dim_k}, opts);
  at::Tensor exp_sums = torch::zeros(
      {virtual_batch, kernel_h_q, num_kv_splits}, float_opts);
  at::Tensor max_logits = torch::full(
      {virtual_batch, kernel_h_q, num_kv_splits},
      -std::numeric_limits<float>::infinity(), float_opts);

  // ---- Dispatch to MLA paged decode kernel ----
  cutlass_mla_dense_decode_interface(
      queue,
      q_kernel,          // [virtual_batch, padded_h_q, d_k]
      kcache,
      kcache,            // V = K for MLA (fused KV cache)
      out,
      temp_out,
      exp_sums,
      max_logits,
      block_table_exp,
      cu_seqlens_q,
      seqlens_k_exp,
      /*max_seqlen_q=*/1,
      max_seqlen_k,
      softmax_scale,
      is_causal,
      num_kv_splits);

  // ---- Extract valid heads ----
  // Kernel output: [virtual_batch, kernel_h_q, d_k]
  //  = [bs_sq * num_groups, h_kv * kQPacked, d_k]
  // → reshape  [bs_sq, num_groups, h_kv, kQPacked, d_k]
  // → permute  [bs_sq, h_kv, num_groups, kQPacked, d_k]
  // → reshape  [bs_sq, h_kv, padded_gqa, d_k]
  // → trim     [bs_sq, h_kv, gqa_ratio, d_k]
  // → reshape  [bs_sq, h_q, d_k]
  auto extract_heads = [&](at::Tensor t, int last_dim) -> at::Tensor {
    return t
        .view({bs_sq, num_groups, num_heads_kv, kQPacked, last_dim})
        .permute({0, 2, 1, 3, 4})
        .reshape({bs_sq, num_heads_kv, padded_gqa, last_dim})
        .narrow(2, 0, gqa_ratio)
        .reshape({bs_sq, num_heads_q, last_dim})
        .contiguous();
  };
  out = extract_heads(out, head_dim_k);
  exp_sums = extract_heads(exp_sums, num_kv_splits);
  max_logits = extract_heads(max_logits, num_kv_splits);
  // out/exp_sums/max_logits: [bs_sq, h_q, ...]

  // ---- Compute LSE from exp_sums and max_logits ----
  // The kernel uses exp2 (base-2) for online softmax, so we use exp2 here too.
  // LSE_base2 = max_global + log2(sum(exp_sums * exp2(max_logits - max_global)))
  // LSE_base_e = LSE_base2 * ln(2)
  at::Tensor max_global = std::get<0>(max_logits.max(-1, true));
  at::Tensor lse = max_global.squeeze(-1) +
      (exp_sums * (max_logits - max_global).exp2()).sum(-1).log2();
  lse = lse * static_cast<float>(M_LN2);  // convert to natural log
  // lse: [bs_sq, h_q]

  // ---- Slice output to head_dim_v ----
  out = out.narrow(-1, 0, head_dim_v);
  // out: [bs_sq, h_q, d_v]

  // ---- Reshape output to [batch, s_q, h_q, d_v] ----
  out = out
      .view({batch_size, seqlen_q, num_heads_q, head_dim_v})
      .contiguous();

  // ---- Reshape LSE to [batch, h_q, s_q] ----
  lse = lse
      .view({batch_size, seqlen_q, num_heads_q})
      .permute({0, 2, 1})
      .contiguous();

  // ---- Fix zero-seqlen batches ----
  // When seqlens_k=0, ReduceSplitK produces NaN (from -inf - (-inf)).
  // Correct output: out=0, lse=+inf for those batch elements.
  {
    auto zero_mask = (seqlens_k == 0);  // [batch]
    if (zero_mask.any().item<bool>()) {
      auto zm_out = zero_mask.view({batch_size, 1, 1, 1});
      out.masked_fill_(zm_out, 0);
      auto zm_lse = zero_mask.view({batch_size, 1, 1});
      lse.masked_fill_(zm_lse, std::numeric_limits<float>::infinity());
    }
  }

  // Return: [out, lse, dummy_metadata, dummy_splits]
  auto dummy = torch::empty({0}, opts.dtype(torch::kInt32));
  return {out, lse, dummy, dummy};
}

std::vector<at::Tensor> sparse_mla_decode_fwd(
    at::Tensor& q,                                // [batch, seqlen_q, num_heads_q, head_dim_k]
    const at::Tensor& kcache,                     // [num_blocks, page_block_size, num_heads_k, head_dim_k] (or FP8 layout)
    const at::Tensor& indices,                    // [batch, seqlen_q, topk], int32
    int head_dim_v,
    float softmax_scale,
    bool is_fp8_kvcache,
    std::optional<at::Tensor>& topk_length_opt,   // [batch], int32
    std::optional<at::Tensor>& attn_sink_opt) {   // [num_heads_q], float32

  auto q_dtype = q.scalar_type();
  TORCH_CHECK(
      q_dtype == at::ScalarType::BFloat16,
      "Sparse MLA decode only supports bf16 query");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(kcache);
  CHECK_DEVICE(indices);
  TORCH_CHECK(q.stride(-1) == 1, "q must have contiguous last dim");

  // Ensure contiguous layout
  q = q.contiguous();
  at::Tensor indices_c = indices.contiguous();

  const int batch_size = q.size(0);
  const int seqlen_q = q.size(1);
  const int num_heads_q = q.size(2);
  const int head_dim_k = q.size(3);
  const int topk = indices_c.size(2);

  TORCH_CHECK(head_dim_v > 0 && head_dim_v <= head_dim_k);

  auto opts = q.options();
  auto float_opts = opts.dtype(at::kFloat);

  // Output tensors
  at::Tensor out = torch::zeros(
      {batch_size, seqlen_q, num_heads_q, head_dim_v}, opts);
  // Use [batch*s_q, h_q, 1] layout so that virtual-batch dim is contiguous with batch*s_q.
  // We reshape to [batch, h_q, s_q] at the very end.
  at::Tensor lse = torch::full(
      {batch_size * seqlen_q, num_heads_q, 1},
      std::numeric_limits<float>::infinity(), float_opts);

  // Determine FP8 model type and prepare KV data
  int fp8_model = 0;  // 0=NONE, 1=V32, 2=MODEL1
  int fp8_bytes_per_token = 0;
  int fp8_page_block_size = 0;
  int total_tokens = 0;
  at::Tensor kv_for_kernel;

  if (is_fp8_kvcache) {
    // FP8 in-kernel dequant path
    const int num_blocks = kcache.size(0);
    const int page_block_sz = kcache.size(1);
    fp8_page_block_size = page_block_sz;
    total_tokens = num_blocks * page_block_sz;

    if (head_dim_k == 576) {
      // V32: per-token layout, 656 bytes/token
      // Shape: [num_blocks, block_size, 1, 656] as fp8
      // After contiguous(): clean per-token layout
      fp8_model = 1;
      fp8_bytes_per_token = 656;
      auto kv_c = kcache.contiguous();
      kv_for_kernel = kv_c.reshape({-1, 656});  // [total_tokens, 656]
    } else {
      // MODEL1: block-level layout
      // Data: [block_size*576 nope_rope bytes] [block_size*8 scale bytes] per block
      // DON'T call contiguous() — it would corrupt the block-level data layout.
      // Flatten to [num_blocks, flat_bytes] to preserve original byte ordering.
      fp8_model = 2;
      auto kv_2d = kcache.view({num_blocks, -1});
      fp8_bytes_per_token = kv_2d.stride(0);  // block byte stride (may include padding)
      kv_for_kernel = kv_2d;
    }
  } else {
    // BF16 path
    at::Tensor kcache_c = kcache.contiguous();
    const int num_heads_kv = kcache_c.size(2);
    TORCH_CHECK(num_heads_q % num_heads_kv == 0);
    kv_for_kernel = kcache_c.reshape({-1, num_heads_kv, head_dim_k});
    total_tokens = kv_for_kernel.size(0);
  }

  auto& stream = vllm::xpu::vllmGetQueue(q.device().index());

  // ---- GQA Group Splitting ----
  // The kernel processes kQPacked=8 Q heads per MMA tile.
  // When gqa_ratio > 8, we split into num_groups virtual batches.
  constexpr int kQPacked = 8;
  const int num_heads_kv = is_fp8_kvcache ? 1 : 1;  // MLA always has h_kv=1
  const int gqa_ratio = num_heads_q / num_heads_kv;
  const int num_groups = (gqa_ratio + kQPacked - 1) / kQPacked;
  const int padded_gqa = num_groups * kQPacked;
  const int kernel_h_q = kQPacked * num_heads_kv;  // =8

  const int bs_sq = batch_size * seqlen_q;
  const int virtual_batch = bs_sq * num_groups;

  // Reshape Q: [b, s_q, h_q, d] → [b*s_q, h_kv, gqa_ratio, d]
  //   → pad → [b*s_q, h_kv, num_groups, kQPacked, d]
  //   → permute → [b*s_q, num_groups, h_kv, kQPacked, d]
  //   → reshape → [virtual_batch, 1, kernel_h_q, d]
  at::Tensor q_kernel = q.reshape({bs_sq, num_heads_kv, gqa_ratio, head_dim_k});
  if (padded_gqa != gqa_ratio) {
    at::Tensor q_padded = torch::zeros(
        {bs_sq, num_heads_kv, padded_gqa, head_dim_k}, opts);
    q_padded.narrow(2, 0, gqa_ratio).copy_(q_kernel);
    q_kernel = q_padded;
  }
  q_kernel = q_kernel
      .view({bs_sq, num_heads_kv, num_groups, kQPacked, head_dim_k})
      .permute({0, 2, 1, 3, 4})
      .reshape({virtual_batch, 1, kernel_h_q, head_dim_k})
      .contiguous();

  // Replicate indices for virtual batches: [b, s_q, topk] → [virtual_batch, 1, topk]
  at::Tensor indices_kernel = indices_c.reshape({bs_sq, 1, topk});
  if (num_groups > 1) {
    indices_kernel = indices_kernel.repeat_interleave(num_groups, 0);
  }
  indices_kernel = indices_kernel.contiguous();

  // Output tensors sized for virtual batch.
  // Use head_dim_k (not head_dim_v) so the FMHA's TileShapeOutput (576) matches.
  // The reduce kernel writes head_dim_k elements; we slice to head_dim_v below.
  at::Tensor out_kernel = torch::zeros(
      {virtual_batch, 1, kernel_h_q, head_dim_k}, opts);
  at::Tensor lse_kernel = torch::full(
      {virtual_batch, kernel_h_q, 1},
      std::numeric_limits<float>::infinity(), float_opts);

  // Make optional tensors contiguous
  std::optional<at::Tensor> topk_length_contig;
  if (topk_length_opt.has_value()) {
    topk_length_contig = topk_length_opt.value().contiguous();
  }
  std::optional<at::Tensor> attn_sink_contig;
  if (attn_sink_opt.has_value()) {
    attn_sink_contig = attn_sink_opt.value().contiguous();
  }
  // topk_length: [b] — replicate for s_q only (all groups share same topk_length).
  // For num_groups==1: replicate to [b*s_q] = [virtual_batch]
  // For num_groups>1: replicate to [b*s_q] = [bs_sq] (per-group kernel uses local batch idx)
  at::Tensor topk_length_kernel;
  if (topk_length_contig.has_value()) {
    topk_length_kernel = topk_length_contig.value();
    if (seqlen_q > 1) {
      topk_length_kernel = topk_length_kernel.repeat_interleave(seqlen_q, 0);
    }
    if (num_groups == 1) {
      // No groups, but still might need contiguous
    }
    topk_length_kernel = topk_length_kernel.contiguous();
  }
  // attn_sink: [h_q] → slice to [kernel_h_q] per group (reindex per launch)
  // For simplicity, pass full sink; kernel only reads [0..kernel_h_q)
  // We handle per-group sink below in the loop
  const at::Tensor* topk_len_p = topk_length_contig.has_value() ? &topk_length_kernel : nullptr;

  // Compute num_splits for Split-K
  constexpr int TOKENS_PER_SPLIT = 128;
  constexpr int MAX_SPLITS = 8;
  int num_splits = std::max(1, std::min(MAX_SPLITS, topk / TOKENS_PER_SPLIT));

  // Allocate split buffers: defer to avoid wasting memory when num_groups > 1
  // (each group allocates its own per-group buffers in the loop below)
  // Init accumulators so unused splits (from topk_length) don't produce NaN
  at::Tensor o_accum, lse_accum, max_logits_accum;
  if (num_groups == 1) {
    o_accum = torch::zeros(
        {virtual_batch, 1, num_splits, kernel_h_q, head_dim_k},
        opts.dtype(at::kFloat));
    lse_accum = torch::zeros(
        {virtual_batch, 1, num_splits, kernel_h_q},
        opts.dtype(at::kFloat));
    max_logits_accum = torch::full(
        {virtual_batch, 1, num_splits, kernel_h_q},
        -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
  }

  // Per-group attn_sink slicing
  std::vector<at::Tensor> sink_slices;
  if (attn_sink_contig.has_value()) {
    auto& full_sink = attn_sink_contig.value();
    for (int g = 0; g < num_groups; g++) {
      int start = g * kQPacked * num_heads_kv;
      int end = std::min(start + kQPacked * num_heads_kv, num_heads_q);
      if (start < num_heads_q) {
        auto slice = full_sink.narrow(0, start, end - start);
        if (slice.size(0) < kernel_h_q) {
          auto padded = torch::zeros({kernel_h_q}, full_sink.options());
          padded.narrow(0, 0, slice.size(0)).copy_(slice);
          sink_slices.push_back(padded.contiguous());
        } else {
          sink_slices.push_back(slice.contiguous());
        }
      } else {
        sink_slices.push_back(torch::zeros({kernel_h_q}, full_sink.options()));
      }
    }
  }

  // Launch kernel for each group (or single launch if num_groups=1)
  for (int g = 0; g < num_groups; g++) {
    // Slice virtual batch for this group
    int vb_start = g;
    int vb_stride = num_groups;
    // q_kernel is already [virtual_batch, 1, kernel_h_q, d] in interleaved order
    // Virtual batch g, g+num_groups, g+2*num_groups, ... correspond to group g
    // But we shaped it with repeat_interleave order: [b0_g0, b0_g1, ..., b0_gN, b1_g0, ...]

    // Actually the q_kernel is already in [vb, 1, h, d] order where
    // vb = [b0_g0, b0_g1, ..., b0_gN-1, b1_g0, ...] via the reshape+permute.
    // We need to slice out every num_groups-th element starting at g.
    at::Tensor q_group, idx_group, out_group, lse_group;
    at::Tensor o_acc_group, lse_acc_group, ml_acc_group;

    if (num_groups == 1) {
      q_group = q_kernel;
      idx_group = indices_kernel;
      out_group = out_kernel;
      lse_group = lse_kernel;
      o_acc_group = o_accum;
      lse_acc_group = lse_accum;
      ml_acc_group = max_logits_accum;
    } else {
      // Use index_select with stride pattern
      auto group_indices = torch::arange(0, virtual_batch, 1,
          torch::TensorOptions().dtype(torch::kLong).device(q.device()));
      // Reshape to [bs_sq, num_groups] and select column g
      group_indices = group_indices.reshape({bs_sq, num_groups}).select(1, g).contiguous();
      q_group = q_kernel.index_select(0, group_indices).contiguous();
      idx_group = indices_kernel.index_select(0, group_indices).contiguous();
      out_group = torch::zeros({bs_sq, 1, kernel_h_q, head_dim_k}, opts);
      lse_group = torch::full({bs_sq, kernel_h_q, 1},
          std::numeric_limits<float>::infinity(), float_opts);
      // Init accumulators so unused splits (from topk_length) don't produce NaN
      o_acc_group = torch::zeros({bs_sq, 1, num_splits, kernel_h_q, head_dim_k},
          opts.dtype(at::kFloat));
      lse_acc_group = torch::zeros({bs_sq, 1, num_splits, kernel_h_q},
          opts.dtype(at::kFloat));
      ml_acc_group = torch::full({bs_sq, 1, num_splits, kernel_h_q},
          -std::numeric_limits<float>::infinity(), opts.dtype(at::kFloat));
    }

    const at::Tensor* sink_p = (!sink_slices.empty()) ? &sink_slices[g] : nullptr;
    at::Tensor* o_acc_p = &o_acc_group;
    at::Tensor* lse_acc_p = &lse_acc_group;
    at::Tensor* ml_p = &ml_acc_group;

    int effective_batch = (num_groups == 1) ? virtual_batch : bs_sq;

    cutlass_mla_sparse_decode_interface(
        stream, q_group, kv_for_kernel, idx_group, out_group, lse_group,
        softmax_scale, head_dim_v, total_tokens,
        sink_p, topk_len_p,
        fp8_model, fp8_bytes_per_token, fp8_page_block_size,
        num_splits, o_acc_p, lse_acc_p, ml_p);

    if (num_groups > 1) {
      // Scatter results back: out_group [bs_sq, 1, 8, d_qk] → out [b, s_q, h_q, d_v]
      int h_start = g * kQPacked;
      int h_end = std::min(h_start + kQPacked, num_heads_q);
      int h_count = h_end - h_start;
      out.view({bs_sq, 1, num_heads_q, head_dim_v})
          .narrow(2, h_start, h_count)
          .copy_(out_group.narrow(2, 0, h_count).narrow(-1, 0, head_dim_v));
      lse.view({bs_sq, num_heads_q, 1})
          .narrow(1, h_start, h_count)
          .copy_(lse_group.narrow(1, 0, h_count));
    }
  }

  // For num_groups==1, reshape directly
  if (num_groups == 1) {
    int h_count = std::min(kQPacked, num_heads_q);
    out.view({bs_sq, 1, num_heads_q, head_dim_v})
        .narrow(2, 0, h_count)
        .copy_(out_kernel.narrow(2, 0, h_count).narrow(-1, 0, head_dim_v));
    lse.view({bs_sq, num_heads_q, 1})
        .narrow(1, 0, h_count)
        .copy_(lse_kernel.narrow(1, 0, h_count));
  }

  // Reshape lse from [bs_sq, h_q, 1] to [batch, h_q, s_q]
  lse = lse.squeeze(-1)                                   // [bs_sq, h_q]
      .view({batch_size, seqlen_q, num_heads_q})           // [batch, s_q, h_q]
      .permute({0, 2, 1})                                  // [batch, h_q, s_q]
      .contiguous();

  // Fix all-invalid batches (e.g. topk_length==0): kernel produces NaN/−inf
  // Correct: out=0, lse=+inf for those entries.
  {
    auto invalid_lse = (lse != lse) | (lse <= -1e38f);  // NaN or very negative
    if (invalid_lse.any().item<bool>()) {
      // lse: [batch, h_q, s_q]  out: [batch, s_q, h_q, d_v]
      lse.masked_fill_(invalid_lse, std::numeric_limits<float>::infinity());
      auto inv_out = invalid_lse.permute({0, 2, 1}).unsqueeze(-1);  // [batch, s_q, h_q, 1]
      out.masked_fill_(inv_out, 0);
    }
  }

  auto dummy = torch::empty({0}, opts.dtype(torch::kInt32));
  return {out, lse, dummy, dummy};
}

std::vector<at::Tensor> sparse_mla_prefill_fwd(
    at::Tensor& q,                                // [s_q, h_q, d_qk]
    const at::Tensor& kv,                         // [s_kv, h_kv, d_qk]
    const at::Tensor& indices,                    // [s_q, h_kv, topk], int32
    float sm_scale,
    int d_v,
    std::optional<at::Tensor>& attn_sink_opt,     // [h_q], float32
    std::optional<at::Tensor>& topk_length_opt) { // [s_q], int32

  auto q_dtype = q.scalar_type();
  TORCH_CHECK(
      q_dtype == at::ScalarType::BFloat16,
      "Sparse MLA prefill only supports bf16");
  TORCH_CHECK(kv.scalar_type() == q_dtype, "q and kv must have the same dtype");
  TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(kv);
  CHECK_DEVICE(indices);

  // Ensure contiguous layout (tests use non_contiguousify)
  q = q.contiguous();
  at::Tensor kv_c = kv.contiguous();
  at::Tensor indices_c = indices.contiguous();

  const int s_q = q.size(0);
  const int h_q = q.size(1);
  const int d_qk = q.size(2);
  const int s_kv = kv_c.size(0);
  const int h_kv = kv_c.size(1);
  const int topk = indices_c.size(2);

  TORCH_CHECK(d_v > 0 && d_v <= d_qk);
  TORCH_CHECK(h_q % h_kv == 0);

  auto opts = q.options();
  auto float_opts = opts.dtype(at::kFloat);

  // Allocate outputs — kernel writes d_qk width, Python slices to d_v
  at::Tensor out = torch::zeros({s_q, h_q, d_qk}, opts);
  at::Tensor max_logits = torch::full({s_q, h_q},
      -std::numeric_limits<float>::infinity(), float_opts);
  // lse tensor: kernel writes exp_sums here (not log-sum-exp).
  // Initialize to 0 (sum of 0 exps = 0 when no valid tokens).
  at::Tensor lse = torch::zeros({s_q, h_q}, float_opts);

  auto& stream = vllm::xpu::vllmGetQueue(q.device().index());

  // Indices: [s_q, h_kv, topk] → [s_q, topk] (select first kv head)
  at::Tensor indices_flat = indices_c.select(1, 0).contiguous();

  // Prepare optional tensors
  std::optional<at::Tensor> attn_sink_contig;
  if (attn_sink_opt.has_value()) {
    auto sink = attn_sink_opt.value().contiguous();
    if (sink.device() != q.device()) sink = sink.to(q.device());
    attn_sink_contig = sink;
  }
  std::optional<at::Tensor> topk_length_contig;
  if (topk_length_opt.has_value()) {
    topk_length_contig = topk_length_opt.value().contiguous();
  }

  const at::Tensor* sink_p = nullptr;  // sink applied host-side, not in kernel
  at::Tensor topk_length_c;
  const at::Tensor* topk_len_p = nullptr;
  if (topk_length_contig.has_value()) {
    topk_length_c = topk_length_contig.value().contiguous();
    topk_len_p = &topk_length_c;
  }

  // Single kernel launch — handles all s_q × num_groups × h_kv
  // Don't pass attn_sink to kernel: returned lse/max_logits must be token-only.
  cutlass_mla_sparse_prefill_interface(
      stream, q, kv_c, indices_flat, out, lse, max_logits,
      sm_scale, d_v, /*total_tokens=*/s_kv,
      /*attn_sink=*/nullptr, topk_len_p);

  // Kernel writes max_logits in log2 scale and exp_sums (not lse).
  // Convert: lse = max_logits_log2 * ln(2) + ln(exp_sums)
  //          max_logits = max_logits_log2 * ln(2)
  constexpr float kLn2 = static_cast<float>(M_LN2);  // 0.693...

  // Save max_logits_log2 before overwriting (needed for lse computation)
  at::Tensor ml_log2 = max_logits.clone();

  // max_logits: log2 → natural scale
  max_logits.mul_(kLn2);
  max_logits.masked_fill_(max_logits <= -1e38f,
                          -std::numeric_limits<float>::infinity());

  // lse: the kernel wrote exp_sums into the lse tensor.
  // lse = ml_log2 * ln(2) + ln(exp_sums)
  // Guard: when exp_sums=0 and ml_log2=-inf, result should be -inf
  at::Tensor ln_exp_sums = torch::log(lse);  // lse currently holds exp_sums
  lse = ml_log2 * kLn2 + ln_exp_sums;

  // Apply attn_sink scaling to output (host-side):
  // out *= 1 / (1 + exp(sink - lse))
  if (attn_sink_contig.has_value()) {
    auto& full_sink = attn_sink_contig.value();  // [h_q] float32
    auto sink_expanded = full_sink.unsqueeze(0).expand({s_q, h_q});  // [s_q, h_q]
    auto scaling = torch::reciprocal(
        1.0f + torch::exp(sink_expanded - lse));
    scaling = torch::nan_to_num(scaling, /*nan=*/0.0f);
    out.mul_(scaling.unsqueeze(-1));  // broadcast over d_qk dim
  }

  // Slice output to d_v (kernel writes full d_qk)
  out = out.narrow(-1, 0, d_v);

  // Prefill convention: convert -inf LSE → +inf (indicates no valid tokens)
  lse.masked_fill_(lse == -std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity());

  return {out, max_logits, lse};
}

std::vector<at::Tensor> dense_mla_prefill_fwd(
    at::Tensor& q,                                // [total_q, h_q, d_qk]
    const at::Tensor& kv,                         // [total_kv, h_kv, d_qk]
    const at::Tensor& cu_seqlens_q,               // [batch+1] int32
    const at::Tensor& cu_seqlens_k,               // [batch+1] int32
    float sm_scale,
    int d_v,
    bool is_causal) {

  auto q_dtype = q.scalar_type();
  TORCH_CHECK(
      q_dtype == at::ScalarType::BFloat16,
      "Dense MLA prefill only supports bf16");
  TORCH_CHECK(kv.scalar_type() == q_dtype, "q and kv must have the same dtype");
  TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must be int32");
  TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must be int32");

  CHECK_DEVICE(q);
  CHECK_DEVICE(kv);
  CHECK_DEVICE(cu_seqlens_q);
  CHECK_DEVICE(cu_seqlens_k);

  // Ensure contiguous layout
  q = q.contiguous();
  at::Tensor kv_c = kv.contiguous();

  const int total_q = q.size(0);
  const int h_q = q.size(1);
  const int d_qk = q.size(2);
  const int h_kv = kv_c.size(1);

  TORCH_CHECK(d_v > 0 && d_v <= d_qk);
  TORCH_CHECK(h_q % h_kv == 0);

  auto opts = q.options();
  auto float_opts = opts.dtype(at::kFloat);

  // Allocate outputs — kernel writes d_qk width, Python slices to d_v
  at::Tensor out = torch::zeros({total_q, h_q, d_qk}, opts);
  at::Tensor max_logits = torch::full({total_q, h_q},
      -std::numeric_limits<float>::infinity(), float_opts);
  at::Tensor lse = torch::zeros({total_q, h_q}, float_opts);

  auto& stream = vllm::xpu::vllmGetQueue(q.device().index());

  // Single kernel launch
  cutlass_mla_dense_prefill_interface(
      stream, q, kv_c, cu_seqlens_q, cu_seqlens_k,
      out, lse, max_logits, sm_scale, d_v, is_causal);

  // Kernel writes max_logits in log2 scale and exp_sums (not lse).
  // Convert: lse = max_logits_log2 * ln(2) + ln(exp_sums)
  //          max_logits = max_logits_log2 * ln(2)
  constexpr float kLn2 = static_cast<float>(M_LN2);

  at::Tensor ml_log2 = max_logits.clone();

  // max_logits: log2 → natural scale
  max_logits.mul_(kLn2);
  max_logits.masked_fill_(max_logits <= -1e38f,
                          -std::numeric_limits<float>::infinity());

  // lse: the kernel wrote exp_sums into the lse tensor.
  at::Tensor ln_exp_sums = torch::log(lse);
  lse = ml_log2 * kLn2 + ln_exp_sums;

  // Slice output to d_v
  out = out.narrow(-1, 0, d_v);

  // Prefill convention: convert -inf LSE → +inf
  lse.masked_fill_(lse == -std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::infinity());

  return {out, max_logits, lse};
}

}  // namespace MLA_NAMESPACE

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "dense_mla_decode_fwd(Tensor q, Tensor kcache, int head_dim_v, "
      "Tensor seqlens_k, Tensor block_table, "
      "float softmax_scale, bool is_causal, "
      "Tensor? tile_scheduler_metadata, Tensor? num_splits) -> Tensor[]");
  ops.impl(
      "dense_mla_decode_fwd",
      torch::kXPU,
      make_pytorch_shim(&MLA_NAMESPACE::dense_mla_decode_fwd));

  ops.def(
      "sparse_mla_decode_fwd(Tensor q, Tensor kcache, Tensor indices, "
      "int head_dim_v, float softmax_scale, bool is_fp8_kvcache, "
      "Tensor? topk_length, Tensor? attn_sink) -> Tensor[]");
  ops.impl(
      "sparse_mla_decode_fwd",
      torch::kXPU,
      make_pytorch_shim(&MLA_NAMESPACE::sparse_mla_decode_fwd));

  ops.def(
      "sparse_mla_prefill_fwd(Tensor q, Tensor kv, Tensor indices, "
      "float sm_scale, int d_v, Tensor? attn_sink, "
      "Tensor? topk_length) -> Tensor[]");
  ops.impl(
      "sparse_mla_prefill_fwd",
      torch::kXPU,
      make_pytorch_shim(&MLA_NAMESPACE::sparse_mla_prefill_fwd));

  ops.def(
      "dense_mla_prefill_fwd(Tensor q, Tensor kv, Tensor cu_seqlens_q, "
      "Tensor cu_seqlens_k, float sm_scale, int d_v, "
      "bool is_causal) -> Tensor[]");
  ops.impl(
      "dense_mla_prefill_fwd",
      torch::kXPU,
      make_pytorch_shim(&MLA_NAMESPACE::dense_mla_prefill_fwd));
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
