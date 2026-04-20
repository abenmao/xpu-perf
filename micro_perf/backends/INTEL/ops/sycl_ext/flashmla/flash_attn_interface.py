# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple
import dataclasses

import torch

#isort: off

import flash_attn_xpu  # noqa: F401

#isort: on



@dataclasses.dataclass
class FlashMLASchedMeta:
    """Stores the tile scheduler metadata for FlashMLA decode."""

    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool = False
        topk: Optional[int] = None

    have_initialized: bool = False
    config: Optional['FlashMLASchedMeta.Config'] = None
    tile_scheduler_metadata: Optional[torch.Tensor] = None
    num_splits: Optional[torch.Tensor] = None


def get_mla_metadata(*args, **kwargs) -> Tuple['FlashMLASchedMeta', None]:
    """Returns an empty FlashMLASchedMeta. Metadata is lazily initialized."""
    return FlashMLASchedMeta(), None


def _dequantize_fp8_kvcache(kcache: torch.Tensor, d_qk: int) -> torch.Tensor:
    """Dequantize FP8 KV cache to BF16.

    Supports two layouts based on d_qk:
      - V32_FP8Sparse (d_qk=576): 512 FP8 NoPE + 16 bytes scale + 128 bytes RoPE
      - MODEL1_FP8Sparse (d_qk=512): 448 FP8 NoPE + 128 bytes RoPE + scales

    Returns: [num_blocks, block_size, 1, d_qk] in bfloat16
    """
    try:
        import quant
        if d_qk == 576:
            layout = quant.FP8KVCacheLayout.V32_FP8Sparse
        elif d_qk == 512:
            layout = quant.FP8KVCacheLayout.MODEL1_FP8Sparse
        else:
            raise ValueError(f"Unsupported d_qk={d_qk} for FP8 KV cache")
        return quant.dequantize_k_cache(kcache, layout)
    except ImportError:
        raise RuntimeError("quant.py module is required for FP8 KV cache dequantization")


def _sparse_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    is_fp8_kvcache: bool,
    topk_length: Optional[torch.Tensor],
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices_in_kvcache: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Python reference implementation for sparse MLA decode."""
    batch_size, seqlen_q, num_heads_q, d_qk = q.shape
    topk = indices_in_kvcache.shape[-1]

    # Dequantize KV cache if FP8
    if is_fp8_kvcache:
        kv_bf16 = _dequantize_fp8_kvcache(k_cache, d_qk)
    else:
        kv_bf16 = k_cache
    kv_flat = kv_bf16.reshape(-1, d_qk)  # [total_tokens, d_qk]

    # Gather KV by indices
    idx_fixed = torch.clamp_min(indices_in_kvcache, 0)  # [b, s_q, topk]
    gathered_kv = kv_flat.index_select(0, idx_fixed.reshape(-1)).reshape(
        batch_size, seqlen_q, topk, d_qk
    )
    invalid_mask = indices_in_kvcache == -1  # [b, s_q, topk]
    if topk_length is not None:
        arange_topk = torch.arange(0, topk, device=q.device).view(1, 1, topk)
        invalid_mask = invalid_mask | (arange_topk >= topk_length.view(batch_size, 1, 1))

    # Handle extra KV scope
    if extra_k_cache is not None and extra_indices_in_kvcache is not None:
        extra_topk = extra_indices_in_kvcache.shape[-1]
        if is_fp8_kvcache:
            extra_kv_bf16 = _dequantize_fp8_kvcache(extra_k_cache, d_qk)
        else:
            extra_kv_bf16 = extra_k_cache
        extra_kv_flat = extra_kv_bf16.reshape(-1, d_qk)
        extra_idx_fixed = torch.clamp_min(extra_indices_in_kvcache, 0)
        extra_gathered = extra_kv_flat.index_select(0, extra_idx_fixed.reshape(-1)).reshape(
            batch_size, seqlen_q, extra_topk, d_qk
        )
        extra_invalid = extra_indices_in_kvcache == -1
        if extra_topk_length is not None:
            arange_extra = torch.arange(0, extra_topk, device=q.device).view(1, 1, extra_topk)
            extra_invalid = extra_invalid | (arange_extra >= extra_topk_length.view(batch_size, 1, 1))

        gathered_kv = torch.cat([gathered_kv, extra_gathered], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid], dim=2)

    # Compute attention
    total_topk = gathered_kv.shape[2]
    gathered_kv_f = gathered_kv.reshape(batch_size * seqlen_q, total_topk, d_qk).float()
    gathered_kv_f[gathered_kv_f != gathered_kv_f] = 0.0  # NaN → 0

    q_f = q.float().reshape(batch_size * seqlen_q, num_heads_q, d_qk)
    attn_weight = q_f @ gathered_kv_f.transpose(-1, -2)  # [b*s_q, h_q, total_topk]
    attn_weight *= softmax_scale

    inv_3d = invalid_mask.reshape(batch_size * seqlen_q, 1, total_topk).expand_as(attn_weight)
    attn_weight.masked_fill_(inv_3d, float("-inf"))

    lse = attn_weight.logsumexp(dim=-1)  # [b*s_q, h_q]
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv_f[..., :head_dim_v]  # [b*s_q, h_q, d_v]

    output = output.reshape(batch_size, seqlen_q, num_heads_q, head_dim_v)
    lse = lse.reshape(batch_size, seqlen_q, num_heads_q)

    # Attention sink scaling
    if attn_sink is not None:
        sink = attn_sink.reshape(1, 1, num_heads_q)
        output = output * (1.0 / (1.0 + torch.exp(sink - lse))).unsqueeze(-1)

    # Fix lonely queries (all KV invalid → output=0, lse=+inf)
    lonely_mask = lse == float("-inf")
    output[lonely_mask.unsqueeze(-1).expand_as(output)] = 0.0
    lse[lonely_mask] = float("+inf")

    out = output.to(q.dtype)
    lse = lse.transpose(1, 2)  # [b, h_q, s_q]
    return out, lse


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: 'FlashMLASchedMeta',
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FlashMLA decode on XPU — supports both dense and sparse attention.

    Args:
        q: (batch_size, seq_len_q, num_heads_q, head_dim_k).
        k_cache: (num_blocks, page_block_size, num_heads_kv, head_dim_k).
        block_table: (batch_size, max_num_blocks_per_seq), int32. Can be None for sparse.
        cache_seqlens: (batch_size,), int32. Can be None for sparse.
        head_dim_v: V head dim (512 for DeepSeek V3).
        tile_scheduler_metadata: FlashMLASchedMeta from get_mla_metadata().
        num_splits: Must be None.
        softmax_scale: Scaling factor. Default 1/sqrt(head_dim_k).
        causal: Whether to apply causal mask (dense only).
        is_fp8_kvcache: Whether KV cache is in FP8 format.
        indices: (batch_size, seq_len_q, topk) int32. KV indices for sparse attention.
        attn_sink: (num_heads_q,) float32. Optional attention sink scaling.
        extra_k_cache: Extra KV cache for dual-scope sparse attention (not yet supported).
        extra_indices_in_kvcache: Extra indices (not yet supported).
        topk_length: (batch_size,) int32. Per-batch valid topk count.
        extra_topk_length: Extra topk length (not yet supported).

    Returns:
        (out, lse)
    """
    sched_meta = tile_scheduler_metadata
    indices_in_kvcache = indices
    assert isinstance(sched_meta, FlashMLASchedMeta)
    assert num_splits is None

    topk = indices_in_kvcache.shape[-1] if indices_in_kvcache is not None else None

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    if topk is not None:
        # Sparse attention path
        assert not causal, "causal must be False when sparse attention is enabled"

        def _run_one_scope(kc, idx, tkl):
            """Run SYCL kernel for one KV scope (no attn_sink).
            Passes FP8 data directly to kernel for in-kernel dequant."""
            o, l, _, _ = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
                q, kc, idx, head_dim_v, softmax_scale,
                is_fp8_kvcache, tkl, None,  # no attn_sink per scope
            )
            return o, l  # o: [b, s_q, h_q, d_v], l: [b, h_q, s_q]

        out0, lse0 = _run_one_scope(k_cache, indices_in_kvcache, topk_length)

        if extra_k_cache is not None and extra_indices_in_kvcache is not None:
            out1, lse1 = _run_one_scope(extra_k_cache, extra_indices_in_kvcache, extra_topk_length)
            # Online softmax merge of two scopes
            # lse shape: [b, h_q, s_q], out shape: [b, s_q, h_q, d_v]
            lse0_t = lse0.transpose(1, 2).unsqueeze(-1).float()  # [b, s_q, h_q, 1]
            lse1_t = lse1.transpose(1, 2).unsqueeze(-1).float()  # [b, s_q, h_q, 1]

            # Lonely queries have lse=+inf (kernel convention).
            # For merge, treat +inf as "no contribution": weight=0, output=0.
            lonely0 = lse0_t == float('inf')
            lonely1 = lse1_t == float('inf')

            # Replace +inf with -inf for max/exp computation (so exp(-inf - x) = 0)
            safe_lse0 = torch.where(lonely0, torch.tensor(float('-inf'), device=lse0.device), lse0_t)
            safe_lse1 = torch.where(lonely1, torch.tensor(float('-inf'), device=lse1.device), lse1_t)

            max_lse = torch.maximum(safe_lse0, safe_lse1)
            exp0 = torch.exp(safe_lse0 - max_lse)
            exp1 = torch.exp(safe_lse1 - max_lse)
            denom = exp0 + exp1
            # Both lonely → denom=0, avoid div-by-zero
            both_lonely = lonely0 & lonely1
            denom = torch.where(both_lonely, torch.ones_like(denom), denom)

            out = (out0.float() * exp0 + out1.float() * exp1) / denom
            # Both lonely → output=0 (already from kernel), lse=+inf
            out = torch.where(both_lonely, torch.zeros_like(out), out)
            lse_merged = max_lse.squeeze(-1) + torch.log(denom.squeeze(-1))
            lse_merged = torch.where(both_lonely.squeeze(-1), torch.tensor(float('inf'), device=lse0.device), lse_merged)
            lse = lse_merged.transpose(1, 2)  # [b, h_q, s_q]
            out = out.to(q.dtype)
        else:
            out, lse = out0, lse0

        # Apply attn_sink scaling after merge
        if attn_sink is not None:
            batch_size, seqlen_q, num_heads_q = out.shape[:3]
            sink = attn_sink.reshape(1, 1, num_heads_q).float()
            lse_bsqh = lse.transpose(1, 2).float()  # [b, s_q, h_q]
            # scale = 1/(1 + exp(sink - lse))
            # Edge cases: lse=+inf (lonely) → output=0, any scale is fine (use 1.0)
            #             sink=+inf, lse finite → scale=0 (output→0)
            #             sink=-inf → scale=1 (no sink effect)
            lonely = lse_bsqh == float('inf')
            safe_diff = torch.where(lonely, torch.zeros_like(lse_bsqh), sink - lse_bsqh)
            scale = 1.0 / (1.0 + torch.exp(safe_diff))
            out = (out.float() * scale.unsqueeze(-1)).to(q.dtype)
        new_sched_metadata = sched_meta.tile_scheduler_metadata
        new_num_splits = sched_meta.num_splits
    else:
        # Dense attention path
        assert block_table is not None and cache_seqlens is not None
        out, lse, new_sched_metadata, new_num_splits = (
            torch.ops.flash_attn_xpu.dense_mla_decode_fwd(
                q,
                k_cache,
                head_dim_v,
                cache_seqlens,
                block_table,
                softmax_scale,
                causal,
                sched_meta.tile_scheduler_metadata,
                sched_meta.num_splits,
            )
        )

    sched_meta.tile_scheduler_metadata = new_sched_metadata
    sched_meta.num_splits = new_num_splits
    return (out, lse)


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sparse attention prefill kernel on XPU.

    Args:
        q: [s_q, h_q, d_qk], bfloat16
        kv: [s_kv, h_kv, d_qk], bfloat16
        indices: [s_q, h_kv, topk], int32. Invalid indices should be -1 or >= s_kv.
        sm_scale: Softmax scaling factor.
        d_v: Value dimension (must be 512).
        attn_sink: Optional [h_q] float32 for sink scaling.
        topk_length: Optional [s_q] int32 for per-query topk count.

    Returns:
        (output, max_logits, lse)
    """
    if attn_sink is not None and attn_sink.device != q.device:
        attn_sink = attn_sink.to(q.device)
    if topk_length is not None and topk_length.device != q.device:
        topk_length = topk_length.to(q.device)
    results = torch.ops.flash_attn_xpu.sparse_mla_prefill_fwd(
        q, kv, indices, sm_scale, d_v, attn_sink, topk_length
    )
    return results


def flash_mla_dense_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Dense attention prefill kernel on XPU for MLA (head_dim 512/576).

    Args:
        q: [total_q, h_q, d_qk], bfloat16
        kv: [total_kv, h_kv, d_qk], bfloat16
        cu_seqlens_q: [batch+1], int32. Cumulative query sequence lengths.
        cu_seqlens_k: [batch+1], int32. Cumulative KV sequence lengths.
        sm_scale: Softmax scaling factor.
        d_v: Value dimension (e.g. 512).
        is_causal: Whether to apply causal masking.

    Returns:
        (output, max_logits, lse)
        - output: [total_q, h_q, d_v], bfloat16
        - max_logits: [total_q, h_q], float32
        - lse: [total_q, h_q], float32
    """
    results = torch.ops.flash_attn_xpu.dense_mla_prefill_fwd(
        q, kv, cu_seqlens_q, cu_seqlens_k, sm_scale, d_v, is_causal
    )
    return results
