from typing import Optional, Tuple

import gc
import torch

from lib import TestParam, Testcase, TestcaseForDecode, KVScope, DEVICE


def _get_free_mem(reserve_bytes: int = 2 * 1024**3) -> int:
    """Return usable device memory in bytes (free - reserve)."""
    if DEVICE == 'xpu':
        free, _ = torch.xpu.mem_get_info()
    else:
        free, _ = torch.cuda.mem_get_info()
    return max(0, free - reserve_bytes)


def _merge_two_lse(lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([
                lse0.view(s_q, h_q),
                lse1.broadcast_to(s_q, h_q)
            ], dim=0),
            dim=0
        )

        
def ref_sparse_attn_fwd(p: TestParam, t: Testcase) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs on XPU with chunked processing to avoid OOM.
    Returns (on CPU):
    - o: [s_q, h_q, dv] bf16
    - o_fp32: [s_q, h_q, dv] float32
    - max_logits: [s_q, h_q] float32
    - lse: [s_q, h_q] float32
    """
    dev = t.q.device

    # Prepare indices and invalid_mask on device (small tensors)
    indices = t.indices.clone().squeeze(1)  # [s_q, topk]
    if t.topk_length is not None:
        mask = torch.arange(p.topk, device=dev).unsqueeze(0).expand(p.s_q, p.topk) >= t.topk_length.unsqueeze(1)
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= p.s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    # Estimate per-query memory (float32): gathered_kv + P + softmax temps + output
    # Multiply by 3 for intermediate tensors (matmul temps, softmax, exp, etc.)
    per_q_bytes = (p.topk * p.d_qk + p.h_q * p.topk + p.h_q * p.d_v) * 4 * 3
    free = _get_free_mem()
    chunk = max(0, free // per_q_bytes) if per_q_bytes > 0 else p.s_q
    chunk = min(chunk, p.s_q)

    if chunk == 0:
        raise MemoryError(
            f"[ref_sparse_attn_fwd] Not enough device memory for even 1 query. "
            f"Need {per_q_bytes/1e6:.1f}MB, free {free/1e6:.1f}MB. Skipping."
        )

    # Pre-allocate CPU output buffers
    out_cpu = torch.zeros(p.s_q, p.h_q, p.d_v, dtype=torch.float32, device='cpu')
    max_logits_cpu = torch.full((p.s_q, p.h_q), float('-inf'), dtype=torch.float32, device='cpu')
    lse_cpu = torch.full((p.s_q, p.h_q), float('-inf'), dtype=torch.float32, device='cpu')

    attn_sink = t.attn_sink  # [h_q] on device or None

    for start in range(0, p.s_q, chunk):
        end = min(start + chunk, p.s_q)
        clen = end - start

        # Gather KV for this chunk
        idx_chunk = indices[start:end]  # [clen, topk]
        inv_chunk = invalid_mask[start:end]  # [clen, topk]
        gathered = t.kv.index_select(0, idx_chunk.flatten()).reshape(clen, p.topk, p.d_qk).float()
        q_chunk = t.q[start:end].float()  # [clen, h_q, d_qk]

        P = q_chunk @ gathered.transpose(1, 2)  # [clen, h_q, topk]
        P *= t.sm_scale
        P[inv_chunk.unsqueeze(1).expand_as(P)] = float('-inf')

        orig_lse = torch.logsumexp(P, dim=-1)  # [clen, h_q]
        ml = P.max(dim=-1).values  # [clen, h_q]

        lse_for_o = _merge_two_lse(orig_lse, attn_sink, clen, p.h_q)
        if not torch.is_inference_mode_enabled():
            lse_for_o = lse_for_o.clone()
        lse_for_o[lse_for_o == float('-inf')] = float('+inf')
        s = torch.exp(P - lse_for_o.unsqueeze(-1))
        o = s @ gathered[..., :p.d_v]  # [clen, h_q, d_v]

        lonely = orig_lse == float('-inf')
        orig_lse[lonely] = float('+inf')

        # Copy to CPU immediately
        out_cpu[start:end] = o.cpu()
        max_logits_cpu[start:end] = ml.cpu()
        lse_cpu[start:end] = orig_lse.cpu()

        del gathered, q_chunk, P, orig_lse, ml, lse_for_o, s, o
    
    del indices, invalid_mask

    return (out_cpu.to(torch.bfloat16), out_cpu, max_logits_cpu, lse_cpu)


def ref_sparse_attn_decode(
    p: TestParam,
    t: TestcaseForDecode
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference sparse decoding attention on XPU with chunked batch processing.
    Gather + matmul are done per-chunk to avoid allocating huge intermediates.
    """
    assert p.h_kv == 1
    assert p.decode is not None
    b = p.decode.b
    dev = t.q.device
    n = b * p.s_q

    # Build gather info for each kv_scope (indices + invalid_mask stay on device, small)
    scopes = [t.kv_scope]
    if t.extra_kv_scope is not None:
        scopes.append(t.extra_kv_scope)

    # Compute total topk across all scopes
    total_topk = sum(sc.indices_in_kvcache.size(-1) for sc in scopes)

    # Per-sample memory: gathered_kv_chunk [total_topk, d_qk]*4 + attn_weight [h_q, total_topk]*4 + out [h_q, d_v]*4
    # Multiply by 3 for intermediate tensors (boolean masks, matmul temps, softmax temps)
    per_sample_bytes = (total_topk * p.d_qk + p.h_q * total_topk + p.h_q * p.d_v) * 4 * 3
    free = _get_free_mem()
    chunk = max(0, free // per_sample_bytes) if per_sample_bytes > 0 else n
    chunk = min(chunk, n)

    if chunk == 0:
        raise MemoryError(
            f"[ref_sparse_attn_decode] Not enough device memory for even 1 sample. "
            f"Need {per_sample_bytes/1e6:.1f}MB, free {free/1e6:.1f}MB. Skipping."
        )

    q_flat = t.q.float().view(n, p.h_q, p.d_qk)
    out_cpu = torch.zeros(n, p.h_q, p.d_v, dtype=torch.float32, device='cpu')
    lse_cpu = torch.full((n, p.h_q), float('-inf'), dtype=torch.float32, device='cpu')

    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        clen = end - start
        # Map flat indices back to (batch, sq) for gather
        b_start = start // p.s_q
        b_end = (end - 1) // p.s_q + 1

        # Gather KV for this chunk across all scopes
        gathered_parts = []
        inv_parts = []
        for sc in scopes:
            topk_sc = sc.indices_in_kvcache.size(-1)
            idx_flat = sc.indices_in_kvcache.view(n, topk_sc)[start:end]  # [clen, topk_sc]
            idx_fixed = torch.clamp_min(idx_flat, 0)
            g = sc.blocked_k.view(-1, p.d_qk).index_select(0, idx_fixed.view(-1)).view(clen, topk_sc, p.d_qk)
            inv = idx_flat == -1
            if sc.topk_length is not None:
                # topk_length is [b], expand to [b, s_q] then flatten
                tkl = sc.topk_length.unsqueeze(1).expand(b, p.s_q).reshape(n)[start:end]  # [clen]
                inv = inv | (torch.arange(topk_sc, device=dev).view(1, topk_sc).expand(clen, topk_sc) >= tkl.view(clen, 1))
            gathered_parts.append(g)
            inv_parts.append(inv)

        gathered = torch.cat(gathered_parts, dim=1).float()  # [clen, total_topk, d_qk]
        gathered = torch.nan_to_num(gathered, nan=0.0)
        inv = torch.cat(inv_parts, dim=1)  # [clen, total_topk]
        del gathered_parts, inv_parts

        q_c = q_flat[start:end]  # [clen, h_q, d_qk]
        aw = q_c @ gathered.transpose(-1, -2)  # [clen, h_q, total_topk]
        aw *= t.sm_scale
        aw[inv.unsqueeze(1).expand_as(aw)] = float('-inf')
        l = aw.logsumexp(dim=-1)  # [clen, h_q]
        aw = torch.exp(aw - l.unsqueeze(-1))
        o = aw @ gathered[..., :p.d_v]  # [clen, h_q, d_v]

        out_cpu[start:end] = o.cpu()
        lse_cpu[start:end] = l.cpu()
        del gathered, inv, q_c, aw, l, o

    del q_flat

    output = out_cpu.view(b, p.s_q, p.h_q, p.d_v)
    lse = lse_cpu.view(b, p.s_q, p.h_q)

    # Attention sink
    if t.attn_sink is not None:
        attn_sink_cpu = t.attn_sink.cpu()
        output *= (1.0 / (1.0 + torch.exp(attn_sink_cpu.view(1, 1, p.h_q) - lse))).unsqueeze(-1)

    # Correct for q tokens which has no attendable k
    lonely_q_mask = (lse == float("-inf"))
    output[lonely_q_mask.unsqueeze(-1).expand(b, p.s_q, p.h_q, p.d_v)] = 0.0
    lse[lonely_q_mask] = float("+inf")

    return output.to(torch.bfloat16), lse.transpose(1, 2)
