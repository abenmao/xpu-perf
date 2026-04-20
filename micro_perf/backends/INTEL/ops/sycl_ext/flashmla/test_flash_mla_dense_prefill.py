"""Test script for MLA dense prefill kernel.

Tests the dense attention prefill with head_dim 512/576 (MLA-specific).
Compares SYCL kernel output against a Python reference implementation.
"""
import sys
import gc

import torch

DEVICE = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cuda'


def device_synchronize():
    if DEVICE == 'xpu':
        torch.xpu.synchronize()
    else:
        torch.cuda.synchronize()


def device_empty_cache():
    if DEVICE == 'xpu':
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def ref_dense_prefill(q, kv, cu_seqlens_q, cu_seqlens_k, sm_scale, d_v, is_causal):
    """Python reference implementation for dense MLA prefill.

    Args:
        q: [total_q, h_q, d_qk] bf16
        kv: [total_kv, h_kv, d_qk] bf16
        cu_seqlens_q: [batch+1] int32
        cu_seqlens_k: [batch+1] int32
        sm_scale: float
        d_v: int
        is_causal: bool

    Returns:
        out: [total_q, h_q, d_v] float32
        max_logits: [total_q, h_q] float32
        lse: [total_q, h_q] float32
    """
    total_q = q.shape[0]
    h_q = q.shape[1]
    d_qk = q.shape[2]
    h_kv = kv.shape[1]

    q_f = q.float()
    kv_f = kv.float()

    batch_size = cu_seqlens_q.shape[0] - 1
    cu_q = cu_seqlens_q.cpu().tolist()
    cu_k = cu_seqlens_k.cpu().tolist()

    out = torch.zeros(total_q, h_q, d_v, dtype=torch.float32, device=q.device)
    max_logits = torch.full((total_q, h_q), float('-inf'), dtype=torch.float32, device=q.device)
    lse = torch.full((total_q, h_q), float('+inf'), dtype=torch.float32, device=q.device)

    for b in range(batch_size):
        q_start = cu_q[b]
        q_end = cu_q[b + 1]
        k_start = cu_k[b]
        k_end = cu_k[b + 1]
        seq_q = q_end - q_start
        seq_k = k_end - k_start

        if seq_q == 0 or seq_k == 0:
            continue

        # q_batch: [seq_q, h_q, d_qk], kv_batch: [seq_k, h_kv, d_qk]
        q_batch = q_f[q_start:q_end]
        kv_batch = kv_f[k_start:k_end]

        # GQA: expand kv to match h_q
        # kv_batch: [seq_k, h_kv, d_qk] → [seq_k, h_q, d_qk]
        gqa_ratio = h_q // h_kv
        kv_expanded = kv_batch.repeat_interleave(gqa_ratio, dim=1)

        # Attention: [seq_q, h_q, d_qk] @ [seq_k, h_q, d_qk]^T → [h_q, seq_q, seq_k]
        # Use einsum for clarity
        attn = torch.einsum('qhd,khd->hqk', q_batch, kv_expanded)
        attn *= sm_scale

        # Causal mask
        if is_causal:
            # For varlen: q_pos i attends to k_pos j if j <= i
            # Here both are relative to batch start
            q_idx = torch.arange(seq_q, device=q.device).unsqueeze(1)  # [seq_q, 1]
            k_idx = torch.arange(seq_k, device=q.device).unsqueeze(0)  # [1, seq_k]
            causal_mask = k_idx > q_idx  # [seq_q, seq_k]
            attn.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))

        # Softmax + output
        ml = attn.max(dim=-1).values  # [h_q, seq_q]
        attn_lse = attn.logsumexp(dim=-1)  # [h_q, seq_q]
        attn_prob = torch.softmax(attn, dim=-1)  # [h_q, seq_q, seq_k]

        # Output: [h_q, seq_q, d_v]
        out_batch = torch.einsum('hqk,khd->qhd', attn_prob, kv_expanded[..., :d_v])

        out[q_start:q_end] = out_batch
        max_logits[q_start:q_end] = ml.permute(1, 0)  # [seq_q, h_q]
        lse[q_start:q_end] = attn_lse.permute(1, 0)

    return out, max_logits, lse


@torch.inference_mode()
def run_test(batch_size, seq_q_list, seq_k_list, h_q, h_kv, d_qk, d_v, is_causal, seed=42):
    """Run a single test case.

    Args:
        batch_size: number of sequences
        seq_q_list: list of query seq lengths per batch
        seq_k_list: list of KV seq lengths per batch
        h_q, h_kv: head counts
        d_qk: head dimension
        d_v: value dimension
        is_causal: causal masking
        seed: random seed
    """
    device_empty_cache()
    torch.manual_seed(seed)

    assert len(seq_q_list) == batch_size
    assert len(seq_k_list) == batch_size

    total_q = sum(seq_q_list)
    total_kv = sum(seq_k_list)

    print(f"  batch={batch_size}, seq_q={seq_q_list}, seq_k={seq_k_list}, "
          f"h_q={h_q}, h_kv={h_kv}, d_qk={d_qk}, d_v={d_v}, causal={is_causal}")

    q = torch.randn(total_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE)
    kv = torch.randn(total_kv, h_kv, d_qk, dtype=torch.bfloat16, device=DEVICE)

    cu_seqlens_q = torch.tensor([0] + list(torch.tensor(seq_q_list).cumsum(0).tolist()),
                                 dtype=torch.int32, device=DEVICE)
    cu_seqlens_k = torch.tensor([0] + list(torch.tensor(seq_k_list).cumsum(0).tolist()),
                                 dtype=torch.int32, device=DEVICE)

    sm_scale = d_qk ** (-0.5)

    # Run kernel
    from flash_attn_interface import flash_mla_dense_prefill_fwd
    kernel_out, kernel_ml, kernel_lse = flash_mla_dense_prefill_fwd(
        q, kv, cu_seqlens_q, cu_seqlens_k, sm_scale, d_v, is_causal
    )
    device_synchronize()

    # Run reference
    ref_out, ref_ml, ref_lse = ref_dense_prefill(
        q, kv, cu_seqlens_q, cu_seqlens_k, sm_scale, d_v, is_causal
    )
    device_synchronize()

    # Compare output using per-element check (same method as sparse prefill)
    kernel_out_f = kernel_out.float()
    ref_out_f = ref_out.float()

    # Tolerances: bf16 DPAS-based attention has ~0.01 max absolute error.
    # With small seq_k, attention weights are larger → bf16 rounding is more
    # significant. Use cosine similarity as primary check, element-wise as secondary.
    out_atol = 0.012  # covers bf16 precision for small seq_k
    out_rtol = 3.01 / 128
    lse_atol = 1e-4
    cos_diff_tol = 7e-6

    is_correct = True

    # Per-element check: pass if abs_err < atol OR rel_err < rtol
    raw_abs_err = (kernel_out_f - ref_out_f).abs()
    raw_rel_err = raw_abs_err / (ref_out_f.abs() + 1e-6)
    rel_err = raw_rel_err.masked_fill(raw_abs_err < out_atol, 0)
    abs_err = raw_abs_err.masked_fill(raw_rel_err < out_rtol, 0)
    pass_mask = (abs_err < out_atol) | (rel_err < out_rtol)

    # Cosine similarity check
    ans_d, ref_d = kernel_out_f.double(), ref_out_f.double()
    denom = (ans_d * ans_d + ref_d * ref_d).sum().item()
    cos_diff = 1 - 2 * (ans_d * ref_d).sum().item() / max(denom, 1e-12)

    out_max_diff = raw_abs_err.max().item()
    if not pass_mask.all():
        n_fail = (~pass_mask).sum().item()
        print(f"    FAIL out: max_abs={out_max_diff:.6f}, cos_diff={cos_diff:.2e}, {n_fail} elements failed")
        is_correct = False
    elif abs(cos_diff) > cos_diff_tol:
        print(f"    FAIL out: cos_diff={cos_diff:.2e} > {cos_diff_tol:.1e}")
        is_correct = False
    else:
        print(f"    PASS out: max_abs={out_max_diff:.6f}, cos_diff={cos_diff:.2e}")

    # Check LSE (skip +inf entries — those are empty sequences)
    valid = (ref_lse != float('+inf')) & (ref_lse != float('-inf'))
    if valid.any():
        lse_diff = (kernel_lse[valid] - ref_lse[valid]).abs()
        lse_max_diff = lse_diff.max().item()
        if lse_max_diff > lse_atol:
            print(f"    FAIL lse: max_abs={lse_max_diff:.6f}")
            is_correct = False
        else:
            print(f"    PASS lse: max_abs={lse_max_diff:.6f}")

    # Check max_logits
    if valid.any():
        ml_diff = (kernel_ml[valid] - ref_ml[valid]).abs()
        ml_max_diff = ml_diff.max().item()
        if ml_max_diff > lse_atol:
            print(f"    FAIL max_logits: max_abs={ml_max_diff:.6f}")
            is_correct = False
        else:
            print(f"    PASS max_logits: max_abs={ml_max_diff:.6f}")

    return is_correct


if __name__ == '__main__':
    device = torch.device(f"{DEVICE}:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    if DEVICE == 'xpu':
        torch.xpu.set_device(device)
    else:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    all_passed = True
    test_id = 0

    # Test configurations generated via comprehension
    # (batch, seq_q_list, seq_k_list, h_q, h_kv, d_qk, d_v, is_causal)

    # Basic single-batch tests: all combos of d_qk × causal
    basic_configs = [
        (1, [seq_q], [seq_k], h_q, h_kv, d_qk, 512, causal)
        for seq_q, seq_k in [(4, 8), (1, 8), (8, 8), (16, 32)]
        for h_q, h_kv in [(128, 1)]
        for d_qk in [512, 576]
        for causal in [False, True]
    ]

    # Multi-batch tests
    multi_batch_configs = [
        (batch, sq_list, sk_list, 128, 1, d_qk, 512, causal)
        for batch, sq_list, sk_list in [
            (2, [4, 3], [8, 6]),
            (3, [2, 4, 1], [6, 8, 3]),
            (4, [8, 4, 12, 6], [16, 8, 24, 12]),
        ]
        for d_qk in [512, 576]
        for causal in [False, True]
    ]

    # Long sequences: tests multi TILE_K (=64) iterations + online softmax rescaling
    long_seq_configs = [
        (1, [seq_q], [seq_k], 128, 1, d_qk, 512, causal)
        for seq_q, seq_k in [
            (32, 128), (64, 256), (128, 512), (64, 1024),
            (128, 2048), (64, 4096), (32, 8192), (16, 16384),
        ]
        for d_qk in [512, 576]
        for causal in [False, True]
    ]

    # Boundary cases: TILE_K remainder, square causal, multi-batch long
    boundary_configs = [
        # TILE_K remainder (seq_k=65 = 64+1)
        (1, [16], [65], 128, 1, d_qk, 512, causal)
        for d_qk in [512, 576]
        for causal in [False, True]
    ] + [
        # Square causal attention
        (1, [n], [n], 128, 1, d_qk, 512, True)
        for n in [128, 256]
        for d_qk in [512, 576]
    ] + [
        # Multi-batch with long sequences
        (2, [64, 32], [256, 128], 128, 1, d_qk, 512, causal)
        for d_qk in [512, 576]
        for causal in [False, True]
    ]

    test_configs = basic_configs + multi_batch_configs + long_seq_configs + boundary_configs

    for config in test_configs:
        test_id += 1
        batch, sq_list, sk_list, hq, hkv, dqk, dv, causal = config
        print(f"\nTest {test_id}:")
        try:
            passed = run_test(batch, sq_list, sk_list, hq, hkv, dqk, dv, causal)
            if not passed:
                all_passed = False
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
        gc.collect()
        device_empty_cache()

    print("\n" + "=" * 60)
    if all_passed:
        print(f"ALL {test_id} TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
