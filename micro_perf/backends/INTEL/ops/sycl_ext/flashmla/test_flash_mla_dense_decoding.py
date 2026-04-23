import argparse
import math
import random
import dataclasses
from typing import Tuple

import torch

import kernelkit as kk

from flash_attn_interface import flash_mla_with_kvcache, get_mla_metadata

DEVICE = 'xpu' if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'cuda'

@dataclasses.dataclass
class TestParam:
    b: int    # Batch size
    s_q: int  # Number of queries for one request
    s_k: int  # Seq len, or mean seq len if varlen == True
    is_varlen: bool
    is_causal: bool
    test_performance: bool = True
    have_zero_seqlen_k: bool = False
    block_size: int = 64
    h_q: int = 128    # Number of q heads
    h_kv: int = 1     # Number of kv heads
    d: int = 576      # Q/K head dim (= dv + RoPE dim)
    dv: int = 512     # V head dim
    seed: int = 0


def device_synchronize():
    if DEVICE == 'xpu':
        torch.xpu.synchronize()
    else:
        torch.cuda.synchronize()


def generate_test_data(t: TestParam) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate test data from a given configuration
    Return: [cache_seqlens, q, block_table, blocked_k]
    Pay attention: This function changes the random seed
    """
    random.seed(t.seed)
    torch.manual_seed(t.seed)

    assert t.h_q % t.h_kv == 0

    cache_seqlens_cpu = torch.full((t.b,), t.s_k, dtype=torch.int32, device='cpu')
    if t.is_varlen:
        for i in range(t.b):
            cache_seqlens_cpu[i] = max(random.normalvariate(t.s_k, t.s_k / 2), t.s_q)

    if t.have_zero_seqlen_k:
        zeros_mask = torch.randn(t.b, dtype=torch.float32, device='cpu') > 0
        cache_seqlens_cpu[zeros_mask] = 0

    max_seqlen = int(cache_seqlens_cpu.max().item())
    max_seqlen_pad = kk.cdiv(max_seqlen, 256) * 256
    cache_seqlens = cache_seqlens_cpu.to(DEVICE)

    q = torch.randn(t.b, t.s_q, t.h_q, t.d) / 10
    q.clamp_(min=-1.0, max=1.0)

    block_table = torch.arange(t.b * max_seqlen_pad // t.block_size, dtype=torch.int32).view(t.b, max_seqlen_pad // t.block_size)
    block_table = block_table.view(-1)[torch.randperm(block_table.numel())].view(t.b, -1)
    blocked_k = torch.randn(block_table.numel(), t.block_size, t.h_kv, t.d) / 10
    blocked_k.clamp_(min=-1.0, max=1.0)

    for i in range(t.b):
        cur_len = int(cache_seqlens_cpu[i].item())
        cur_num_blocks = kk.cdiv(cur_len, t.block_size)
        blocked_k[block_table[i][cur_num_blocks:]] = float("nan")
        if cur_len % t.block_size != 0:
            blocked_k[block_table[i][cur_num_blocks - 1]][cur_len % t.block_size:] = float("nan")
        block_table[i][cur_num_blocks:] = 2147480000
    return cache_seqlens, q, block_table, blocked_k


def reference_torch(
    cache_seqlens: torch.Tensor,    # [batch_size]
    block_table: torch.Tensor,      # [batch_size, ?]
    q: torch.Tensor,    # [batch_size, s_q, h_q, d]
    blocked_k: torch.Tensor,    # [?, block_size, h_kv, d]
    dv: int,
    is_causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation in PyTorch running on the input device.
    Uses GQA broadcasting (instead of repeat_interleave) to avoid OOM.
    """
    b, s_q, h_q, d = q.size()
    block_size = blocked_k.size(1)
    h_kv = blocked_k.size(2)
    gqa_ratio = h_q // h_kv
    dev = q.device

    cache_seqlens_cpu = cache_seqlens.cpu()

    out_ref = torch.zeros(b, s_q, h_q, dv, dtype=torch.float32, device=dev)
    lse_ref = torch.full((b, h_q, s_q), float("+inf"), dtype=torch.float32, device=dev)

    for i in range(b):
        cur_len = int(cache_seqlens_cpu[i].item())
        if cur_len == 0:
            continue

        cur_num_blocks = kk.cdiv(cur_len, block_size)
        cur_block_indices = block_table[i, :cur_num_blocks]
        # [num_blocks*block_size, h_kv, d] -> [cur_len, h_kv, d]
        cur_kv = blocked_k[cur_block_indices].reshape(-1, h_kv, d)[:cur_len].float()
        cur_kv = torch.nan_to_num(cur_kv, nan=0.0)

        # Q: [s_q, h_q, d] -> [h_kv, gqa_ratio, s_q, d]
        cur_q = q[i].float().view(s_q, h_kv, gqa_ratio, d).permute(1, 2, 0, 3)

        # KV: [cur_len, h_kv, d] -> [h_kv, 1, d, cur_len]
        # attn_weight: [h_kv, gqa_ratio, s_q, cur_len]  (broadcast over gqa_ratio)
        attn_weight = cur_q @ cur_kv.permute(1, 0, 2).unsqueeze(1).transpose(-2, -1)
        attn_weight /= math.sqrt(d)

        if is_causal and s_q > 1:
            causal_mask = torch.ones(s_q, cur_len, dtype=torch.bool, device=dev)
            causal_mask = causal_mask.tril(diagonal=cur_len - s_q)
            attn_weight.masked_fill_(~causal_mask, float("-inf"))

        lse_i = attn_weight.logsumexp(dim=-1)  # [h_kv, gqa_ratio, s_q]
        attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)

        # V: [h_kv, 1, cur_len, dv]
        cur_v = cur_kv[:, :, :dv].permute(1, 0, 2).unsqueeze(1)
        output = attn_weight @ cur_v  # [h_kv, gqa_ratio, s_q, dv]

        # Handle lonely Q tokens (no attendable K)
        lonely = (lse_i == float("-inf"))
        output[lonely.unsqueeze(-1).expand_as(output)] = 0.0
        lse_i[lonely] = float("+inf")

        # [h_kv, gqa_ratio, s_q, dv] -> [s_q, h_q, dv]
        out_ref[i] = output.permute(2, 0, 1, 3).reshape(s_q, h_q, dv)
        # [h_kv, gqa_ratio, s_q] -> [h_q, s_q]
        lse_ref[i] = lse_i.reshape(h_q, s_q)

    out_ref = out_ref.to(q.dtype)
    return out_ref, lse_ref

@torch.inference_mode()
def test_flash_mla(t: TestParam):
    print('-------------------------------')
    print(f"Running on {t}...")

    # Generating test data
    device_synchronize()
    cache_seqlens, q, block_table, blocked_k, = generate_test_data(t)

    tile_scheduler_metadata, num_splits = get_mla_metadata()

    def run_flash_mla():
        return flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            t.dv,
            tile_scheduler_metadata,
            num_splits,
            causal=t.is_causal
        )

    out_ans, lse_ans = run_flash_mla()
    out_ref, lse_ref = reference_torch(cache_seqlens, block_table, q, blocked_k, t.dv, t.is_causal)
    is_correct = True
    is_correct &= kk.check_is_allclose("out", out_ans.cpu(), out_ref.cpu(), abs_tol=8e-4, rel_tol=2.01 / 128, cos_diff_tol=6e-6)
    is_correct &= kk.check_is_allclose("lse", lse_ans.cpu(), lse_ref.cpu(), abs_tol=1e-6, rel_tol=8.01 / 65536)
    assert is_correct

    if t.test_performance:
        # Simple timing for XPU (no kineto/CUPTI)
        warmup = 5
        repeat = 10
        for _ in range(warmup):
            run_flash_mla()
        device_synchronize()

        import time
        start = time.perf_counter()
        for _ in range(repeat):
            run_flash_mla()
        device_synchronize()
        time_usage = (time.perf_counter() - start) / repeat

        mean_attended_seqlens = cache_seqlens.float().mean().item()
        compute_volume_flop = t.b * t.h_q * t.s_q * sum([
            2 * t.d * mean_attended_seqlens,   # Q * K^T
            2 * mean_attended_seqlens * t.dv,  # attention * V
        ])
        q_elem_size = torch.bfloat16.itemsize
        kv_token_size = t.d * torch.bfloat16.itemsize
        memory_volume_B = t.b * sum([
            t.s_q * t.h_q * (t.d * q_elem_size),    # Q
            mean_attended_seqlens * t.h_kv * kv_token_size,    # K/V
            t.s_q * t.h_q * (t.dv * q_elem_size),   # Output
        ])
        achieved_tflops = compute_volume_flop / time_usage / 1e12
        achieved_gBps = memory_volume_B / time_usage / 1e9

        print(f"{time_usage * 1000:.3f} ms, {achieved_tflops:.0f} TFLOPS, {achieved_gBps:.0f} GB/s")


def main(torch_dtype):
    device = torch.device(f"{DEVICE}:0")
    torch.set_default_dtype(torch_dtype)
    torch.set_default_device(device)

    correctness_cases = [
        TestParam(b, s_q, s_k, is_varlen, is_causal, test_performance=False, have_zero_seqlen_k=False, block_size=64, h_q=h_q, h_kv=h_kv)
        for b in [1, 2, 6, 64]
        for s_q in [1, 2, 4]
        for s_k in [20, 140, 4096]
        for h_q in [1, 3, 9, 63, 64, 126, 128]
        for h_kv in [1, 2, 3, 8]
        for is_varlen in [False, True]
        for is_causal in [False, True]
        if h_q % h_kv == 0
    ]

    corner_cases = [
        # Cases where some kv cache have zero length
        TestParam(128, 2, 4096, is_varlen=True, is_causal=is_causal, test_performance=False, have_zero_seqlen_k=True, h_q=h_q, h_kv=h_kv)
        for h_q in [1, 3, 9, 63, 64, 126, 128]
        for h_kv in [1, 2, 3, 8]
        for is_causal in [False, True]
        if h_q % h_kv == 0
    ]

    performance_cases = [
        TestParam(128, s_q, s_k, is_varlen=True, is_causal=is_causal, test_performance=True)
        for is_causal in [False, True]
        for s_q in [1, 2]
        for s_k in [4096, 8192, 16384, 32768]
    ]

    testcases = correctness_cases + corner_cases + performance_cases

    for testcase in testcases:
        test_flash_mla(testcase)
        if DEVICE == "xpu":
            import gc; gc.collect()
            torch.xpu.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type to use for testing (bf16 or fp16)",
    )

    args = parser.parse_args()

    torch_dtype = torch.bfloat16
    if args.dtype == "fp16":
        torch_dtype = torch.float16

    main(torch_dtype)