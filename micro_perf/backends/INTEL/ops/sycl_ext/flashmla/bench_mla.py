"""
Performance benchmark for all 4 MLA kernels on XPU.

Usage:
    python bench_mla.py                    # Run all benchmarks
    python bench_mla.py --kernel dense_decode
    python bench_mla.py --kernel sparse_prefill --no-kernel-time
    python bench_mla.py --warmup 10 --runs 50

Timing methods:
  1. API time:    torch.xpu.Event (measures host→device→host round-trip)
  2. Kernel time: torch.profiler + Chrome trace parsing (GPU-only execution)
"""

import argparse
import dataclasses
import json
import math
import os
import pathlib
import random
import shutil
import tempfile
from typing import Callable, Dict, List, Optional, Tuple

import torch

torch.ops.load_library("flash_attn_xpu.abi3.so")

from flash_attn_interface import (
    FlashMLASchedMeta,
    flash_mla_dense_prefill_fwd,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
import quant
import kernelkit as kk

DEVICE = "xpu"


# ============================================================================
# Timing utilities
# ============================================================================

def device_synchronize():
    torch.xpu.synchronize()


def device_empty_cache():
    torch.xpu.empty_cache()


def bench_api_time(fn: Callable, num_warmups: int, num_runs: int, flush_l2: bool = True) -> float:
    """Measure API-level latency using torch.xpu.Event. Returns seconds per call."""
    for _ in range(num_warmups):
        fn()
    device_synchronize()

    start_events = [torch.xpu.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(num_runs)]

    for i in range(num_runs):
        if flush_l2:
            torch.empty(int(256e6 // 4), dtype=torch.int32, device=DEVICE).zero_()
        start_events[i].record()
        fn()
        end_events[i].record()

    device_synchronize()
    total_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    return total_ms / num_runs / 1000.0  # seconds


def _is_mla_kernel(name: str) -> bool:
    """Check if a kernel name is an MLA attention kernel (not a PyTorch helper)."""
    # CUTLASS/FMHA kernels contain these patterns in their mangled names
    mla_patterns = ["cutlass", "fmha", "sparse_attn", "MhaName", "Mha"]
    return any(p in name for p in mla_patterns)


def bench_kernel_time(fn: Callable, num_warmups: int, num_runs: int) -> Tuple[float, List[str]]:
    """Measure kernel-level latency via torch.profiler Chrome trace.

    Filters to only count MLA attention kernels (CUTLASS/FMHA), excluding
    PyTorch elementwise helper kernels.

    Returns (avg_kernel_seconds, [kernel_names]).
    """
    trace_dir = pathlib.Path(tempfile.mkdtemp(prefix="bench_mla_"))
    trace_file = trace_dir / "trace.json"

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.XPU],
            schedule=torch.profiler.schedule(
                wait=0, warmup=num_warmups, active=num_runs, repeat=1
            ),
            on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_file)),
        ) as prof:
            for _ in range(num_warmups + num_runs):
                fn()
                device_synchronize()
                prof.step()

        # Parse trace JSON — only count MLA kernels
        mla_kernel_us = 0.0
        all_kernel_us = 0.0
        mla_kernel_names = {}
        if trace_file.exists():
            data = json.loads(trace_file.read_text())
            for event in data.get("traceEvents", []):
                if event.get("cat", None) in ["kernel", "gpu_memcpy"]:
                    name = event["name"]
                    dur = event["dur"]  # microseconds
                    all_kernel_us += dur
                    if _is_mla_kernel(name):
                        mla_kernel_us += dur
                        if name not in mla_kernel_names:
                            mla_kernel_names[name] = []
                        mla_kernel_names[name].append(dur)

            mla_kernel_us /= num_runs
            all_kernel_us /= num_runs

        return mla_kernel_us / 1e6, list(mla_kernel_names.keys())  # seconds
    finally:
        if trace_dir.exists():
            shutil.rmtree(trace_dir, ignore_errors=True)


# ============================================================================
# FLOPS calculation
# ============================================================================

def compute_flops(h_q: int, seq_q: int, seq_k: int, d_qk: int, d_v: int,
                  batch: int = 1, is_causal: bool = False) -> float:
    """Compute total FLOPs for attention.

    QK^T: 2 * seq_q * seq_k * d_qk
    softmax: ~5 * seq_q * seq_k
    PV: 2 * seq_q * seq_k * d_v
    Per head, then multiply by h_q and batch.
    For causal, effective seq_k ≈ seq_k * seq_q / (seq_q + seq_k) on average.
    """
    effective_sk = seq_k
    if is_causal and seq_q > 1:
        # Triangular mask: average attended tokens ≈ (seq_k - seq_q/2) per query
        # More precisely: sum_{i=0}^{sq-1} min(sk, sk-sq+i+1) / sq
        effective_sk = seq_k - seq_q / 2.0
    flops_per_head = seq_q * effective_sk * (2 * d_qk + 2 * d_v + 5)
    return batch * h_q * flops_per_head


def compute_mem_bytes(h_q: int, h_kv: int, seq_q: int, seq_k: int, d_qk: int, d_v: int,
                      batch: int = 1, is_fp8: bool = False, kv_token_bytes: Optional[int] = None) -> float:
    """Estimate memory traffic in bytes."""
    # Q read
    q_bytes = batch * seq_q * h_q * d_qk * 2  # bf16
    # KV read
    if kv_token_bytes is not None:
        kv_bytes = batch * seq_k * h_kv * kv_token_bytes
    elif is_fp8:
        kv_bytes = batch * seq_k * h_kv * (656 if d_qk == 576 else 576)
    else:
        kv_bytes = batch * seq_k * h_kv * d_qk * 2  # bf16
    # Output write
    o_bytes = batch * seq_q * h_q * d_v * 2  # bf16
    return q_bytes + kv_bytes + o_bytes


# ============================================================================
# Result formatting
# ============================================================================

@dataclasses.dataclass
class BenchResult:
    config_str: str
    flops: float
    mem_bytes: float
    api_time_s: float
    kernel_time_s: Optional[float] = None
    kernel_names: Optional[List[str]] = None


def print_results(title: str, results: List[BenchResult]):
    print(f"\n{'=' * 110}")
    print(f"  {title}")
    print(f"{'=' * 110}")

    has_kernel = any(r.kernel_time_s is not None for r in results)

    if has_kernel:
        header = f"{'Config':<50} | {'API(us)':>8} | {'Kern(us)':>8} | {'API TF/s':>8} | {'Kern TF/s':>9} | {'BW(GB/s)':>8}"
    else:
        header = f"{'Config':<50} | {'API(us)':>8} | {'API TF/s':>8} | {'BW(GB/s)':>8}"

    print(header)
    print("-" * len(header))

    for r in results:
        api_us = r.api_time_s * 1e6
        api_tflops = r.flops / r.api_time_s / 1e12 if r.api_time_s > 0 else 0
        api_bw = r.mem_bytes / r.api_time_s / 1e9 if r.api_time_s > 0 else 0

        if has_kernel and r.kernel_time_s is not None and r.kernel_time_s > 0:
            kern_us = r.kernel_time_s * 1e6
            kern_tflops = r.flops / r.kernel_time_s / 1e12
            print(f"{r.config_str:<50} | {api_us:>8.1f} | {kern_us:>8.1f} | {api_tflops:>8.2f} | {kern_tflops:>9.2f} | {api_bw:>8.1f}")
        elif has_kernel:
            print(f"{r.config_str:<50} | {api_us:>8.1f} | {'N/A':>8} | {api_tflops:>8.2f} | {'N/A':>9} | {api_bw:>8.1f}")
        else:
            print(f"{r.config_str:<50} | {api_us:>8.1f} | {api_tflops:>8.2f} | {api_bw:>8.1f}")

    print()


# ============================================================================
# Dense Decode Benchmark
# ============================================================================

def bench_dense_decode(num_warmups: int, num_runs: int, do_kernel_time: bool) -> List[BenchResult]:
    configs = [
        # (batch, s_q, s_k, h_q, h_kv, d_qk, d_v, causal, block_size)
        (1,   1, 1024,  64, 1, 576, 512, False, 64),
        (4,   1, 4096,  64, 1, 576, 512, False, 64),
        (16,  1, 16384, 64, 1, 576, 512, False, 64),
        (1,   1, 1024, 128, 1, 512, 512, False, 64),
        (4,   1, 4096, 128, 1, 512, 512, False, 64),
        (16,  1, 16384,128, 1, 512, 512, False, 64),
        (4,   4, 4096,  64, 1, 576, 512, False, 64),
    ]

    results = []
    for b, s_q, s_k, h_q, h_kv, d_qk, d_v, causal, block_size in configs:
        device_empty_cache()
        torch.manual_seed(42)

        config_str = f"b={b},sq={s_q},sk={s_k},hq={h_q},d={d_qk},c={'T' if causal else 'F'}"

        # Generate data
        cache_seqlens = torch.full((b,), s_k, dtype=torch.int32, device=DEVICE)
        q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10

        max_seqlen_pad = kk.cdiv(s_k, 256) * 256
        n_blocks_per_seq = max_seqlen_pad // block_size
        total_blocks = b * n_blocks_per_seq
        block_table = torch.arange(total_blocks, dtype=torch.int32, device=DEVICE).view(b, n_blocks_per_seq)
        blocked_k = torch.randn(total_blocks, block_size, h_kv, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10

        sched_meta, _ = get_mla_metadata()
        sm_scale = d_qk ** (-0.5)

        def run():
            return flash_mla_with_kvcache(
                q, blocked_k, block_table, cache_seqlens, d_v,
                sched_meta, None, softmax_scale=sm_scale, causal=causal,
            )

        # Warmup to init scheduler metadata
        run()
        device_synchronize()

        flops = compute_flops(h_q, s_q, s_k, d_qk, d_v, batch=b, is_causal=causal)
        mem_bytes = compute_mem_bytes(h_q, h_kv, s_q, s_k, d_qk, d_v, batch=b)

        api_time = bench_api_time(run, num_warmups, num_runs)
        kernel_time = None
        kernel_names = None
        if do_kernel_time:
            kernel_time, kernel_names = bench_kernel_time(run, num_warmups, num_runs)

        results.append(BenchResult(config_str, flops, mem_bytes, api_time, kernel_time, kernel_names))

    return results


# ============================================================================
# Dense Prefill Benchmark
# ============================================================================

def bench_dense_prefill(num_warmups: int, num_runs: int, do_kernel_time: bool) -> List[BenchResult]:
    configs = [
        # (batch, seq_q, seq_k, h_q, h_kv, d_qk, d_v, causal)
        (1,  64,  1024,  64, 1, 576, 512, False),
        (1,  64,  1024,  64, 1, 576, 512, True),
        (1, 256,  4096,  64, 1, 576, 512, False),
        (1, 256,  4096,  64, 1, 576, 512, True),
        (4,  64,  1024, 128, 1, 512, 512, False),
        (4,  64,  4096, 128, 1, 512, 512, False),
        (1,  64, 16384,  64, 1, 576, 512, False),
    ]

    results = []
    for batch, seq_q, seq_k, h_q, h_kv, d_qk, d_v, causal in configs:
        device_empty_cache()
        torch.manual_seed(42)

        config_str = f"b={batch},sq={seq_q},sk={seq_k},hq={h_q},d={d_qk},c={'T' if causal else 'F'}"

        total_q = batch * seq_q
        total_kv = batch * seq_k
        q = torch.randn(total_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10
        kv = torch.randn(total_kv, h_kv, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10

        cu_seqlens_q = torch.tensor(
            [i * seq_q for i in range(batch + 1)], dtype=torch.int32, device=DEVICE
        )
        cu_seqlens_k = torch.tensor(
            [i * seq_k for i in range(batch + 1)], dtype=torch.int32, device=DEVICE
        )

        sm_scale = d_qk ** (-0.5)

        def run():
            return flash_mla_dense_prefill_fwd(q, kv, cu_seqlens_q, cu_seqlens_k, sm_scale, d_v, causal)

        run()
        device_synchronize()

        flops = compute_flops(h_q, seq_q, seq_k, d_qk, d_v, batch=batch, is_causal=causal)
        mem_bytes = compute_mem_bytes(h_q, h_kv, seq_q, seq_k, d_qk, d_v, batch=batch)

        api_time = bench_api_time(run, num_warmups, num_runs)
        kernel_time = None
        kernel_names = None
        if do_kernel_time:
            kernel_time, kernel_names = bench_kernel_time(run, num_warmups, num_runs)

        results.append(BenchResult(config_str, flops, mem_bytes, api_time, kernel_time, kernel_names))

    return results


# ============================================================================
# Sparse Decode Benchmark
# ============================================================================

def bench_sparse_decode(num_warmups: int, num_runs: int, do_kernel_time: bool) -> List[BenchResult]:
    configs = [
        # (batch, topk, s_kv, h_q, h_kv, d_qk, d_v, is_fp8, block_size)
        (1,   256,  16384,  64, 1, 576, 512, False, 64),
        (4,   256,  32768,  64, 1, 576, 512, False, 64),
        (16, 1024,  65536,  64, 1, 576, 512, False, 64),
        (4,   256,  32768, 128, 1, 512, 512, False, 64),
        (4,   256,  32768,  64, 1, 576, 512, True,  64),
        (16, 1024,  65536,  64, 1, 576, 512, True,  64),
        (4,  1024, 131072, 128, 1, 512, 512, True,  64),
    ]

    results = []
    for b, topk, s_kv, h_q, h_kv, d_qk, d_v, is_fp8, block_size in configs:
        device_empty_cache()
        torch.manual_seed(42)
        s_q = 1

        config_str = f"b={b},topk={topk},skv={s_kv},hq={h_q},d={d_qk},fp8={'T' if is_fp8 else 'F'}"

        q = torch.randn(b, s_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10

        # Generate blocked KV cache
        max_seqlen_pad = kk.cdiv(s_kv, 4 * block_size) * (4 * block_size)
        n_blocks_per_seq = max_seqlen_pad // block_size
        total_blocks = b * n_blocks_per_seq
        blocked_k = torch.randn(total_blocks, block_size, h_kv, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10

        # Generate indices (flat into blocked KV cache)
        total_tokens = total_blocks * block_size
        indices = torch.randint(0, total_tokens, (b, s_q, topk), dtype=torch.int32, device=DEVICE)

        # Quantize if FP8
        if is_fp8:
            if d_qk == 576:
                layout = quant.FP8KVCacheLayout.V32_FP8Sparse
            else:
                layout = quant.FP8KVCacheLayout.MODEL1_FP8Sparse
            blocked_k_q = quant.quantize_k_cache(blocked_k, layout)
            k_cache = blocked_k_q
        else:
            k_cache = blocked_k

        sched_meta, _ = get_mla_metadata()
        sm_scale = d_qk ** (-0.55)

        def run():
            return flash_mla_with_kvcache(
                q, k_cache, None, None, d_v,
                sched_meta, None, softmax_scale=sm_scale, causal=False,
                is_fp8_kvcache=is_fp8, indices=indices,
            )

        run()
        device_synchronize()

        flops = compute_flops(h_q, s_q, topk, d_qk, d_v, batch=b)
        if is_fp8:
            kv_token_bytes = 656 if d_qk == 576 else 576
        else:
            kv_token_bytes = d_qk * 2
        mem_bytes = compute_mem_bytes(h_q, h_kv, s_q, topk, d_qk, d_v, batch=b, kv_token_bytes=kv_token_bytes)

        api_time = bench_api_time(run, num_warmups, num_runs)
        kernel_time = None
        kernel_names = None
        if do_kernel_time:
            kernel_time, kernel_names = bench_kernel_time(run, num_warmups, num_runs)

        results.append(BenchResult(config_str, flops, mem_bytes, api_time, kernel_time, kernel_names))

    return results


# ============================================================================
# Sparse Prefill Benchmark
# ============================================================================

def bench_sparse_prefill(num_warmups: int, num_runs: int, do_kernel_time: bool) -> List[BenchResult]:
    configs = [
        # (s_q, topk, s_kv, h_q, h_kv, d_qk, d_v, is_fp8)
        (16,   256,  16384,  64, 1, 576, 512, False),
        (64,  1024,  32768,  64, 1, 576, 512, False),
        (256, 1024,  65536,  64, 1, 576, 512, False),
        (64,   256,  32768, 128, 1, 512, 512, False),
        (64,  1024,  32768,  64, 1, 576, 512, False),
        (256, 1024,  65536,  64, 1, 576, 512, False),
        (64,  2048, 131072, 128, 1, 512, 512, False),
    ]

    results = []
    for s_q, topk, s_kv, h_q, h_kv, d_qk, d_v, is_fp8 in configs:
        device_empty_cache()
        torch.manual_seed(42)

        config_str = f"sq={s_q},topk={topk},skv={s_kv},hq={h_q},d={d_qk}"

        q = torch.randn(s_q, h_q, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10
        kv = torch.randn(s_kv, h_kv, d_qk, dtype=torch.bfloat16, device=DEVICE) / 10

        # Generate random indices [s_q, h_kv, topk]
        indices = torch.randint(0, s_kv, (s_q, h_kv, topk), dtype=torch.int32, device=DEVICE)

        sm_scale = 0.5

        def run():
            return flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v)

        run()
        device_synchronize()

        flops = compute_flops(h_q, s_q, topk, d_qk, d_v, batch=1)
        mem_bytes = compute_mem_bytes(h_q, h_kv, s_q, topk, d_qk, d_v, batch=1)

        api_time = bench_api_time(run, num_warmups, num_runs)
        kernel_time = None
        kernel_names = None
        if do_kernel_time:
            kernel_time, kernel_names = bench_kernel_time(run, num_warmups, num_runs)

        results.append(BenchResult(config_str, flops, mem_bytes, api_time, kernel_time, kernel_names))

    return results


# ============================================================================
# Main
# ============================================================================

KERNEL_MAP = {
    "dense_decode": ("Dense Decode", bench_dense_decode),
    "dense_prefill": ("Dense Prefill", bench_dense_prefill),
    "sparse_decode": ("Sparse Decode", bench_sparse_decode),
    "sparse_prefill": ("Sparse Prefill", bench_sparse_prefill),
}


def main():
    parser = argparse.ArgumentParser(description="MLA Kernel Performance Benchmark")
    parser.add_argument("--kernel", type=str, default=None,
                        choices=list(KERNEL_MAP.keys()),
                        help="Run only a specific kernel benchmark")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations (default: 5)")
    parser.add_argument("--runs", type=int, default=30,
                        help="Number of timed iterations (default: 30)")
    parser.add_argument("--no-kernel-time", action="store_true",
                        help="Skip kernel-level profiling (faster)")
    parser.add_argument("--show-kernels", action="store_true",
                        help="Print kernel names from profiler trace")
    args = parser.parse_args()

    device = torch.device(f"{DEVICE}:0")
    torch.xpu.set_device(device)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)

    do_kernel_time = not args.no_kernel_time
    kernels_to_run = [args.kernel] if args.kernel else list(KERNEL_MAP.keys())

    print(f"MLA Benchmark: warmup={args.warmup}, runs={args.runs}, "
          f"kernel_time={'ON' if do_kernel_time else 'OFF'}")
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"Kernels: {', '.join(kernels_to_run)}")

    for kernel_name in kernels_to_run:
        title, bench_fn = KERNEL_MAP[kernel_name]
        try:
            results = bench_fn(args.warmup, args.runs, do_kernel_time)
            print_results(title, results)
            if args.show_kernels and results:
                for r in results:
                    if r.kernel_names:
                        print(f"  Kernels for {r.config_str}:")
                        for kn in r.kernel_names:
                            print(f"    - {kn}")
        except Exception as e:
            print(f"\n  ERROR in {title}: {e}")
            import traceback
            traceback.print_exc()

    print("Benchmark complete.")


if __name__ == "__main__":
    main()
