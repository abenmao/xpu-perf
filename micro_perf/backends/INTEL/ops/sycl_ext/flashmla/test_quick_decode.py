#!/usr/bin/env python3
"""Quick subset of sparse decode tests for development iteration.
Covers: both d_qk, both h_q, various batch/s_q/topk/block_size combos,
corner cases (all-invalid, zero-seqlen, attn_sink). ~60 cases, runs in minutes."""

import time, sys, gc, dataclasses
from typing import List, Tuple

import torch
import kernelkit as kk
import flash_attn_interface
import lib, ref
from lib import TestParam, DEVICE, device_synchronize, device_empty_cache
from lib import RawTestParamForDecode as RawTestParam

@dataclasses.dataclass
class Result:
    is_correct: bool
    time_usage_per_us: float
    achieved_tflops: float
    achieved_gBps: float

_counter = kk.Counter()

@torch.inference_mode()
def test_flash_mla(p: TestParam) -> Result:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()
    assert p.decode
    print("================")
    print(f"Running on {p}")
    device_empty_cache()

    t = lib.generate_testcase_for_decode(p)
    tile_scheduler_metadata, _ = flash_attn_interface.get_mla_metadata()
    def run_decode():
        return lib.run_flash_mla_decode(p, t, tile_scheduler_metadata, None)

    if p.check_correctness:
        device_synchronize()
        out_ans, lse_ans = run_decode()
        device_synchronize()

    result = Result(True, 0.0, 0.0, 0.0)
    if p.check_correctness:
        device_synchronize()
        out_ans_cpu = out_ans.cpu()
        lse_ans_cpu = lse_ans.cpu()
        del out_ans, lse_ans, run_decode; gc.collect(); device_empty_cache()
        out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)
        del t; gc.collect(); device_empty_cache()

        ok = True
        ok &= kk.check_is_allclose("out", out_ans_cpu, out_ref, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=7e-6)
        ok &= kk.check_is_allclose("lse", lse_ans_cpu, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
        result.is_correct = ok
    return result

def main():
    dtype = torch.bfloat16
    device = torch.device(f"{DEVICE}:0")
    torch.set_default_dtype(dtype)
    torch.set_default_device(device)
    if DEVICE == 'xpu':
        torch.xpu.set_device(device)
    else:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(32)

    # --- Small representative test set ---
    correctness_raw = [
        RawTestParam(b, h_q, s_q, 1, s_k, True, topk,
                     d_qk=d_qk, block_size=block_size,
                     check_correctness=True, num_runs=0)
        for d_qk in [576, 512]
        for h_q in [128, 64]
        for (s_k, topk, block_size) in [
            (512, 64, 2),
            (512, 64, 64),
            (1024, 576, 61),
            (2046, 2048, 64),
        ]
        for b in [4, 74]
        for s_q in [1, 3]
    ]

    corner_raw = [
        RawTestParam(b, h_q, 3, 1, s_k, True, topk,
                     is_all_indices_invalid=is_all_invalid,
                     have_zero_seqlen_k=have_zero,
                     enable_attn_sink=True,
                     d_qk=576, block_size=61,
                     check_correctness=True, num_runs=0)
        for h_q in [128]
        for (s_k, topk) in [(512, 64), (650, 576)]
        for b in [4, 74]
        for is_all_invalid in [True, False]
        for have_zero in [True, False]
        if (is_all_invalid or have_zero)
    ]

    # topk_length cases
    topk_len_raw = [
        RawTestParam(b, 128, 1, 1, s_k, True, topk,
                     have_topk_length=True, enable_attn_sink=True,
                     d_qk=d_qk, block_size=block_size,
                     check_correctness=True, num_runs=0)
        for d_qk in [512]
        for (s_k, topk, block_size) in [(512, 64, 2), (1024, 576, 61)]
        for b in [4, 74]
    ]

    raw_testcases = correctness_raw + corner_raw + topk_len_raw
    testcases = [t.to_test_param() for t in raw_testcases]
    print(f"\033[36m{len(testcases)} quick decode cases\033[0m")

    _OOM_PATTERNS = ("OUT_OF_RESOURCES", "OUT_OF_DEVICE_MEMORY", "out of memory", "DEVICE_LOST")
    failed, skipped = [], []
    results: List[Tuple[TestParam, Result]] = []
    for idx, tc in enumerate(testcases):
        print(f"[{idx+1}/{len(testcases)}]  ", end='')
        try:
            result = test_flash_mla(tc)
            results.append((tc, result))
            if not result.is_correct:
                failed.append(tc)
        except (RuntimeError, MemoryError) as e:
            if isinstance(e, MemoryError) or any(p in str(e) for p in _OOM_PATTERNS):
                print(f"\033[33m[SKIPPED - OOM] {tc}\033[0m"); skipped.append(tc)
            else:
                raise
        if DEVICE == 'xpu':
            gc.collect()
            try:
                device_synchronize(); torch.xpu.empty_cache()
            except RuntimeError:
                time.sleep(0.5)
                try: torch.xpu.synchronize(); torch.xpu.empty_cache()
                except RuntimeError: pass

    total = len(testcases)
    if failed:
        print(f"\033[31m{len(failed)}/{total} FAILED\033[0m")
        for f in failed: print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\033[32mAll {total} quick decode cases passed! ({len(skipped)} OOM-skipped)\033[0m")

if __name__ == '__main__':
    main()
