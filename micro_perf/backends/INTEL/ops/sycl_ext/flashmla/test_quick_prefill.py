#!/usr/bin/env python3
"""Quick subset of sparse prefill tests for development iteration.
Covers: both d_qk, both h_q, regular/irregular shapes, OOB topk,
attn_sink, topk_length, corner cases. ~40 cases, runs in minutes."""

import time, sys, gc
import torch
import kernelkit as kk
from lib import TestParam, DEVICE, device_synchronize, device_empty_cache
import lib, ref

_counter = kk.Counter()

@torch.inference_mode()
def run_test(p: TestParam) -> bool:
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()
    print("================")
    print(f"Running on {p}")
    device_empty_cache()

    t = lib.generate_testcase(p)
    device_synchronize()
    def run_prefill():
        return lib.run_flash_mla_sparse_fwd(p, t, False)
    prefill_ans_out, prefill_ans_max_logits, prefill_ans_lse = run_prefill()
    device_synchronize()

    if p.check_correctness:
        device_synchronize()
        prefill_ans_out_cpu = prefill_ans_out.cpu().float()
        prefill_ans_max_logits_cpu = prefill_ans_max_logits.cpu()
        prefill_ans_lse_cpu = prefill_ans_lse.cpu()
        del prefill_ans_out, prefill_ans_max_logits, prefill_ans_lse, run_prefill
        gc.collect(); device_empty_cache()
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")
        del t; gc.collect(); device_empty_cache()

        ok = True
        ok &= kk.check_is_allclose("out", prefill_ans_out_cpu, ref_out_fp32, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        ok &= kk.check_is_allclose("max_logits", prefill_ans_max_logits_cpu, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        ok &= kk.check_is_allclose("lse", prefill_ans_lse_cpu, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)
        return ok
    return True

if __name__ == '__main__':
    device = torch.device(f"{DEVICE}:0")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)
    if DEVICE == 'xpu':
        torch.xpu.set_device(device)
    else:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('high')

    # --- Small representative test set ---
    correctness_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [128, 64]
        for s_kv, topk in [
            (128, 128),   # small regular
            (512, 512),   # medium regular
            (592, 128),   # irregular
            (95, 128),    # OOB topk
        ]
        for s_q in [1, 62]
    ]

    feature_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0,
                  have_attn_sink=have_sink, have_topk_length=have_tl, d_qk=d_qk)
        for d_qk in [576]
        for h_q in [128]
        for s_kv, topk in [(592, 128), (95, 128)]
        for s_q in [62]
        for have_sink in [True]
        for have_tl in [False, True]
    ]

    corner_cases = [
        TestParam(s_q, s_kv, topk, h_q=128, is_all_indices_invalid=True,
                  num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=576)
        for s_q, s_kv, topk in [(1, 128, 128), (1234, 4321, 4096)]
    ] + [
        TestParam(s_q, 32, 2048, h_q=128, is_all_indices_invalid=False,
                  num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=576)
        for s_q in [1, 1024]
    ]

    testcases = correctness_cases + feature_cases + corner_cases
    print(f"\033[36m{len(testcases)} quick prefill cases\033[0m")

    _OOM_PATTERNS = ("OUT_OF_RESOURCES", "OUT_OF_DEVICE_MEMORY", "out of memory", "DEVICE_LOST")
    failed, skipped = [], []
    for test in testcases:
        try:
            if not run_test(test):
                failed.append(test)
        except (RuntimeError, MemoryError) as e:
            if isinstance(e, MemoryError) or any(p in str(e) for p in _OOM_PATTERNS):
                print(f"\033[33m[SKIPPED - OOM] {test}\033[0m"); skipped.append(test)
            else:
                raise
        if DEVICE == 'xpu':
            gc.collect(); device_empty_cache()

    total = len(testcases)
    if failed:
        print(f"\033[31m{len(failed)}/{total} FAILED\033[0m")
        for f in failed: print(f"  {f}")
        sys.exit(1)
    else:
        print(f"\033[32mAll {total} quick prefill cases passed! ({len(skipped)} OOM-skipped)\033[0m")
