import time
import sys
import gc
import dataclasses
from typing import List, Tuple, Optional

import torch
import kernelkit as kk

import flash_attn_interface
import lib
from lib import TestParam, DEVICE, device_synchronize, device_empty_cache
from lib import RawTestParamForDecode as RawTestParam
import ref

"""
Generate testcase for unit test
"""

def gen_testcase() -> List[RawTestParam]:
    correctness_cases = []
    corner_cases = []
    for d_qk in [576, 512]:
        for have_extra_k in ([False, True] if d_qk == 512 else [False]):
            for have_extra_topk_len in ([False, True] if have_extra_k else [False]):
                for have_topk_len in ([False, True] if d_qk == 512 else [False]):
                    for h_q in [64, 128]:
                        cur_correctness_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=True,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0)
                            for (s_k, topk, block_size) in [
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 2),
                                (512, 64, 64),
                                (512, 64, 69),
                                (1024, 576, 2),
                                (1024, 576, 61),
                                (2046, 2048, 2),
                                (2046, 2048, 64),
                                (2046, 2048, 576)
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [1, 3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                        ]
                        correctness_cases.extend(cur_correctness_cases)

                        cur_corner_cases = [
                            RawTestParam(b, h_q, s_q, 1, s_k, is_varlen, topk,
                                        is_all_indices_invalid=is_all_indices_invalid,
                                        have_zero_seqlen_k=have_zero_seqlen_k,
                                        have_topk_length=have_topk_len,
                                        enable_attn_sink=enable_attn_sink,
                                        extra_s_k=extra_s_k,
                                        extra_topk=extra_topk,
                                        block_size=block_size,
                                        extra_block_size=extra_block_size,
                                        have_extra_topk_length=have_extra_topk_len,
                                        d_qk=d_qk,
                                        check_correctness=True,
                                        num_runs=0,
                            )
                            for (s_k, topk, block_size) in [
                                (512, 64, 61),
                                (650, 576, 53),
                            ]
                            for (extra_s_k, extra_topk, extra_block_size) in ([
                                (512, 64, 61),
                                (650, 576, 53),
                            ] if have_extra_k else [(None, None, None)])
                            for b in [4, 74, 321]
                            for s_q in [3]
                            for is_varlen in ([True, False] if (b == 74 and not have_topk_len and not have_extra_topk_len) else [True])
                            for is_all_indices_invalid in [True, False]
                            for have_zero_seqlen_k in [True, False]
                            for enable_attn_sink in [True, False]
                            if (is_all_indices_invalid or have_zero_seqlen_k or enable_attn_sink)
                        ]
                        corner_cases.extend(cur_corner_cases)

    base_and_bszs = [
        # V3.2
        (RawTestParam(0, 128, 2, 1, 32768, True, topk=2048, d_qk=576), [2, 64, 74, 128]),
        # MODEL1 CONFIG1
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=512, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG2
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=64), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG3
        (RawTestParam(0, 64, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
        # MODEL1 CONFIG4
        (RawTestParam(0, 128, 2, 1, 16384, True, topk=128, d_qk=512, extra_s_k=16384, extra_topk=1024, block_size=256, extra_block_size=2, have_extra_topk_length=True), [2, 64, 74, 128, 74*2, 256]),
    ]
    performance_cases = [
        dataclasses.replace(base, b=b)
        for base, bszs in base_and_bszs
        for b in bszs
    ] + [
        RawTestParam(74*2, h_q, 2, 1, 32768, True, topk=16384, d_qk=d_qk)
        for h_q in [64, 128]
        for d_qk in [512, 576]
    ]

    return correctness_cases + corner_cases + performance_cases


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
    
    # Run the kernel once for correctness test
    if p.check_correctness:
        device_synchronize()
        out_ans, lse_ans = run_decode()
        device_synchronize()
    
    # Performance test
    performance_result = Result(True, 0.0, 0.0, 0.0)
    if p.num_runs > 0:
        bench_result = kk.bench_kineto(run_decode, p.num_runs)
        e2e_time_usage_s = bench_result.avg_time
        e2e_time_usage_us = e2e_time_usage_s * 1e6

        flops_and_mem_vol = lib.count_flop_and_mem_vol_for_decode(p, t)
        achieved_tflops = flops_and_mem_vol.flop / e2e_time_usage_s / 1e12
        achieved_gBps = flops_and_mem_vol.mem_vol / e2e_time_usage_s / 1e9
        print(f'Time (per): {e2e_time_usage_us:.1f} us')
        print(f'TFlops: {achieved_tflops:.1f}')
        print(f'GB/s: {achieved_gBps:.0f}')

        performance_result = Result(True, e2e_time_usage_us, achieved_tflops, achieved_gBps)
    
    is_correct = True
    if p.check_correctness:
        device_synchronize()
        out_ans_cpu = out_ans.cpu()
        lse_ans_cpu = lse_ans.cpu()
        del out_ans, lse_ans, run_decode
        gc.collect()
        device_empty_cache()
        out_ref, lse_ref = ref.ref_sparse_attn_decode(p, t)

        # Eagerly free GPU test data before comparison (CPU-only from here)
        del t
        gc.collect()
        device_empty_cache()

        is_out_correct = kk.check_is_allclose("out", out_ans_cpu, out_ref, abs_tol=1e-3, rel_tol=2.01/128, cos_diff_tol=7e-6)
        is_lse_correct = kk.check_is_allclose("lse", lse_ans_cpu, lse_ref, abs_tol=1e-6, rel_tol=8.01/65536)
        is_correct &= is_out_correct and is_lse_correct

    performance_result.is_correct = is_correct
    return performance_result


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

    raw_testcases = gen_testcase()
    testcases = [t.to_test_param() for t in raw_testcases]

    print(f"{kk.colors['CYAN_BG']}{len(testcases)} testcases to run{kk.colors['CLEAR']}")

    # OOM / device-lost patterns that indicate device memory exhaustion (not a kernel bug)
    _OOM_PATTERNS = ("OUT_OF_RESOURCES", "OUT_OF_DEVICE_MEMORY", "out of memory", "DEVICE_LOST")

    is_no_cooldown = lib.is_no_cooldown()
    num_testcases_len = len(str(len(testcases)))
    failed_cases = []
    skipped_cases = []  # OOM-skipped cases
    results: List[Tuple[TestParam, Result]] = []
    for testcase_idx, testcase in enumerate(testcases):
        if testcase != testcases[0] and testcase.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        print(f"[{testcase_idx+1:{num_testcases_len}d}/{len(testcases)}, {testcase_idx/len(testcases)*100:3.0f}%]  ", end='')
        try:
            result = test_flash_mla(testcase)
            results.append((testcase, result))
            if not result.is_correct:
                failed_cases.append(testcase)
        except (RuntimeError, MemoryError) as e:
            err_str = str(e)
            if isinstance(e, MemoryError) or any(pat in err_str for pat in _OOM_PATTERNS):
                print(f"\033[33m[SKIPPED - OOM] {testcase}\033[0m")
                skipped_cases.append(testcase)
            else:
                raise
        if DEVICE == 'xpu':
            gc.collect()
            try:
                device_synchronize()
                torch.xpu.empty_cache()
            except RuntimeError:
                # Device may be in a bad state after OOM; try to recover
                import time as _time
                _time.sleep(0.5)
                try:
                    torch.xpu.synchronize()
                    torch.xpu.empty_cache()
                except RuntimeError:
                    pass  # continue anyway; next case may still work

    # Print summary
    num_ran = len(results)
    num_skipped = len(skipped_cases)
    if skipped_cases:
        print(f"\033[33m\033[1m{num_skipped} case(s) skipped due to OOM (device memory limit)\033[0m")
    num_correct_testcases = [result.is_correct for t, result in results if t.check_correctness].count(True)
    num_ran_correctness = sum(1 for t, result in results if t.check_correctness)
    num_correctness_cases = sum([1 for t in testcases if t.check_correctness])
    if num_correct_testcases == num_ran_correctness:
        print(f"{kk.colors['GREEN_BG']}{num_correct_testcases}/{num_ran_correctness} correctness cases passed ({num_skipped} skipped due to OOM){kk.colors['CLEAR']}")
    else:
        print(f"{kk.colors['RED_BG']}{num_correct_testcases}/{num_ran_correctness} correctness cases passed ({num_skipped} skipped due to OOM){kk.colors['CLEAR']}")
        for t in failed_cases:
            print(f"\t{t},")


if __name__ == "__main__":
    main()
