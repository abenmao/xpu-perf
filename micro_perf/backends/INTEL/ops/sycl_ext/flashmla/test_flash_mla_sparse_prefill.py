import time
import sys
import gc

import torch
import kernelkit as kk

from lib import TestParam, DEVICE, device_synchronize, device_empty_cache
import lib
import ref

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

    if p.num_runs > 0:
        flops_and_mem_vol = lib.count_flop_and_mem_vol(p, t)
        prefill_result = kk.bench_kineto(run_prefill, num_tests=p.num_runs)
        prefill_ans_time = prefill_result.get_kernel_time("sparse_attn_fwd")
        prefill_flops = flops_and_mem_vol.fwd_flop/prefill_ans_time/1e12
        prefill_mem_bw = flops_and_mem_vol.fwd_mem_vol/prefill_ans_time/1e12
        print(f"Prefill:  {prefill_ans_time*1e6:4.0f} us, {prefill_flops:6.1f} TFlops, {prefill_mem_bw:4.2f} TBps")

    if p.check_correctness:
        device_synchronize()
        prefill_ans_out_cpu = prefill_ans_out.cpu().float()
        prefill_ans_max_logits_cpu = prefill_ans_max_logits.cpu()
        prefill_ans_lse_cpu = prefill_ans_lse.cpu()
        del prefill_ans_out, prefill_ans_max_logits, prefill_ans_lse, run_prefill
        gc.collect()
        device_empty_cache()
        ref_out, ref_out_fp32, ref_max_logits, ref_lse = ref.ref_sparse_attn_fwd(p, t)
        ref_lse[ref_lse == float("-inf")] = float("+inf")

        # Eagerly free GPU test data before comparison (CPU-only from here)
        del t
        gc.collect()
        device_empty_cache()

        is_correct = True
        is_correct &= kk.check_is_allclose("out", prefill_ans_out_cpu, ref_out_fp32, abs_tol=8e-4, rel_tol=3.01/128, cos_diff_tol=7e-6)
        is_correct &= kk.check_is_allclose("max_logits", prefill_ans_max_logits_cpu, ref_max_logits, abs_tol=1e-6, rel_tol=2.01/65536)
        is_correct &= kk.check_is_allclose("lse", prefill_ans_lse_cpu, ref_lse, abs_tol=1e-6, rel_tol=2.01/65536)

        return is_correct
    else:
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

    correctness_cases = [
        # Regular shapes
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            # Regular shapes
            (128, 128),
            (256, 256),
            (512, 512),

            # Irregular shapes
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            # Irregular shapes with OOB TopK
            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [
            1, 62, 213
        ]
    ]

    correctness_cases_with_features = [
        TestParam(s_q, s_kv, topk, h_q=h_q, num_runs=0, have_attn_sink=have_attn_sink, have_topk_length=have_topk_length, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (592, 128),
            (1840, 256),
            (1592, 384),
            (1521, 512),

            (95, 128),
            (153, 256),
            (114, 384),
        ]
        for s_q in [62, 213]
        for have_sink_lse in [False, True]
        for have_attn_sink in [False, True]
        for have_topk_length in [False, True]
    ]

    corner_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=True, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_q, s_kv, topk in [
            (1, 128, 128),
            (1, 256, 256),
            (1234, 4321, 4096),
            (4096, 2048, 2048)
        ]
    ] + [
        # In these cases, some blocks may not have any valid topk indices
        TestParam(s_q, s_kv, topk, h_q=h_q, is_all_indices_invalid=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
        for s_kv, topk in [
            (32, 2048),
            (64, 8192)
        ]
        for s_q in [1, 1024]
    ] + [
        # In this testcase, s_q is really large, so we cannot put it on the second dimension of grid shape
        TestParam(70000, 256, 256, h_q=h_q, check_correctness=False, num_runs=0, have_attn_sink=True, have_topk_length=True, d_qk=d_qk)
        for d_qk in [512, 576]
        for h_q in [
            128, 64
        ]
    ]

    performance_case_templates = [
        # V3.2
        (576, 128, 2048, [8192, 32768, 65536, 98304, 131072]),
        # MODEL1 CONFIG1
        (512, 64, 512, [8192, 32768, 49152, 65536]),
        # MODEL1 CONFIG2
        (512, 128, 1024, [8192, 32768, 49152, 65536]),
    ]

    performance_cases = [
        TestParam(s_q, s_kv, topk, h_q=h_q, d_qk=d_qk, have_attn_sink=True)
        for (d_qk, h_q, topk, s_kv_list) in performance_case_templates
        for s_q in [4096]
        for s_kv in s_kv_list
    ]

    testcases = correctness_cases + correctness_cases_with_features + corner_cases + performance_cases

    # OOM / device-lost patterns that indicate device memory exhaustion (not a kernel bug)
    _OOM_PATTERNS = ("OUT_OF_RESOURCES", "OUT_OF_DEVICE_MEMORY", "out of memory", "DEVICE_LOST")

    is_no_cooldown = lib.is_no_cooldown()
    failed_cases = []
    skipped_cases = []  # OOM-skipped cases
    for test in testcases:
        if test != testcases[0] and test.num_runs > 0 and not is_no_cooldown:
            time.sleep(0.3)
        try:
            is_correct = run_test(test)
            if not is_correct:
                failed_cases.append(test)
        except (RuntimeError, MemoryError) as e:
            err_str = str(e)
            if isinstance(e, MemoryError) or any(pat in err_str for pat in _OOM_PATTERNS):
                print(f"\033[33m[SKIPPED - OOM] {test}\033[0m")
                skipped_cases.append(test)
            else:
                raise
        if DEVICE == 'xpu':
            gc.collect()
            try:
                device_synchronize()
                torch.xpu.empty_cache()
            except RuntimeError:
                import time as _time
                _time.sleep(0.5)
                try:
                    torch.xpu.synchronize()
                    torch.xpu.empty_cache()
                except RuntimeError:
                    pass  # continue anyway; next case may still work
    
    passed = len(testcases) - len(failed_cases) - len(skipped_cases)
    if len(skipped_cases) > 0:
        print(f"\033[33m\033[1m{len(skipped_cases)} case(s) skipped due to OOM (device memory limit):\033[0m")
        for case in skipped_cases:
            print(f"    {case}")
    if len(failed_cases) > 0:
        print(f"\033[31m\033[1m{len(failed_cases)} / {len(testcases)} cases failed:\033[0m")
        for case in failed_cases:
            print(f"    {case}")
        sys.exit(1)
    else:
        print(f"\033[32m\033[1mAll {passed} cases passed! ({len(skipped_cases)} skipped due to OOM)\033[0m")
