"""
BCS-aware performance measurement for Intel XPU backend.

On Intel XE GPUs the GuC microcontroller manages hardware exec_queues.
When BCS (Blitter/DMA) has a deep command backlog, exec_queue_destroy_async
blocks >10 ms, triggering watchdog warnings and eventually Engine Reset.

run_perf() overrides the base Backend.perf() with:
  1. empty_cache() *before* allocation  (correct free-memory reading)
  2. Reduced min_test_iters for large BCS-heavy tensors  (3 vs 10)
  3. Capped max_data_cnt=1 for CPU-tensor ops  (avoid pinned-memory blowup)
  4. Cooldown sync + sleep after measurement  (let BCS drain)
"""

import math
import random
import time
import traceback

# ---------------------------------------------------------------------------
# Per-op policy helpers (thresholds for BCS pressure mitigation)
# ---------------------------------------------------------------------------

# Tensors >= 512 MB are considered "large" for BCS purposes.
_BCS_THRESHOLD = 512 * 1024 * 1024


def _is_xccl_op(op_instance):
    return getattr(op_instance.__class__, "__module__", "").endswith("core.ops.xccl_ops")


def _has_cpu_tensor(op_instance):
    for tensors in (getattr(op_instance, "input_tensor_info", {}),
                    getattr(op_instance, "output_tensor_info", {})):
        for info in tensors.values():
            if getattr(info, "device", None) == "cpu":
                return True
    return False


def _should_throttle(op_instance):
    size = getattr(op_instance, "tensor_size", 0)
    return size >= _BCS_THRESHOLD and (_is_xccl_op(op_instance) or _has_cpu_tensor(op_instance))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_perf(backend, op_instance):
    """Drop-in replacement for Backend.perf() with BCS mitigations."""
    tensor_size = op_instance.tensor_size
    backend.empty_cache()                       # fix: before allocation

    avail = backend.get_mem_info()[0]
    assume_avail = int(avail * 0.9)
    assume_cache = 1 * (1024 ** 3)

    latency_us = 0.
    kernel_mapping = {}

    try:
        # --- iter / data-cnt caps ---
        min_iters = 3 if _should_throttle(op_instance) else 10
        max_data_cnt = 1
        if not op_instance.is_concurrent:
            if tensor_size > assume_avail:
                raise RuntimeError("Not enough memory to run the op")
            elif 2 * tensor_size > assume_avail:
                max_data_cnt = 1
            elif tensor_size > assume_cache:
                max_data_cnt = 2
            else:
                max_data_cnt = min(
                    math.floor(max(assume_avail, assume_cache) / tensor_size),
                    math.floor(assume_cache / tensor_size),
                )
        if _has_cpu_tensor(op_instance):         # fix: cap for pinned memory
            max_data_cnt = 1

        tensor_list = op_instance.create_tensors(max_data_cnt)
        random.shuffle(tensor_list)

        # Probe latency
        latency_us, _ = backend.core_perf(op_instance, 2, 2, tensor_list, profiling=False)
        prefer_iters = min(max(int(1e6 / latency_us), 2), min_iters)
        if op_instance.group_size > 1:
            dist = backend.get_dist_module()
            buf = [None] * op_instance.group_size
            dist.all_gather_object(buf, prefer_iters, group=op_instance.op_group)
            prefer_iters = max(buf)

        if _should_throttle(op_instance):
            backend.device_synchronize()
        time.sleep(0.2)

        # Measure
        profiling = backend.enable_profiling and op_instance.require_profiling
        latency_us, kernel_mapping = backend.core_perf(
            op_instance, 2, prefer_iters, tensor_list, profiling=profiling)

        del tensor_list
        backend.empty_cache()

        # Cooldown for heavy BCS ops
        if _should_throttle(op_instance):
            backend.device_synchronize()
            time.sleep(0.5)
    except Exception:
        traceback.print_exc()

    return op_instance.summary(latency_us, kernel_mapping)
