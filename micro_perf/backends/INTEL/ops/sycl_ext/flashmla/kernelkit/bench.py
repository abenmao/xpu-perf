from typing import Tuple, List, Callable, Union, Dict, overload
import dataclasses
import time

import torch

from .utils import is_using_profiling_tools


@dataclasses.dataclass
class BenchXPUResult:
    """
    A struct holding XPU benchmark timing result.
    Provides the same API as the CUDA BenchKinetoRawResult for compatibility.
    """
    num_tests: int
    avg_time: float  # average time per call in seconds

    def get_kernel_time(self, kernel_name_substr: str = "") -> float:
        return self.avg_time


def bench_kineto(fn: Callable, num_tests: int = 30,
                 flush_l2: bool = True) -> BenchXPUResult:
    """
    Benchmark `fn` on XPU using torch.xpu.Event timing.
    Returns a BenchXPUResult with average per-call time in seconds.
    """
    num_warmups = 5

    # Warmup
    for _ in range(num_warmups):
        fn()
    torch.xpu.synchronize()

    start_events = [torch.xpu.Event(enable_timing=True) for _ in range(num_tests)]
    end_events = [torch.xpu.Event(enable_timing=True) for _ in range(num_tests)]

    for i in range(num_tests):
        if flush_l2:
            torch.empty(int(256e6 // 4), dtype=torch.int32, device='xpu').zero_()
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.xpu.synchronize()
    # elapsed_time returns milliseconds
    total_ms = sum(s.elapsed_time(e) for s, e in zip(start_events, end_events))
    avg_time = total_ms / num_tests / 1000.0  # convert to seconds

    return BenchXPUResult(num_tests=num_tests, avg_time=avg_time)


@overload
def bench_by_xpu_events(kernels: List[Callable], num_warmups_each: int, num_runs_each: int) -> List[float]: ...

@overload
def bench_by_xpu_events(kernels: Callable, num_warmups_each: int, num_runs_each: int) -> float: ...

def bench_by_xpu_events(kernels: Union[List[Callable], Callable], num_warmups_each: int, num_runs_each: int) -> Union[List[float], float]:
    buf_for_l2_clear = torch.empty(int(256e6 // 4), dtype=torch.int32, device='xpu')

    is_kernel_single_callable = isinstance(kernels, Callable)
    if is_kernel_single_callable:
        kernels = [kernels]

    torch.xpu.synchronize()
    for i in range(num_warmups_each):
        for kernel in kernels:
            kernel()
            if i == 0:
                try:
                    torch.xpu.synchronize()
                except Exception as e:
                    print(f"Kernel failed on warmup run {i}: {e}")
                    return []

    start_events = [[torch.xpu.Event(enable_timing=True) for _ in range(num_runs_each)] for _ in kernels]
    end_events = [[torch.xpu.Event(enable_timing=True) for _ in range(num_runs_each)] for _ in kernels]
    for i in range(num_runs_each):
        for j, kernel in enumerate(kernels):
            buf_for_l2_clear.random_()
            start_events[j][i].record()
            kernel()
            end_events[j][i].record()

    torch.xpu.synchronize()
    time_usages = [
        sum([start_events[j][i].elapsed_time(end_events[j][i]) * 1e-3 for i in range(num_runs_each)]) / num_runs_each
        for j in range(len(kernels))
    ]
    if is_kernel_single_callable:
        time_usages = time_usages[0]
    return time_usages

# Alias for backward compatibility
bench_by_cuda_events = bench_by_xpu_events
