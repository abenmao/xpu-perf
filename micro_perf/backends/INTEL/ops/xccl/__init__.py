"""
XCCL / BCS queue-pressure mitigation for Intel XPU backend.

* perf.py       – run_perf() with BCS-aware iter/data caps
* infer_loop.py – run_infer_loop() with ws-descending batch sort
"""

from .perf import run_perf
from .infer_loop import run_infer_loop

__all__ = ["run_perf", "run_infer_loop"]
