import sys
import os
import re
import pathlib
import subprocess

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.tensor_gemm_ops import GemmOp

try:
    # Resolve sycl-tla path relative to xpu-perf root (xpu-perf/../sycl-tla)
    # gemm.py -> sycl_tla -> ops -> INTEL -> backends -> micro_perf -> xpu-perf
    _XPU_PERF_ROOT = pathlib.Path(__file__).resolve().parents[4]
    SYCL_TLA_DIR = str(_XPU_PERF_ROOT.parent / "sycl-tla")
    SYCL_TLA_GEMM_EXAMPLE_DIR = os.path.join(
        SYCL_TLA_DIR, "build/examples/00_bmg_gemm"
    )
    SYCL_TLA_GEMM_EXAMPLE_BINARY = os.path.join(
        SYCL_TLA_GEMM_EXAMPLE_DIR, "00_bmg_gemm"
    )

    if not os.path.isfile(SYCL_TLA_GEMM_EXAMPLE_BINARY):
        print(
            f"[WARNING] sycl-tla GEMM example binary not found at {SYCL_TLA_GEMM_EXAMPLE_BINARY}. "
            f"sycl_tla_gemm provider will NOT be available. "
            f"Expected sycl-tla repo at {SYCL_TLA_DIR} (sibling of {_XPU_PERF_ROOT})."
        )
        raise FileNotFoundError(SYCL_TLA_GEMM_EXAMPLE_BINARY)


    @ProviderRegistry.register_vendor_impl("gemm", "sycl_tla_gemm")
    class SyclTlaGemmOp(GemmOp):
        # 00_bmg_gemm example uses BF16xBF16->FP32 path.
        SUPPORTED_DTYPES = ["bfloat16"]

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.dtype not in self.SUPPORTED_DTYPES:
                raise ValueError(
                    f"SyclTlaGemmOp only supports dtype in {self.SUPPORTED_DTYPES}, "
                    f"got {self.dtype}"
                )

            # Run sycl-tla example binary and store parsed results
            self._sycl_tla_result = self._run_sycl_tla()

            # Override run to no-op; benchmarking was done by the binary
            self._run_func = lambda tensor_mapping: None
            self._create_tensors_func = lambda instance_num: [{}] * max(instance_num, 1)

        def _run_sycl_tla(self):
            """Run 00_bmg_gemm example binary and parse text output metrics."""
            cmd = (
                'SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file" '
                f"{SYCL_TLA_GEMM_EXAMPLE_BINARY} "
                f"--m={self.M} --n={self.N} --k={self.K} --l=1 "
                f"--alpha=1 --beta=0 --iterations=100 --verify=0"
            )

            print(f"[SyclTlaGemmOp] Running: {cmd}")

            result = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
                start_new_session=True,
                close_fds=True,
            )

            output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")

            if result.returncode != 0:
                raise RuntimeError(
                    f"sycl-tla GEMM example failed (rc={result.returncode}):\n{output}"
                )

            print(f"[SyclTlaGemmOp] Output:\n{output}")

            perf_match = re.search(
                r"Cutlass GEMM Performance:\s*\[([\d\.]+)\]TFlop/s\s*\(([\d\.]+)\)ms",
                output,
            )
            if not perf_match:
                raise RuntimeError(
                    f"Failed to parse 00_bmg_gemm output:\n{output}"
                )

            return {
                "tflops": float(perf_match.group(1)),
                "latency_ms": float(perf_match.group(2)),
            }

        def summary(self, latency_us, kernel_mapping={}):
            if self._sycl_tla_result:
                latency_ms = self._sycl_tla_result.get("latency_ms", 0.0)
                if latency_ms > 0:
                    latency_us = latency_ms * 1000.0
            return super().summary(latency_us, kernel_mapping)


except Exception as e:
    print(f"[SyclTlaGemmOp] Failed to register: {e}")
    pass
