import os
import pathlib
import re
import sys
import tempfile
import importlib.util
import ctypes

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeGatingGemmOp


def _flush_all_stdio():
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass


try:
    _XPU_PERF_ROOT = pathlib.Path(__file__).resolve().parents[5]
    SYCL_EXT_SO = os.path.join(
        str(_XPU_PERF_ROOT),
        "micro_perf/backends/INTEL/ops/sycl_ext/bmg_moe_gating_gemm_sycl.so",
    )

    if not os.path.isfile(SYCL_EXT_SO):
        print(
            f"[WARNING] sycl_ext shared object not found: {SYCL_EXT_SO}. "
            "sycl_tla_moe_gating_gemm provider will NOT be available."
        )
        raise FileNotFoundError(SYCL_EXT_SO)

    _spec = importlib.util.spec_from_file_location(
        "bmg_moe_gating_gemm_sycl", SYCL_EXT_SO
    )
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)


    @ProviderRegistry.register_vendor_impl("moe_gating_gemm", "sycl_tla_moe_gating_gemm")
    class SyclTlaMoeGatingGemmOp(MoeGatingGemmOp):
        SUPPORTED_DTYPES = {"float32", "bfloat16"}

        def vendor_parser(self):
            if self.dtype not in self.SUPPORTED_DTYPES:
                raise ValueError(
                    "sycl_tla_moe_gating_gemm requires dtype in "
                    f"{sorted(self.SUPPORTED_DTYPES)}, got {self.dtype}"
                )
            self.dst_dtype = self.args_dict.get("dst_dtype", "float32")
            if self.dst_dtype != "float32":
                raise ValueError(
                    "sycl_tla_moe_gating_gemm requires dst_dtype=float32, "
                    f"got {self.dst_dtype}"
                )

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self._provider = "sycl_tla_moe_gating_gemm"
            self._sycl_tla_result = self._run_sycl_ext()

            self._run_func = lambda tensor_mapping: None
            self._create_tensors_func = (
                lambda instance_num: [{} for _ in range(max(instance_num, 1))]
            )

        def _capture_extension_output(self, func, *args):
            with tempfile.TemporaryFile(mode="w+b") as temp_file:
                stdout_fd = os.dup(sys.stdout.fileno())
                stderr_fd = os.dup(sys.stderr.fileno())
                try:
                    _flush_all_stdio()
                    os.dup2(temp_file.fileno(), sys.stdout.fileno())
                    os.dup2(temp_file.fileno(), sys.stderr.fileno())
                    result = func(*args)
                    _flush_all_stdio()
                finally:
                    _flush_all_stdio()
                    os.dup2(stdout_fd, sys.stdout.fileno())
                    os.dup2(stderr_fd, sys.stderr.fileno())
                    os.close(stdout_fd)
                    os.close(stderr_fd)

                temp_file.seek(0)
                output_text = temp_file.read().decode("utf-8", errors="replace")
                return result, output_text

        def _parse_sycl_tla_output(self, output_text):
            perf_match = re.search(
                r"Cutlass GEMM Performance:\s*\[([\d\.eE+-]+)\]TFlop/s\s*\(([\d\.eE+-]+)\)ms",
                output_text,
            )
            if perf_match is None:
                raise RuntimeError(
                    "Failed to parse 00_bmg_moe_gating_gemm output:\n"
                    f"{output_text}"
                )

            return {
                "tflops": float(perf_match.group(1)),
                "latency_ms": float(perf_match.group(2)),
                "raw_output": output_text,
            }

        def _run_sycl_ext(self):
            command = "bmg_moe_gating_gemm_sycl.run_moe_gating_gemm(...)"
            print(f"[SyclTlaMoeGatingGemmOp] Running: {command}")

            retcode, output_text = self._capture_extension_output(
                _sycl_ext.run_moe_gating_gemm,
                self.num_tokens,
                self.num_experts,
                self.hidden_size,
                1,
                1.0,
                0.0,
                200,
                100,
                0,
                self.dtype,
            )

            if retcode != 0:
                raise RuntimeError(
                    f"sycl_ext call failed (rc={retcode}):\n{output_text}"
                )

            print(f"[SyclTlaMoeGatingGemmOp] Output:\n{output_text}")

            parsed_result = self._parse_sycl_tla_output(output_text)
            return {
                "latency_ms": parsed_result["latency_ms"],
                "tflops": parsed_result["tflops"],
                "raw_output": parsed_result["raw_output"],
                "command": command,
            }

        def summary(self, latency_us, kernel_mapping={}):
            if self._sycl_tla_result:
                latency_us = self._sycl_tla_result["latency_ms"] * 1000.0
                kernel_mapping = dict(kernel_mapping)
                kernel_mapping["tflops"] = self._sycl_tla_result["tflops"]
            return super().summary(latency_us, kernel_mapping)


except Exception as e:
    print(f"[SyclTlaMoeGatingGemmOp] Failed to register: {e}")
    pass