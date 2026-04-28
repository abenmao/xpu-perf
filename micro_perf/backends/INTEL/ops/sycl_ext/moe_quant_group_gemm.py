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
from core.ops.llm_ops import MoeQuantGroupGemmOp
import torch


def _flush_all_stdio():
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        ctypes.CDLL(None).fflush(None)
    except Exception:
        pass

try:
    _XPU_PERF_ROOT = pathlib.Path(__file__).resolve().parents[5]
    _SYCL_EXT_DIR = os.path.join(
        str(_XPU_PERF_ROOT),
        "micro_perf/backends/INTEL/ops/sycl_ext",
    )
    SYCL_EXT_INT_SO = os.path.join(
        _SYCL_EXT_DIR,
        "bmg_moe_quant_grouped_gemm_int8_sycl.so",
    )
    SYCL_EXT_FLOAT8_SO = os.path.join(
        _SYCL_EXT_DIR,
        "bmg_moe_quant_grouped_gemm_fp8_sycl.so",
    )

    def _load_sycl_ext(module_name, module_path):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _has_sycl_ext(module_path):
        return os.path.isfile(module_path)

    if not _has_sycl_ext(SYCL_EXT_INT_SO) and not _has_sycl_ext(SYCL_EXT_FLOAT8_SO):
        print(
            "[WARNING] sycl_ext moe_quant_group_gemm shared objects not found. "
            "sycl_tla_moe_quant_group_gemm provider will NOT be available. "
            f"Expected built .so files under {_SYCL_EXT_DIR}."
        )
        raise FileNotFoundError(_SYCL_EXT_DIR)

    @ProviderRegistry.register_vendor_impl(
        "moe_quant_group_gemm", "sycl_tla_moe_quant_group_gemm"
    )
    class SyclTlaMoeQuantGroupGemmOp(MoeQuantGroupGemmOp):
        SUPPORTED_OUTPUT_DTYPES = {
            "bfloat16",
            "float16",
        }
        FLOAT8_DTYPES = {
            "float8",
            "float8_e4m3",
            "float8_e5m2",
        }

        def vendor_parser(self):
            if self._is_float8_path():
                if self.dtype not in self.FLOAT8_DTYPES:
                    raise ValueError(
                        "sycl_tla_moe_quant_group_gemm float8 path requires dtype in "
                        f"{sorted(self.FLOAT8_DTYPES)}, got {self.dtype}"
                    )
                if self.w_dtype not in self.FLOAT8_DTYPES:
                    raise ValueError(
                        "sycl_tla_moe_quant_group_gemm float8 path requires w_dtype in "
                        f"{sorted(self.FLOAT8_DTYPES)}, got {self.w_dtype}"
                    )
                if self._normalize_float8_dtype(self.dtype) != self._normalize_float8_dtype(self.w_dtype):
                    raise ValueError(
                        "sycl_tla_moe_quant_group_gemm float8 path requires dtype and w_dtype "
                        f"to match, got dtype={self.dtype}, w_dtype={self.w_dtype}"
                    )
                if self.dst_dtype not in self.SUPPORTED_OUTPUT_DTYPES:
                    raise ValueError(
                        "sycl_tla_moe_quant_group_gemm float8 path requires dst_dtype in "
                        f"{sorted(self.SUPPORTED_OUTPUT_DTYPES)}, got {self.dst_dtype}"
                    )
            else:
                if self.dtype != "int8":
                    raise ValueError(
                        f"sycl_tla_moe_quant_group_gemm requires dtype=int8, got {self.dtype}"
                    )
                if self.w_dtype not in {"int8", "int4"}:
                    raise ValueError(
                        "sycl_tla_moe_quant_group_gemm requires w_dtype in ['int4', 'int8'], "
                        f"got {self.w_dtype}"
                    )
            if self.dst_dtype not in self.SUPPORTED_OUTPUT_DTYPES:
                raise ValueError(
                    "sycl_tla_moe_quant_group_gemm requires dst_dtype in "
                    f"{sorted(self.SUPPORTED_OUTPUT_DTYPES)}, got {self.dst_dtype}"
                )

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self._provider = "sycl_tla_moe_quant_group_gemm"
            # Default quant_group_size: 128 for int4 (per-group), hidden_size for int8 (per-column)
            _default_gs = 128 if self.w_dtype == "int4" else self.hidden_size
            self._quant_group_size = int(
                self.args_dict.get("quant_group_size", _default_gs)
            )
            self._binary_kind, self._extension_module, self._extension_func_name = self._select_extension_config()
            self._sycl_tla_result = self._run_sycl_ext()

            self._run_func = lambda tensor_mapping: None
            self._create_tensors_func = (
                lambda instance_num: [{} for _ in range(max(instance_num, 1))]
            )

        def _is_float8_path(self):
            return self.dtype in self.FLOAT8_DTYPES or self.w_dtype in self.FLOAT8_DTYPES

        def _normalize_float8_dtype(self, dtype):
            if dtype == "float8":
                return "float8_e4m3"
            return dtype

        def _select_extension_config(self):
            if self._is_float8_path():
                if not _has_sycl_ext(SYCL_EXT_FLOAT8_SO):
                    raise FileNotFoundError(
                        f"float8 sycl_ext module not found: {SYCL_EXT_FLOAT8_SO}"
                    )
                return (
                    "float8",
                    _load_sycl_ext("bmg_moe_quant_grouped_gemm_fp8_sycl", SYCL_EXT_FLOAT8_SO),
                    "run_moe_quant_grouped_gemm_fp8",
                )
            if self.w_dtype not in {"int8", "int4"}:
                raise ValueError(
                    "sycl_tla_moe_quant_group_gemm only supports w_dtype in ['int4', 'int8'], "
                    f"got {self.w_dtype}"
                )
            if not _has_sycl_ext(SYCL_EXT_INT_SO):
                raise FileNotFoundError(
                    f"int8 sycl_ext module not found: {SYCL_EXT_INT_SO}"
                )
            return (
                self.w_dtype,
                _load_sycl_ext("bmg_moe_quant_grouped_gemm_int8_sycl", SYCL_EXT_INT_SO),
                "run_moe_quant_grouped_gemm_int8",
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
            runtime_match = re.search(
                r"Avg runtime\s*:\s*([\d\.eE+-]+)\s*ms", output_text
            )
            gflops_match = re.search(
                r"GFLOPS\s*:\s*([\d\.eE+-]+)", output_text
            )
            groups_match = re.search(
                r"Groups\s*:\s*(\d+)", output_text
            )
            mode_match = re.search(
                r"Running in\s+([A-Za-z]+)\s+mode", output_text
            )
            narrow_match = re.search(
                r"Setting\s+(A\s+and\s+B|[AB])\s+as\s+narrower\s+type", output_text
            )

            if runtime_match is None or gflops_match is None:
                raise RuntimeError(f"Failed to parse sycl-tla output:\n{output_text}")

            return {
                "latency_ms": float(runtime_match.group(1)),
                "gflops": float(gflops_match.group(1)),
                "groups": int(groups_match.group(1)) if groups_match else None,
                "mode": mode_match.group(1) if mode_match else None,
                "narrow_operand": (
                    "A+B" if narrow_match and narrow_match.group(1).startswith("A") and "and" in narrow_match.group(1)
                    else narrow_match.group(1) if narrow_match else None
                ),
                "raw_output": output_text,
            }

        def _run_sycl_ext(self):
            extension_func = getattr(self._extension_module, self._extension_func_name)

            if self._binary_kind == "float8":
                call_args = (
                    self.num_tokens,
                    self.hidden_size,
                    self.new_hidden_size,
                    self.num_experts,
                    self.topk,
                    self.ep_size,
                    self.ep_rank,
                    self._quant_group_size,
                    1.0,
                    0.0,
                    10,
                    1,
                    self.dtype,
                    self.w_dtype,
                    self.compute_dtype,
                    self.dst_dtype,
                )
            else:
                call_args = (
                    self.num_tokens,
                    self.hidden_size,
                    self.new_hidden_size,
                    self.num_experts,
                    self.topk,
                    self.ep_size,
                    self.ep_rank,
                    self._quant_group_size,
                    10,
                    1,
                    self.dtype,
                    self.w_dtype,
                    self.compute_dtype,
                    self.dst_dtype,
                )

            command = f"{self._extension_module.__name__}.{self._extension_func_name}(...)"
            print(f"[SyclTlaMoeQuantGroupGemmOp] Running: {command}")

            retcode, output_text = self._capture_extension_output(extension_func, *call_args)
            if retcode != 0:
                raise RuntimeError(
                    f"sycl_ext call failed (rc={retcode}):\n{output_text}"
                )

            print(f"[SyclTlaMoeQuantGroupGemmOp] Output:\n{output_text}")

            parsed_result = self._parse_sycl_tla_output(output_text)
            return {
                "binary_kind": self._binary_kind,
                "latency_ms": parsed_result["latency_ms"],
                "gflops": parsed_result["gflops"],
                "groups": parsed_result["groups"],
                "mode": parsed_result["mode"],
                "narrow_operand": parsed_result["narrow_operand"],
                "quant_group_size": self._quant_group_size,
                "raw_output": parsed_result["raw_output"],
                "command": command,
            }

        def summary(self, latency_us, kernel_mapping={}):
            if self._sycl_tla_result:
                latency_us = self._sycl_tla_result["latency_ms"] * 1000.0
                kernel_mapping = dict(kernel_mapping)
                kernel_mapping["binary_kind"] = self._sycl_tla_result["binary_kind"]
                kernel_mapping["gflops"] = self._sycl_tla_result["gflops"]
                kernel_mapping["groups"] = self._sycl_tla_result["groups"]
                kernel_mapping["mode"] = self._sycl_tla_result["mode"]
                kernel_mapping["narrow_operand"] = self._sycl_tla_result["narrow_operand"]
                kernel_mapping["quant_group_size"] = self._sycl_tla_result["quant_group_size"]
            return super().summary(latency_us, kernel_mapping)


except Exception as e:
    print(f"[SyclTlaMoeQuantGroupGemmOp] Failed to register: {e}")
    pass
