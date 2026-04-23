import os
import sys
import pathlib
import importlib.util
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.vector_reduction_ops import ReduceMaxOp


try:
    _OP_DIR = pathlib.Path(__file__).resolve().parent
    _SYCL_SO = _OP_DIR / "reduce_max_sycl.so"

    @ProviderRegistry.register_vendor_impl("reduce_max", "sycl_ext")
    class SyclExtReduceMaxOp(ReduceMaxOp):
        _SO_MODULE = None

        def _get_int_option(self, key, env_key, default):
            value = self.args_dict.get(key, os.getenv(env_key, default))
            try:
                return int(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid integer option for {key}/{env_key}: {value}"
                ) from e

        def __init__(self, args_dict, backend, *args, **kwargs):
            # Provider entry: parse runtime config, load SO, and bind run/tensor callbacks.
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.arg_type != "default":
                raise ValueError("SyclExtReduceMaxOp only supports arg_type=default")

            self._so_cfg = self._parse_so_config()
            self._sycl_so = self._load_sycl_so()

            # Optional one-time correctness gate; timing remains in framework core_perf.
            if self._so_cfg["verify"]:
                self._verify_so_compute_once()

            self._run_func = self._run_sycl_so_compute
            self._create_tensors_func = self._create_tensors_for_so

        @classmethod
        def _load_sycl_so(cls):
            if cls._SO_MODULE is not None:
                return cls._SO_MODULE

            if not _SYCL_SO.is_file():
                raise FileNotFoundError(
                    f"reduce_max sycl extension not found: {_SYCL_SO}. "
                    "Please run backends/INTEL/ops/sycl_ext/build.sh first."
                )

            spec = importlib.util.spec_from_file_location("reduce_max_sycl", str(_SYCL_SO))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls._SO_MODULE = module
            return cls._SO_MODULE

        def _resolve_rows_per_group(self, value):
            if value is None:
                return None

            if isinstance(value, str) and value.strip().lower() == "auto":
                # Conservative table tuned for small dims only.
                if self.dim_size <= 1024:
                    return 8
                if self.dim_size <= 2048:
                    return 4
                return None

            try:
                return int(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "Invalid integer option for sycl_ext_rows_per_group/"
                    "SYCL_EXT_REDUCE_MAX_ROWS_PER_GROUP: "
                    f"{value}"
                ) from e

        def _parse_so_config(self):
            # Parse all sycl_ext runtime knobs from args/env and normalize types/defaults.
            iterations = self._get_int_option(
                key="sycl_ext_iterations",
                env_key="SYCL_EXT_REDUCE_MAX_ITERATIONS",
                default=100,
            )
            verify = self._get_int_option(
                key="sycl_ext_verify",
                env_key="SYCL_EXT_REDUCE_MAX_VERIFY",
                default=0,
            )
            warmup = self.args_dict.get(
                "sycl_ext_warmup",
                os.getenv("SYCL_EXT_REDUCE_MAX_WARMUP", 0),
            )
            try:
                warmup = int(warmup)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid integer option for sycl_ext_warmup/SYCL_EXT_REDUCE_MAX_WARMUP: {warmup}"
                ) from e

            input_mode = self.args_dict.get(
                "sycl_ext_input_mode",
                os.getenv("SYCL_EXT_REDUCE_MAX_INPUT_MODE", "sin"),
            )
            input_mode = str(input_mode).strip().lower()
            if input_mode not in {"sin", "tie", "nan_head", "nan_mix"}:
                raise ValueError(
                    "Invalid sycl_ext_input_mode/SYCL_EXT_REDUCE_MAX_INPUT_MODE. "
                    "Expected one of: sin, tie, nan_head, nan_mix"
                )

            smalldim_mode = self.args_dict.get(
                "sycl_ext_smalldim_mode",
                os.getenv("SYCL_EXT_REDUCE_MAX_SMALLDIM_MODE", "multirow"),
            )
            smalldim_mode = str(smalldim_mode).strip().lower()
            if smalldim_mode not in {"baseline", "multirow"}:
                raise ValueError(
                    "Invalid sycl_ext_smalldim_mode/SYCL_EXT_REDUCE_MAX_SMALLDIM_MODE. "
                    "Expected one of: baseline, multirow"
                )

            rows_per_group = self.args_dict.get(
                "sycl_ext_rows_per_group",
                os.getenv("SYCL_EXT_REDUCE_MAX_ROWS_PER_GROUP", "auto"),
            )
            resolved_rows_per_group = self._resolve_rows_per_group(rows_per_group)

            return {
                "iterations": iterations,
                "verify": verify,
                "warmup": warmup,
                "input_mode": input_mode,
                "smalldim_mode": smalldim_mode,
                "rows_per_group": resolved_rows_per_group,
                "so_variant": self.args_dict.get(
                    "sycl_ext_so_variant",
                    os.getenv("SYCL_EXT_REDUCE_MAX_SO_VARIANT", "so_v0"),
                ),
            }

        def _fill_src_tensor(self, src):
            input_mode = self._so_cfg["input_mode"]
            device = src.device
            dtype = src.dtype

            if input_mode == "sin":
                idx = torch.arange(self.batch_size * self.dim_size, device=device, dtype=torch.float32)
                src.copy_(torch.sin(0.001 * idx).reshape(self.batch_size, self.dim_size).to(dtype))
                return

            if input_mode == "tie":
                row = (torch.arange(self.dim_size, device=device, dtype=torch.float32) % 17) - 8
                if self.dim_size > 7:
                    row[7] = 1234.0
                if self.dim_size > 19:
                    row[19] = 1234.0
                src.copy_(row.unsqueeze(0).expand(self.batch_size, -1).to(dtype))
                return

            if input_mode == "nan_head":
                b = torch.arange(self.batch_size, device=device, dtype=torch.float32).unsqueeze(1)
                d = torch.arange(self.dim_size, device=device, dtype=torch.float32).unsqueeze(0)
                t = torch.sin(0.01 * (b + d))
                if self.dim_size > 0:
                    t[:, 0] = float("nan")
                src.copy_(t.to(dtype))
                return

            if input_mode == "nan_mix":
                idx = torch.arange(self.batch_size * self.dim_size, device=device, dtype=torch.float32)
                t = torch.cos(0.01 * idx).reshape(self.batch_size, self.dim_size)
                if self.dim_size > 3:
                    t[:, 3] = float("nan")
                if self.dim_size > 9:
                    t[:, 9] = float("nan")
                src.copy_(t.to(dtype))
                return

            raise ValueError(f"Unsupported input mode: {input_mode}")

        def _create_tensors_for_so(self, instance_num):
            all_tensor_list = self._create_in_out_tensors(
                instance_num,
                create_inputs=True,
                create_outputs=True,
            )
            for tensor_mapping in all_tensor_list:
                self._fill_src_tensor(tensor_mapping["src"])
            return all_tensor_list

        def _run_sycl_so_compute(self, tensor_mapping):
            # Hot path entry used by framework timing: dispatch one compute_into call to SO.
            rows_per_group_arg = -1 if self._so_cfg["rows_per_group"] is None else int(self._so_cfg["rows_per_group"])

            self._sycl_so.reduce_max_compute_into(
                tensor_mapping["src"],
                tensor_mapping["max_value"],
                tensor_mapping["max_indices"],
                str(self._so_cfg["smalldim_mode"]),
                int(rows_per_group_arg),
            )
            return None

        def _verify_so_compute_once(self):
            # Optional one-shot correctness gate against torch.max before perf runs.
            tensor_mapping = self._create_tensors_for_so(1)[0]
            self._run_sycl_so_compute(tensor_mapping)
            got_values = tensor_mapping["max_value"]
            got_indices = tensor_mapping["max_indices"]
            if hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()
            src = tensor_mapping["src"]

            ref_values, ref_indices = torch.max(src, dim=-1, keepdim=True)
            if not torch.equal(got_values, ref_values) or not torch.equal(got_indices, ref_indices.to(torch.int32)):
                raise RuntimeError("reduce_max_sycl.so verify failed")

        def summary(self, latency_us, kernel_mapping={}):
            # Report entry: append sycl_ext config fields to the common summary schema.
            kernel_name = "sycl_ext_reduce_max_sycl_so"
            target_dict = super().summary(
                latency_us,
                [
                    kernel_name,
                ],
            )
            if target_dict:
                target_dict["sycl_ext_iterations"] = self._so_cfg.get("iterations")
                target_dict["sycl_ext_verify"] = self._so_cfg.get("verify")
                target_dict["sycl_ext_warmup"] = self._so_cfg.get("warmup")
                target_dict["sycl_ext_smalldim_mode"] = self._so_cfg.get("smalldim_mode")
                target_dict["sycl_ext_rows_per_group"] = self._so_cfg.get("rows_per_group")
                target_dict["sycl_ext_so_variant"] = self._so_cfg.get("so_variant")
            return target_dict

except Exception as e:
    print(f"[SyclExtReduceMaxOp] Failed to register: {e}")
    pass
