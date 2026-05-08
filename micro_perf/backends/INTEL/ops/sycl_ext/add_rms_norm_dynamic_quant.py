import sys
import pathlib
import importlib.util
from functools import partial

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size


_OP_DIR = pathlib.Path(__file__).resolve().parent
_SYCL_SO = _OP_DIR / "add_rms_norm_dynamic_quant_sycl.so"


try:
    _spec = importlib.util.spec_from_file_location(
        "add_rms_norm_dynamic_quant_sycl", str(_SYCL_SO))
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Failed to create import spec for {_SYCL_SO}")
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl(
        "add_rms_norm_dynamic_quant", "sycl_ext")
    class AddRmsNormDynamicQuantSyclExtOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if self.arg_type not in ["llm"]:
                raise NotImplementedError

            # src_dtype
            self.dtype = self.args_dict["dtype"]
            if self.dtype not in ["float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            # dst_dtype
            self.dst_dtype = self.args_dict["dst_dtype"]
            if self.dst_dtype not in ["int8"]:
                raise NotImplementedError
            self.dst_torch_dtype = getattr(torch, self.dst_dtype)

            # pre-defined attrs
            self.add_residual = self.args_dict.get("add_residual", True)
            self.num_tokens = self.args_dict["num_tokens"]
            self.hidden_size = self.args_dict["hidden_size"]

            self.eps = 1e-5

            # input/output tensors
            self.input_tensor_info = {
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "norm_weight": OpTensorInfo(
                    shape=[self.hidden_size],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones,
                ),
                "smooth_scale": OpTensorInfo(
                    shape=[self.hidden_size],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones,
                ),
            }
            if self.add_residual:
                self.input_tensor_info["residual"] = OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )

            self.output_tensor_info = {
                "quant_tokens": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.dst_torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "per_token_scale": OpTensorInfo(
                    shape=[self.num_tokens],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                ),
                "after_res": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "after_norm": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
            }

            # calculator
            self.input_tensor_size = sum(
                calc_tensor_size(info)
                for info in self.input_tensor_info.values()
            )
            self.output_tensor_size = sum(
                calc_tensor_size(info)
                for info in self.output_tensor_info.values()
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )

            # run func
            self._run_func = self.add_rms_norm_dynamic_quant_run

        def add_rms_norm_dynamic_quant_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            residual = tensor_mapping.get("residual", None)
            norm_weight = tensor_mapping["norm_weight"]
            smooth_scale = tensor_mapping["smooth_scale"]
            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]
            after_res = tensor_mapping["after_res"]
            after_norm = tensor_mapping["after_norm"]

            # Pass empty tensor when no residual
            if residual is None:
                residual = torch.empty(0, device=hidden_states.device,
                                       dtype=hidden_states.dtype)

            _sycl_ext.add_rms_norm_dynamic_quant_forward(
                hidden_states, residual, norm_weight, smooth_scale,
                quant_tokens, per_token_scale, after_res, after_norm,
                self.eps)

            return quant_tokens, per_token_scale, after_res, after_norm

except Exception as e:
    import warnings
    warnings.warn(
        f"Failed to load SYCL add_rms_norm_dynamic_quant extension: {e}")
