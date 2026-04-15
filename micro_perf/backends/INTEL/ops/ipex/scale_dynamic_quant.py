import sys
import pathlib
from functools import partial

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import ScaleDynamicQuantOp

try:
    import torch
    torch.ops.torch_ipex.scale_dynamic_quant

    @ProviderRegistry.register_vendor_impl("scale_dynamic_quant", "ipex")
    class ScaleDynamicQuantIpexOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError

            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            self.dst_dtype = self.args_dict["dst_dtype"]
            if not self.dst_dtype in ["int8"]:
                raise NotImplementedError
            self.dst_torch_dtype = getattr(torch, self.dst_dtype)

            self.num_tokens = self.args_dict["num_tokens"]
            self.hidden_size = self.args_dict["hidden_size"]

            self.input_tensor_info = {
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "smooth_scale": OpTensorInfo(
                    shape=[self.hidden_size],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                )
            }
            self.output_tensor_info = {
                "quant_tokens": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.dst_torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "per_token_scale": OpTensorInfo(
                    shape=[self.num_tokens],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                )
            }

            self.input_tensor_size = sum([
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            ])
            self.output_tensor_size = sum([
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            ])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True
            )

            self._run_func = self.scale_dynamic_quant

        def scale_dynamic_quant(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            smooth_scale = tensor_mapping["smooth_scale"]
            quant_out = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            if hidden_states.dtype == torch.bfloat16:
                hidden_states = hidden_states.to(torch.float16)

            torch.ops.torch_ipex.scale_dynamic_quant(hidden_states, smooth_scale,
                                                     quant_out, per_token_scale)
            return quant_out

except Exception:
    pass

OP_MAPPING = {"torch": ScaleDynamicQuantOp}
