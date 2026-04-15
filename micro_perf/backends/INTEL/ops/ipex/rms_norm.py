import sys
import pathlib
from functools import partial

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.vector_norm_ops import RMSNormOp

try:
    import torch
    torch.ops.torch_ipex.residual_rms_norm

    @ProviderRegistry.register_vendor_impl("rms_norm", "ipex")
    class RMSNormIpexOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["default", "llm"]:
                raise NotImplementedError

            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["float32", "float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            self.add_residual = self.args_dict.get("add_residual", True)
            if not self.add_residual in [True, False]:
                raise NotImplementedError

            self.epsilon = 1e-5

            if self.arg_type == "default":
                self.batch_size = self.args_dict["batch_size"]
                self.dim_size = self.args_dict["dim_size"]
            elif self.arg_type == "llm":
                self.batch_size = self.args_dict["num_tokens"]
                self.dim_size = self.args_dict["hidden_size"]

            self.input_tensor_info = {
                "src": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.randn
                ),
                "weight": OpTensorInfo(
                    shape=[self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ),
                "after_res": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.empty
                )
            }
            self.output_tensor_info = {
                "after_norm": OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.empty
                ),
            }

            if self.add_residual:
                self.input_tensor_info["residual"] = OpTensorInfo(
                    shape=[self.batch_size, self.dim_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.randn
                )

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

            self.calc_flops = self.batch_size * (
                    3 * self.dim_size + 4
            )
            if self.add_residual:
                self.calc_flops += self.batch_size * self.dim_size

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )

            self._run_func = self.rms_norm_run

        def rms_norm_run(self, tensor_mapping):
            src = tensor_mapping["src"]
            weight = tensor_mapping["weight"]
            after_res = tensor_mapping["after_res"]
            after_norm = tensor_mapping["after_norm"]
            if self.add_residual:
                residual = tensor_mapping["residual"]
            else:
                residual = None
            torch.ops.torch_ipex.residual_rms_norm(weight, residual, src,
                                                   after_res, after_norm,
                                                   self.epsilon)
            return after_norm

except Exception:
    pass

OP_MAPPING = {"torch": RMSNormOp}
