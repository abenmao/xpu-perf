import sys
import pathlib
from functools import partial

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import HeadRMSNormOp as _HeadRMSNormBaseOp

try:
    import torch
    torch.ops.torch_ipex.head_rms_norm

    @ProviderRegistry.register_vendor_impl("head_rms_norm", "ipex")
    class HeadRMSNormOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError

            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["float32", "float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            # pre-defined attrs
            self.num_tokens = self.args_dict["num_tokens"]
            self.total_head_num = self.args_dict["total_head_num"]
            self.head_dim = self.args_dict["head_dim"]

            self.norm_head_start = self.args_dict.get("norm_head_start", 0)
            self.norm_head_num = self.args_dict.get("norm_head_num", self.total_head_num)
            self.norm_head_end = self.norm_head_start + self.norm_head_num

            if self.norm_head_start != 0 or self.norm_head_num != self.total_head_num:
                raise NotImplementedError

            self.eps = 1e-5

            self.input_tensor_info = {
                "token_data": OpTensorInfo(
                    shape=[self.num_tokens, self.total_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "weight": OpTensorInfo(
                    shape=[self.norm_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }
            self.output_tensor_info = {
                "y": OpTensorInfo(
                    shape=[self.num_tokens, self.total_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }

            # calculator
            self.input_tensor_size = sum([
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            ])
            self.output_tensor_size = sum([
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            ])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = \
                calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_head_num * self.norm_head_num + \
                calc_tensor_size(self.input_tensor_info["weight"])
            self.write_bytes = \
                calc_tensor_size(self.output_tensor_info["y"]) / self.total_head_num * self.norm_head_num
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True
            )

            # run func
            self._run_func = self.head_rms_norm_run

        def head_rms_norm_run(self, tensor_mapping):
            # get pre-allocated input tensors
            token_data = tensor_mapping["token_data"]
            weight = tensor_mapping["weight"]

            # get pre-allocated output tensors
            y = tensor_mapping["y"]

            torch.ops.torch_ipex.head_rms_norm(weight, token_data, y, self.eps, self.norm_head_num)
            return y



except Exception:
    pass

OP_MAPPING = {"torch": _HeadRMSNormBaseOp}
