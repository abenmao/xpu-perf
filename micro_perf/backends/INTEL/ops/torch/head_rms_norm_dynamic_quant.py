import sys
import pathlib
from functools import partial

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import BasicOp
from core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype, smooth_per_token_dynamic_quant


class HeadRMSNormDynamicQuantOp(BasicOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if not self.arg_type in ["llm"]:
            raise ValueError
        
        self.dtype = self.args_dict["dtype"]
        if not self.dtype in ["bfloat16"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        self.dst_dtype = self.args_dict["dst_dtype"]
        if not self.dst_dtype in ["int8", "float8"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        # pre-defined attrs
        self.sp_size = self.args_dict.get("sp_size", 1)
        self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
        self.head_num = self.args_dict["head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.eps = 1e-5

        # out-place
        self.input_tensor_info = {
            "token_data": OpTensorInfo(
                shape=[self.num_tokens, self.head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            ),
            "norm_weight": OpTensorInfo(
                shape=[self.head_dim, ],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ), 
            # use 1 as smooth scale
            "smooth_scale": OpTensorInfo(
                shape=[self.head_num * self.head_dim],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.ones
            ),
        }
        self.output_tensor_info = {
            "quant_tokens": OpTensorInfo(
                shape=[self.num_tokens, self.head_num * self.head_dim], 
                dtype=self.dst_torch_dtype, 
                device=self.backend.get_torch_device_name(),
            ), 
            "per_token_scale": OpTensorInfo(
                shape=[self.num_tokens], 
                dtype=torch.float32, 
                device=self.backend.get_torch_device_name()
            )
        }

        # calculator
        self.input_tensor_size = 2 * sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
        self.write_bytes = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
        self.io_bytes = self.read_bytes + self.write_bytes

        # creator func
        self._create_tensors_func = partial(
            self._create_in_out_tensors, 
            create_inputs=True, 
            create_outputs=False
        )

        # run func
        self._run_func = self.head_rms_norm_dynamic_quant_run

    def head_rms_norm_dynamic_quant_run(self, tensor_mapping):
        # get pre-allocated input tensors
        token_data = tensor_mapping["token_data"]
        norm_weight = tensor_mapping["norm_weight"]
        smooth_scale = tensor_mapping["smooth_scale"]

        # per head rms_norm
        after_norm = torch.nn.functional.rms_norm(
            token_data, 
            normalized_shape=token_data.shape[-1:],
            weight=norm_weight,
            eps=self.eps
        )
        after_norm = after_norm.view(self.num_tokens, self.head_num * self.head_dim)

        # per token dynamic quant
        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
            after_norm, smooth_scale, self.dst_torch_dtype
        )

        return quant_tokens, per_token_scale


OP_MAPPING = {"torch": HeadRMSNormDynamicQuantOp}
