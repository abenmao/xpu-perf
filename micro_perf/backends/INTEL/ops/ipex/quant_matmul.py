import sys
import pathlib
import torch
import random
from functools import partial
from itertools import combinations

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import QuantMatmulOp

try:
    torch.ops.torch_ipex.mm_w8a8

    @ProviderRegistry.register_vendor_impl("quant_matmul", "ipex")
    class QuantMatmulIpexOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError
        
            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["int8"]:
                raise NotImplementedError

            self.sp_size = self.args_dict.get("sp_size", 1)
            self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
            self.hidden_size = self.args_dict["hidden_size"]
            self.new_hidden_size = self.args_dict["new_hidden_size"]
            self.trans_w = self.args_dict.get("trans_w", False)

            # Weight shape depends on trans_w: if True, [hidden_size, new_hidden_size]
            w_shape = [self.hidden_size, self.new_hidden_size] if self.trans_w else [self.new_hidden_size, self.hidden_size]
        
            # Output dtype from dst_dtype arg
            dst_dtype_str = self.args_dict.get("dst_dtype", "float16")
            dst_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            self.dst_torch_dtype = dst_dtype_map.get(dst_dtype_str, torch.float16)

            self.input_tensor_info = {
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size], 
                    dtype=torch.int8, 
                    device=self.backend.get_torch_device_name()
                ), 
                "weight": OpTensorInfo(
                    shape=w_shape, 
                    dtype=torch.int8, 
                    device=self.backend.get_torch_device_name()
                ), 
                "per_token_scale": OpTensorInfo(
                    shape=[self.num_tokens], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name()
                ), 
                "weight_scale": OpTensorInfo(
                    shape=[self.new_hidden_size], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name()
                ), 
                "zero_points": OpTensorInfo(
                    shape=[1],
                    dtype=torch.int8, 
                    device=self.backend.get_torch_device_name()),
            }
            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.num_tokens, self.new_hidden_size], 
                    dtype=self.dst_torch_dtype, 
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

            # gemm:         [M, K] * [K, N] = [M, N]
            # scale:        [M, 1] * [1, N] --> [M, N]
            # cast:         [M, N] --> [M, N]
            # dequantize:   [M, N] --> [M, N]
            self.calc_flops = 2 * self.num_tokens * self.hidden_size * self.new_hidden_size

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )

            self._run_func = self.quant_matmul_run

        def quant_matmul_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            weight = tensor_mapping["weight"]
            per_token_scale = tensor_mapping["per_token_scale"]
            weight_scale = tensor_mapping["weight_scale"]
            zero_points = tensor_mapping["zero_points"]
            out = tensor_mapping["out"]

            weight = weight.transpose(0, 1)

            torch.ops.torch_ipex.mm_w8a8(
                hidden_states,
                per_token_scale,
                None,
                weight,
                weight_scale,
                zero_points,
                out
            )



except Exception:
    pass

OP_MAPPING = {"torch": QuantMatmulOp}
