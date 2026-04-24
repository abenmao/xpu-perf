import sys
import pathlib
from functools import partial
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeSoftmaxTopkOp
from core.utils import OpTensorInfo, calc_tensor_size

try:
    import vllm_xpu_kernels._moe_C

    @ProviderRegistry.register_vendor_impl("moe_softmax_topk", "vllm_xpu_kernels")
    class VLLMXPUKernelsMoeSoftmaxTopkOp(MoeSoftmaxTopkOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
        
        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError

            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["float32"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            self.num_experts = self.args_dict["num_experts"]
            
            self.topk = self.args_dict["topk"]

            self.compute_mode = self.args_dict["compute_mode"]
            if not self.compute_mode in ["pre-softmax", "post-softmax"]:
                raise NotImplementedError

            self.sp_size = self.args_dict.get("sp_size", 1)
            self.num_tokens = self.args_dict["num_tokens"] // self.sp_size
            self.hidden_size = self.args_dict["hidden_size"]

            self.input_tensor_info = {
                "gating_output": OpTensorInfo(
                    shape=[self.num_tokens, self.num_experts],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }
            self.output_tensor_info = {
                "selected_experts": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                ),
                "moe_weights": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "token_expert_indices": OpTensorInfo(
                    shape=[self.num_tokens * self.topk],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
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

            self.calc_flops = self.num_tokens * (5 * self.num_experts + self.topk * self.num_experts)
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            gating_output = tensor_mapping["gating_output"]
            selected_experts = tensor_mapping["selected_experts"]
            moe_weights = tensor_mapping["moe_weights"]
            token_expert_indices = tensor_mapping["token_expert_indices"]
            
            torch.ops._moe_C.topk_softmax(
                moe_weights, selected_experts, token_expert_indices,
                gating_output, True , None
            )

            return selected_experts, moe_weights

except Exception:
    pass
