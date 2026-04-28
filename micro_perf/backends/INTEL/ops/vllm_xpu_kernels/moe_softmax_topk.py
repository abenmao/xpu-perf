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
            self.output_tensor_info["token_expert_indices"] = OpTensorInfo(
                shape=[self.num_tokens * self.topk],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
            )

            self.output_tensor_size = sum(
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes   
        
        def vendor_impl_run(self, tensor_mapping):
            gating_output = tensor_mapping["gating_output"]
            selected_experts = tensor_mapping["selected_experts"]
            moe_weights = tensor_mapping["moe_weights"]
            token_expert_indices = tensor_mapping["token_expert_indices"]
            # _moe_C::topk_softmax needs int32 output for indices
            
            renormalize = self.compute_mode == "pre-softmax"
            torch.ops._moe_C.topk_softmax(
                moe_weights, selected_experts, token_expert_indices,
                gating_output, renormalize, None
            )

            # copy back int32 indices to the pre-allocated float32 tensor
            return selected_experts, moe_weights

except Exception:
    pass