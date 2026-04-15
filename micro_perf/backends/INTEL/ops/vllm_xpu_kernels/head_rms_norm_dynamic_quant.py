import sys
import pathlib

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import HeadRMSNormDynamicQuantOp
from core.utils import smooth_per_token_dynamic_quant


try:
    import vllm_xpu_kernels._C

    @ProviderRegistry.register_vendor_impl("head_rms_norm_dynamic_quant", "vllm_xpu_kernels")
    class VLLMXPUKernelsHeadRMSNormDynamicQuantOp(HeadRMSNormDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

        def head_rms_norm_dynamic_quant_run(self, tensor_mapping):
            token_data = tensor_mapping["token_data"]
            norm_weight = tensor_mapping["norm_weight"]
            smooth_scale = tensor_mapping["smooth_scale"]

            # vllm_xpu_kernels exposes rms_norm(out, input, weight, eps)
            # and requires input/weight dtype alignment.
            token_data_contiguous = token_data.contiguous()
            norm_weight = norm_weight.to(token_data_contiguous.dtype)
            after_norm = torch.empty_like(token_data_contiguous)
            torch.ops._C.rms_norm(after_norm, token_data_contiguous, norm_weight, self.eps)

            # Follow core op output contract: flatten to [num_tokens, head_num * head_dim].
            after_norm = after_norm.view(self.num_tokens, self.head_num * self.head_dim)

            quant_tokens, per_token_scale = smooth_per_token_dynamic_quant(
                after_norm,
                smooth_scale,
                self.dst_torch_dtype,
            )

            return quant_tokens, per_token_scale

except Exception:
    pass
