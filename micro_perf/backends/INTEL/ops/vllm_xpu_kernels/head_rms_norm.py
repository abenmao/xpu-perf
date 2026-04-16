import sys
import pathlib
from functools import partial

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import HeadRMSNormOp
from core.utils import calc_tensor_size


try:
    import vllm_xpu_kernels._C

    @ProviderRegistry.register_vendor_impl("head_rms_norm", "vllm_xpu_kernels")
    class VLLMXPUKernelsHeadRMSNormOp(HeadRMSNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["vllm_xpu_kernels"]

            # No changes to _create_tensors_func, keep parent's in-place mode
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=False,
            )

            # Fix io_bytes in provider layer: clip active heads to valid range.
            effective_norm_head_num = max(
                0,
                min(self.norm_head_num, self.total_head_num - self.norm_head_start),
            )
            token_data_size = calc_tensor_size(self.input_tensor_info["token_data"])
            norm_weight_size = calc_tensor_size(self.input_tensor_info["norm_weight"])

            self.read_bytes = token_data_size / self.total_head_num * effective_norm_head_num
            self.write_bytes = self.read_bytes
            self.read_bytes += norm_weight_size
            self.io_bytes = self.read_bytes + self.write_bytes

        def vendor_impl_run(self, tensor_mapping):
            # get pre-allocated input tensors
            token_data = tensor_mapping["token_data"]
            norm_weight = tensor_mapping["norm_weight"]

            # in-place norm on specified heads
            head_data = token_data[:, self.norm_head_start:self.norm_head_end, :]
            need_copy = not head_data.is_contiguous()
            head_data_contiguous = head_data.contiguous()
            norm_weight = norm_weight.to(head_data_contiguous.dtype)

            # vllm_xpu_kernels exposes rms_norm(out, input, weight, eps), not head_rms_norm.
            torch.ops._C.rms_norm(head_data_contiguous, head_data_contiguous, norm_weight, self.eps)
            if need_copy:
                head_data.copy_(head_data_contiguous)

            return token_data

except Exception:
    pass
