import sys
import pathlib
import traceback

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)
from core.op import ProviderRegistry
from core.utils import calc_tensor_size
from core.ops.llm_ops import HeadRMSNormOp as BaseHeadRMSNormOp


@ProviderRegistry.register_vendor_impl("head_rms_norm", "torch")
class HeadRMSNormTorchOp(BaseHeadRMSNormOp):

    def vendor_impl(self):
        super().vendor_impl()

        # Keep effective byte accounting consistent with edge case handling.
        effective_norm_head_num = max(
            0,
            min(self.norm_head_num, self.total_head_num - self.norm_head_start)
        )
        per_head_bytes = calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_head_num
        data_bytes = per_head_bytes * effective_norm_head_num
        weight_bytes = calc_tensor_size(self.input_tensor_info["norm_weight"])

        # For a standard C-contiguous 3D tensor [N, H, D], the slice
        # [:, start:end, :] is contiguous only when it covers all H heads (end==H).
        # This determines whether head_data will require .contiguous() + .copy_() in vendor_impl_run.
        need_copy = (effective_norm_head_num < self.total_head_num)

        if need_copy:
            # Actual memory traffic with .contiguous() + rms_norm + .copy_():
            #   contiguous(): read data (non-contig) + write data (contig)
            #   rms_norm():   read data (contig) + read weight + write data (output)
            #   copy_():      read data (output) + write data (non-contig)
            # Total: read = 3*data + weight, write = 3*data
            self.read_bytes = 3 * data_bytes + weight_bytes
            self.write_bytes = 3 * data_bytes
        else:
            # Only rms_norm: read data + read weight + write data
            self.read_bytes = data_bytes + weight_bytes
            self.write_bytes = data_bytes
        self.io_bytes = self.read_bytes + self.write_bytes

        self._run_func = self.vendor_impl_run


    def _head_rms_norm_eager(self, head_data, norm_weight):
        return torch.nn.functional.rms_norm(
            head_data,
            normalized_shape=head_data.shape[-1:],
            weight=norm_weight,
            eps=self.eps,
        )

    def vendor_impl_run(self, tensor_mapping):
        token_data = tensor_mapping["token_data"]
        norm_weight = tensor_mapping["norm_weight"]

        head_data = token_data[:, self.norm_head_start:self.norm_head_end, :]
        # Ensure the input is contiguous and the weight dtype matches the input dtype,
        # otherwise aten::rms_norm falls back to the non-fused composite path
        # (see aten/src/ATen/native/layer_norm.cpp: "Mismatch dtype ..."
        #  warning) and the optimized XPU fused kernel is NOT invoked.
        if head_data.is_contiguous():
            head_data_c = head_data
            need_copy = False
        else:            
            head_data_c = head_data.contiguous()
            need_copy = True
        if norm_weight.dtype != head_data_c.dtype:
            norm_weight = norm_weight.to(head_data_c.dtype)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

        normed_data = self._head_rms_norm_eager(head_data_c, norm_weight)
        
        # CRITICAL: Only modify tensor_mapping for write-back if the slice covers
        # the entire original tensor (i.e., it's a view onto the same backing storage).
        # Otherwise, copy the result back to the non-contiguous view.
        if need_copy:
            # The slice is a non-contiguous view. Copy the result back to keep
            # token_data's original shape and content intact.
            head_data.copy_(normed_data)
            return head_data 
        else:
            # The slice is already contiguous or covers the entire tensor.
            # Safely update token_data since normed_data is a view/replacement of head_data
            # which points to the same logical region.
            token_data = normed_data
            return token_data

