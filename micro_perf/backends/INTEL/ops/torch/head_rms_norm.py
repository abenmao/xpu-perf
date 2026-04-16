import sys
import pathlib
import traceback

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)
from core.utils import calc_tensor_size
from core.ops.llm_ops import HeadRMSNormOp as BaseHeadRMSNormOp

class HeadRMSNormTorchOp(BaseHeadRMSNormOp):

    def vendor_impl(self):
        super().vendor_impl()

        # Keep effective byte accounting consistent with edge case handling.
        effective_norm_head_num = max(
            0,
            min(self.norm_head_num, self.total_head_num - self.norm_head_start)
        )
        per_head_bytes = calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_head_num
        self.read_bytes = per_head_bytes * effective_norm_head_num
        self.write_bytes = self.read_bytes
        self.read_bytes += calc_tensor_size(self.input_tensor_info["norm_weight"])
        self.io_bytes = self.read_bytes + self.write_bytes

        self._head_rms_norm_compiled = self._head_rms_norm_eager
        if hasattr(torch, "compile"):
            try:
                self._head_rms_norm_compiled = torch.compile(
                    self._head_rms_norm_eager,
                    fullgraph=False,
                    dynamic=False,
                )
            except Exception:
                self._head_rms_norm_compiled = self._head_rms_norm_eager
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
        normed_data = self._head_rms_norm_compiled(head_data.contiguous(), norm_weight)
        head_data.copy_(normed_data)
        return token_data


OP_MAPPING = {"torch": HeadRMSNormTorchOp}
