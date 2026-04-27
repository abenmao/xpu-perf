import os
import sys
import pathlib
import importlib.util

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.utils import calc_tensor_size
from core.ops.llm_ops import HeadRMSNormOp as BaseHeadRMSNormOp


_OP_DIR = pathlib.Path(__file__).resolve().parent
_SYCL_SO = _OP_DIR / "rms_norm_sycl.so"


try:
    _spec = importlib.util.spec_from_file_location("rms_norm_sycl", str(_SYCL_SO))
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl("head_rms_norm", "sycl_ext")
    class HeadRMSNormSyclExtOp(BaseHeadRMSNormOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def vendor_impl(self):
            super().vendor_impl()

            effective_norm_head_num = max(
                0,
                min(self.norm_head_num, self.total_head_num - self.norm_head_start)
            )
            per_head_bytes = calc_tensor_size(self.input_tensor_info["token_data"]) / self.total_head_num
            data_bytes = per_head_bytes * effective_norm_head_num
            weight_bytes = calc_tensor_size(self.input_tensor_info["norm_weight"])

            # Same accounting model as torch provider: additional contiguous/copy traffic
            # is only required when the selected head range is a strict subset.
            need_copy = (effective_norm_head_num < self.total_head_num)

            if need_copy:
                self.read_bytes = 3 * data_bytes + weight_bytes
                self.write_bytes = 3 * data_bytes
            else:
                self.read_bytes = data_bytes + weight_bytes
                self.write_bytes = data_bytes
            self.io_bytes = self.read_bytes + self.write_bytes

            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            token_data = tensor_mapping["token_data"]
            norm_weight = tensor_mapping["norm_weight"]

            head_data = token_data[:, self.norm_head_start:self.norm_head_end, :]
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

            normed_data = _sycl_ext.rms_norm_forward(head_data_c, norm_weight, float(self.eps))

            if need_copy:
                head_data.copy_(normed_data)
                return token_data
            else:
                token_data = normed_data
                return token_data

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL head_rms_norm extension: {e}")
