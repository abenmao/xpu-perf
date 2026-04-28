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
_SYCL_SO = _OP_DIR / "head_rms_norm_sycl.so"


try:
    _spec = importlib.util.spec_from_file_location("head_rms_norm_sycl", str(_SYCL_SO))
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Failed to create import spec for {_SYCL_SO}")
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

            self.read_bytes = data_bytes + weight_bytes
            self.write_bytes = data_bytes
            self.io_bytes = self.read_bytes + self.write_bytes

            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            token_data = tensor_mapping["token_data"]
            norm_weight = tensor_mapping["norm_weight"]

            if norm_weight.dtype != token_data.dtype:
                norm_weight = norm_weight.to(token_data.dtype)
            if not norm_weight.is_contiguous():
                norm_weight = norm_weight.contiguous()

            _sycl_ext.head_rms_norm_forward(
                token_data,
                norm_weight,
                int(self.norm_head_start),
                int(self.norm_head_num),
                float(self.eps),
            )
            return token_data

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL head_rms_norm extension: {e}")
