import sys
import pathlib
import importlib.util
from functools import partial

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import ScaleDynamicQuantOp


_OP_DIR = pathlib.Path(__file__).resolve().parent
_SYCL_SO = _OP_DIR / "scale_dynamic_quant_sycl.so"


try:
    _spec = importlib.util.spec_from_file_location(
        "scale_dynamic_quant_sycl", str(_SYCL_SO))
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Failed to create import spec for {_SYCL_SO}")
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl(
        "scale_dynamic_quant", "sycl_ext")
    class ScaleDynamicQuantSyclExtOp(ScaleDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def vendor_impl(self):
            super().vendor_impl()
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )

        def vendor_impl_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            smooth_scale = tensor_mapping["smooth_scale"]
            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            _sycl_ext.scale_dynamic_quant_forward(
                hidden_states, smooth_scale, quant_tokens, per_token_scale)
            return quant_tokens, per_token_scale

except Exception as e:
    import warnings
    warnings.warn(
        f"Failed to load SYCL scale_dynamic_quant extension: {e}")
