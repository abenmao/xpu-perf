from core.ops.llm_ops import MoeSoftmaxTopkOp
from core.op import ProviderRegistry
import os
import sys
import pathlib
import importlib.util

import torch
from functools import partial

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)


# Load the compiled SYCL extension
_SYCL_SO = os.path.join(os.path.dirname(__file__), "moe_softmax_topk_sycl.so")

try:
    _spec = importlib.util.spec_from_file_location("moe_softmax_topk_sycl", _SYCL_SO)
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl("moe_softmax_topk", "sycl_ext")
    class SYCLExtMoeSoftmaxTopKOp(MoeSoftmaxTopkOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def prepare(self):
            """Reuse base class tensor setup, override run func."""
            super().prepare()
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.sycl_moe_softmax_topk_run

        def sycl_moe_softmax_topk_run(self, tensor_mapping):
            gating_output = tensor_mapping["gating_output"]
            selected_experts = tensor_mapping["selected_experts"]
            moe_weights = tensor_mapping["moe_weights"]
            
            if self.compute_mode == "pre-softmax":
                _sycl_ext.moe_softmax_topk(
                    moe_weights, selected_experts, gating_output, 0, True , None
                )
                
                
            # topk --> softmax
            elif self.compute_mode == "post-softmax":
                _sycl_ext.moe_softmax_topk(
                    moe_weights, selected_experts, gating_output, 1, False ,None
                )

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL moe_softmax_topk extension: {e}")
