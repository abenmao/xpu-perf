"""
SYCL extension provider for dequant_kv_cache.

Fuses int8→bf16 cast + scale multiply into a single kernel,
eliminating the intermediate bf16 tensor and saving ~2x memory traffic
compared to the two-step torch path (copy_ + mul_).
"""
from backends.INTEL.ops.torch.dequant_kv_cache import DequantKVCacheOp
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
_SYCL_SO = os.path.join(os.path.dirname(__file__), "dequant_kv_cache_sycl.so")

try:
    _spec = importlib.util.spec_from_file_location("dequant_kv_cache_sycl", _SYCL_SO)
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl("dequant_kv_cache", "sycl_ext")
    class SYCLExtDequantKVCacheOp(DequantKVCacheOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def prepare(self):
            """Reuse base class tensor setup, override run func."""
            super().prepare()
            self._run_func = self.sycl_dequant_kv_cache_run

        def sycl_dequant_kv_cache_run(self, tensor_mapping):
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            dequant_k_cache = tensor_mapping["dequant_k_cache"]
            dequant_v_cache = tensor_mapping["dequant_v_cache"]
            k_scale = tensor_mapping["k_scale"]
            v_scale = tensor_mapping["v_scale"]

            bs = self.batch_size
            kv_len = self.kv_lens[0]

            if self.dtype in ("float8", "float8_e4m3"):
                # FP8 dequant: scale as bf16 (matching torch's .to(bfloat16) truncation)
                k_scale_bf16 = k_scale.to(torch.bfloat16)
                v_scale_bf16 = v_scale.to(torch.bfloat16)

                _sycl_ext.dequant_kv_cache_fp8(
                    k_cache, v_cache,
                    dequant_k_cache, dequant_v_cache,
                    k_scale_bf16, v_scale_bf16,
                    bs, kv_len
                )
            else:
                # INT8 dequant
                k_scale_bf16 = k_scale.to(torch.bfloat16)
                v_scale_bf16 = v_scale.to(torch.bfloat16)

                _sycl_ext.dequant_kv_cache(
                    k_cache, v_cache,
                    dequant_k_cache, dequant_v_cache,
                    k_scale_bf16, v_scale_bf16,
                    bs, kv_len
                )

            return dequant_k_cache, dequant_v_cache

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL dequant_kv_cache extension: {e}")
