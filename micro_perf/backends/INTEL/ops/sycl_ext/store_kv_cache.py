"""
SYCL extension provider for store_kv_cache.

Uses custom fused SYCL kernels:
  - int8: bf16 -> fp32 -> scale -> clamp -> round -> int8 in ONE kernel
  - bf16: bf16 -> bf16 permute+copy in ONE kernel

Same linear cache layout as torch path, so we reuse base tensor setup
and only override the run function.
"""
from core.ops.llm_ops import StoreKVCacheOp
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
_SYCL_SO = os.path.join(os.path.dirname(__file__), "store_kv_cache_sycl.so")

try:
    _spec = importlib.util.spec_from_file_location("store_kv_cache_sycl", _SYCL_SO)
    _sycl_ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_sycl_ext)

    @ProviderRegistry.register_vendor_impl("store_kv_cache", "sycl_ext")
    class SYCLExtStoreKVCacheOp(StoreKVCacheOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]

        def vendor_impl(self):
            """Reuse base class tensor setup, override run func."""
            super().vendor_impl()
            self._run_func = self.sycl_store_kv_cache_run

        def sycl_store_kv_cache_run(self, tensor_mapping):
            packed_qkv = tensor_mapping["packed_qkv"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            q_len = self.q_lens[0]
            cache_len = self.cache_lens[0]
            bs = self.batch_size

            k_head_start = self.q_head_num

            if self.use_quant:
                k_scale = tensor_mapping["k_scale"]
                v_scale = tensor_mapping["v_scale"]
                _sycl_ext.store_kv_cache_int8(
                    packed_qkv, k_cache, v_cache,
                    k_scale, v_scale,
                    k_head_start, self.kv_head_num,
                    bs, q_len, cache_len
                )
            else:
                _sycl_ext.store_kv_cache_bf16(
                    packed_qkv, k_cache, v_cache,
                    k_head_start, self.kv_head_num,
                    bs, q_len, cache_len
                )

            return k_cache, v_cache

except Exception as e:
    import warnings
    warnings.warn(f"Failed to load SYCL store_kv_cache extension: {e}")
