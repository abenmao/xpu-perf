"""
MLA attention providers using the embedded flashmla SYCL extension.

Loads flash_attn_xpu.abi3.so from the co-located flashmla/ directory and
provides 4 MLA operator implementations:
  - Dense Decode
  - Dense Prefill
  - Sparse Decode
  - Sparse Prefill
"""
import os
import sys
import pathlib
import importlib
import importlib.util

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import (
    MLADenseDecodeOp,
    MLADensePrefillOp,
    MLASparseDecodeOp,
    MLASparsePrefillOp,
)

# ── Load local flashmla extension ──────────────────────────────────
_FLASHMLA_DIR = os.path.join(os.path.dirname(__file__), "flashmla")
_FLASHMLA_SO = os.path.join(_FLASHMLA_DIR, "flash_attn_xpu.abi3.so")

try:
    # Load the .so from our embedded flashmla/ directory
    _spec = importlib.util.spec_from_file_location("flash_attn_xpu", _FLASHMLA_SO)
    _ext = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ext)
    sys.modules["flash_attn_xpu"] = _ext

    # Now import the Python interface from the same directory
    if _FLASHMLA_DIR not in sys.path:
        sys.path.insert(0, _FLASHMLA_DIR)
    from flash_attn_interface import (
        flash_mla_with_kvcache,
        flash_mla_sparse_fwd,
        flash_mla_dense_prefill_fwd,
        get_mla_metadata,
        FlashMLASchedMeta,
    )

    # ─────────────────────────────────────────
    # MLA Dense Decode Provider
    # ─────────────────────────────────────────
    @ProviderRegistry.register_vendor_impl("mla_dense_decode", "flashmla_xpu")
    class FlashMLADenseDecodeProvider(MLADenseDecodeOp):

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["flashmla_xpu"]
            self._sched_meta, _ = get_mla_metadata()
            self._run_func = self._run_dense_decode

        def _run_dense_decode(self, tensor_mapping):
            q = tensor_mapping["q"]
            kv_cache = tensor_mapping["kv_cache"]
            block_table = tensor_mapping["block_table"]
            cache_seqlens = tensor_mapping["cache_seqlens"]

            out, lse = flash_mla_with_kvcache(
                q,
                kv_cache,
                block_table,
                cache_seqlens,
                self.d_v,
                self._sched_meta,
                None,
                softmax_scale=self.softmax_scale,
                causal=self.is_causal,
            )
            return out


    # ─────────────────────────────────────────
    # MLA Dense Prefill Provider
    # ─────────────────────────────────────────
    @ProviderRegistry.register_vendor_impl("mla_dense_prefill", "flashmla_xpu")
    class FlashMLADensePrefillProvider(MLADensePrefillOp):

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["flashmla_xpu"]
            self._run_func = self._run_dense_prefill

        def _run_dense_prefill(self, tensor_mapping):
            q = tensor_mapping["q"]
            kv = tensor_mapping["kv"]
            cu_seqlens_q = tensor_mapping["cu_seqlens_q"]
            cu_seqlens_k = tensor_mapping["cu_seqlens_k"]

            out, max_logits, lse = flash_mla_dense_prefill_fwd(
                q, kv,
                cu_seqlens_q, cu_seqlens_k,
                sm_scale=self.softmax_scale,
                d_v=self.d_v,
                is_causal=self.is_causal,
            )
            return out


    # ─────────────────────────────────────────
    # MLA Sparse Decode Provider
    # ─────────────────────────────────────────
    @ProviderRegistry.register_vendor_impl("mla_sparse_decode", "flashmla_xpu")
    class FlashMLASparseDecodeProvider(MLASparseDecodeOp):

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["flashmla_xpu"]
            self._sched_meta, _ = get_mla_metadata()
            self._run_func = self._run_sparse_decode

        def _run_sparse_decode(self, tensor_mapping):
            q = tensor_mapping["q"]
            kv_cache = tensor_mapping["kv_cache"]
            indices = tensor_mapping["indices"]

            topk_length = tensor_mapping.get("topk_length", None)
            attn_sink = tensor_mapping.get("attn_sink", None)

            # Dual-scope
            extra_kv_cache = tensor_mapping.get("extra_kv_cache", None)
            extra_indices = tensor_mapping.get("extra_indices", None)
            extra_topk_length = tensor_mapping.get("extra_topk_length", None)

            out, lse = flash_mla_with_kvcache(
                q,
                kv_cache,
                None,  # no block_table for sparse
                None,  # no cache_seqlens for sparse
                self.d_v,
                self._sched_meta,
                None,
                softmax_scale=self.softmax_scale,
                causal=False,
                is_fp8_kvcache=self.is_fp8,
                indices=indices,
                attn_sink=attn_sink,
                extra_k_cache=extra_kv_cache,
                extra_indices_in_kvcache=extra_indices,
                topk_length=topk_length,
                extra_topk_length=extra_topk_length,
            )
            return out


    # ─────────────────────────────────────────
    # MLA Sparse Prefill Provider
    # ─────────────────────────────────────────
    @ProviderRegistry.register_vendor_impl("mla_sparse_prefill", "flashmla_xpu")
    class FlashMLASparsePrefillProvider(MLASparsePrefillOp):

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["flashmla_xpu"]
            self._run_func = self._run_sparse_prefill

        def _run_sparse_prefill(self, tensor_mapping):
            q = tensor_mapping["q"]
            kv = tensor_mapping["kv"]
            indices = tensor_mapping["indices"]

            topk_length = tensor_mapping.get("topk_length", None)
            attn_sink = tensor_mapping.get("attn_sink", None)

            out, max_logits, lse = flash_mla_sparse_fwd(
                q, kv, indices,
                sm_scale=self.softmax_scale,
                d_v=self.d_v,
                attn_sink=attn_sink,
                topk_length=topk_length,
            )
            return out

except Exception as e:
    import traceback
    print(f"[FlashMLA Provider] Failed to register: {e}")
    traceback.print_exc()
