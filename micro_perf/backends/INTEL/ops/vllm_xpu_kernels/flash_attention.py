import sys
import math
import pathlib
from functools import partial
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp


try:
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func, FA2_AVAILABLE
    if not FA2_AVAILABLE:
        raise ImportError("vllm_xpu_kernels FA2 not available")

    @ProviderRegistry.register_vendor_impl("flash_attention", "vllm_xpu_kernels")
    class VLLMXPUKernelsFlashAttentionOp(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

            if self.attn_mode == "prefill":
                self._prefill_init()
            elif self.attn_mode == "decode":
                self._decode_init()

        def _check_dtypes(self):
            # Only the bf16 path is wired up here; let any other dtype
            # combination be reported as an error early.
            if not (
                self.dtype == "bfloat16"
                and self.pv_compute_dtype == "bfloat16"
                and self.cache_dtype == "bfloat16"
            ):
                raise ValueError(
                    f"VLLMXPUKernelsFlashAttentionOp only supports bfloat16, "
                    f"got dtype={self.dtype}, pv_compute_dtype={self.pv_compute_dtype}, "
                    f"cache_dtype={self.cache_dtype}"
                )
            if self.cache_type not in ("linear", "paged"):
                raise ValueError(
                    f"VLLMXPUKernelsFlashAttentionOp only supports linear/paged "
                    f"cache, got {self.cache_type}"
                )

        def _prefill_init(self):
            self._check_dtypes()
            # vllm_xpu_kernels v0.1.5 only exposes a single Python entry
            # `flash_attn_varlen_func`; the C++ side dispatches to the
            # chunk_prefill kernel when max_seqlen_q > 1.
            self._run_func = self._prefill_run

        def _decode_init(self):
            self._check_dtypes()
            # Same single Python entry; C++ dispatches to the paged_decode
            # kernel when max_seqlen_q == 1.
            self._run_func = self._decode_run

        def _build_common_kwargs(self, tensor_mapping):
            cu_seqlens_q = tensor_mapping["accum_q_lens"].to(torch.int32)
            return dict(
                max_seqlen_q=max(self.q_lens),
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_k=max(self.kv_lens),
                causal=self.is_causal,
            )

        def _build_kv_args(self, tensor_mapping):
            """Convert the base op's KV cache layout into what
            flash_attn_varlen_func expects, returning (k, v, extra_kwargs).
            """
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]

            if self.cache_type == "linear":
                # Base layout [B, H, S, D] -> varlen [total_kv_tokens, H, D].
                B, H, S, D = k_cache.shape
                k = k_cache.permute(0, 2, 1, 3).reshape(B * S, H, D)
                v = v_cache.permute(0, 2, 1, 3).reshape(B * S, H, D)
                extra = dict(
                    cu_seqlens_k=tensor_mapping["accum_kv_lens"].to(torch.int32),
                )
            else:
                # Paged. Base layout [num_blocks, H, block_size, D] -> view
                # [num_blocks, block_size, H, D]. permute keeps stride(-1)=1,
                # so vllm's maybe_contiguous won't copy.
                k = k_cache.permute(0, 2, 1, 3)
                v = v_cache.permute(0, 2, 1, 3)
                # block_table padded with -1 by the base op; clamp so the
                # kernel never dereferences a negative index. seqused_k
                # bounds the real reads.
                block_table = tensor_mapping["block_table"].to(torch.int32).clamp(min=0)
                extra = dict(
                    seqused_k=tensor_mapping["kv_lens"].to(torch.int32),
                    block_table=block_table,
                )
            return k, v, extra

        def _prefill_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k, v, kv_extra = self._build_kv_args(tensor_mapping)
            common = self._build_common_kwargs(tensor_mapping)
            return flash_attn_varlen_func(q, k, v, **common, **kv_extra)

        def _decode_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k, v, kv_extra = self._build_kv_args(tensor_mapping)
            common = self._build_common_kwargs(tensor_mapping)
            return flash_attn_varlen_func(q, k, v, **common, **kv_extra)

except Exception:
    pass
