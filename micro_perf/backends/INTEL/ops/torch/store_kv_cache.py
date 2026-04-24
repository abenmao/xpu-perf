import sys
import pathlib
from functools import partial

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.llm_ops import StoreKVCacheOp as BaseStoreKVCacheOp
from core.op import ProviderRegistry
from core.utils import static_quant
import torch


@ProviderRegistry.register_vendor_impl("store_kv_cache", "torch")
class StoreKVCacheOp(BaseStoreKVCacheOp):
    """INTEL vendor override for store_kv_cache.

    Optimizes the base implementation by replacing the per-batch Python
    for-loop with vectorized tensor operations across all batches.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def vendor_impl(self):
        super().vendor_impl()
        self._run_func = self.vectorized_store_run

    def vectorized_store_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        k_scale = tensor_mapping.get("k_scale")
        v_scale = tensor_mapping.get("v_scale")

        if self.cache_type == "paged":
            return self._paged_store(packed_qkv, k_cache, v_cache, k_scale, v_scale)

        if self.cache_type == "linear":
            q_len = self.q_lens[0]
            cache_len = self.cache_lens[0]

            # For quant + large prefill, per-batch loop avoids huge float32 intermediates
            if self.use_quant and q_len > 32:
                return self._per_batch_store(packed_qkv, k_cache, v_cache, k_scale, v_scale)

            k_head_start = self.q_head_num
            k_head_end = self.q_head_num + self.kv_head_num
            v_head_start = self.q_head_num + self.kv_head_num
            v_head_end = self.q_head_num + self.kv_head_num * 2

            # packed_qkv: [num_tokens, total_head_num, head_dim]
            # Reshape to [batch_size, q_len, head_num, head_dim]
            src_all = packed_qkv.view(self.batch_size, q_len, self.total_head_num, self.head_dim)

            # Extract K and V: [batch_size, q_len, kv_head_num, head_dim]
            src_k = src_all[:, :, k_head_start:k_head_end, :]
            src_v = src_all[:, :, v_head_start:v_head_end, :]

            # Transpose to cache layout: [batch_size, kv_head_num, q_len, head_dim]
            # No .contiguous() needed — copy_() handles non-contiguous sources
            src_k = src_k.transpose(1, 2)
            src_v = src_v.transpose(1, 2)

            cache_end = cache_len + q_len

            if self.use_quant:
                # Only reached for small q_len (decode), safe to materialize
                scale_k = k_scale.view(1, self.kv_head_num, 1, self.head_dim)
                scale_v = v_scale.view(1, self.kv_head_num, 1, self.head_dim)

                if self.cache_torch_dtype == torch.int8:
                    max_val = 127.0
                else:
                    raise ValueError(f"Unsupported cache dtype: {self.cache_torch_dtype}")

                k_quant = src_k.float().mul_(scale_k).clamp_(-max_val, max_val)
                v_quant = src_v.float().mul_(scale_v).clamp_(-max_val, max_val)

                if self.cache_torch_dtype == torch.int8:
                    k_quant.round_()
                    v_quant.round_()

                k_cache[:, :, cache_len:cache_end, :].copy_(k_quant.to(self.cache_torch_dtype))
                v_cache[:, :, cache_len:cache_end, :].copy_(v_quant.to(self.cache_torch_dtype))
            else:
                k_cache[:, :, cache_len:cache_end, :].copy_(src_k)
                v_cache[:, :, cache_len:cache_end, :].copy_(src_v)

        return k_cache, v_cache

    def _per_batch_store(self, packed_qkv, k_cache, v_cache, k_scale, v_scale):
        """Per-batch loop for quant prefill — avoids huge float32 intermediates."""
        k_head_start = self.q_head_num
        k_head_end = self.q_head_num + self.kv_head_num
        v_head_start = self.q_head_num + self.kv_head_num
        v_head_end = self.q_head_num + self.kv_head_num * 2

        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            q_offset = self.accum_q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]
            cache_end = cache_len + q_len

            # [q_len, kv_head_num, head_dim]
            src_k = packed_qkv[q_offset:q_offset + q_len, k_head_start:k_head_end, :]
            src_v = packed_qkv[q_offset:q_offset + q_len, v_head_start:v_head_end, :]

            # Quantize and transpose to [kv_head_num, q_len, head_dim]
            k_cache[batch_idx, :, cache_len:cache_end, :].copy_(
                static_quant(src_k, k_scale, self.cache_torch_dtype).transpose(0, 1))
            v_cache[batch_idx, :, cache_len:cache_end, :].copy_(
                static_quant(src_v, v_scale, self.cache_torch_dtype).transpose(0, 1))

        return k_cache, v_cache

    def _paged_store(self, packed_qkv, k_cache, v_cache, k_scale, v_scale):
        """Paged cache store: write tokens into physical blocks via block_table."""
        k_head_start = self.q_head_num
        v_head_start = self.q_head_num + self.kv_head_num

        if self.use_quant:
            scale_k = k_scale.view(self.kv_head_num, self.head_dim)
            scale_v = v_scale.view(self.kv_head_num, self.head_dim)
            if self.cache_torch_dtype == torch.int8:
                max_val = 127.0
            else:
                raise ValueError(f"Unsupported cache dtype: {self.cache_torch_dtype}")

        for batch_idx in range(self.batch_size):
            q_len = self.q_lens[batch_idx]
            q_offset = self.accum_q_lens[batch_idx]
            cache_len = self.cache_lens[batch_idx]

            for token_idx in range(q_len):
                global_pos = cache_len + token_idx
                block_idx = global_pos // self.block_size
                block_offset = global_pos % self.block_size
                physical_block = self.block_table[batch_idx][block_idx]

                src_token = packed_qkv[q_offset + token_idx]
                src_k = src_token[k_head_start:k_head_start + self.kv_head_num, :]
                src_v = src_token[v_head_start:v_head_start + self.kv_head_num, :]

                if self.use_quant:
                    k_q = src_k.float().mul(scale_k).clamp_(-max_val, max_val).round_().to(self.cache_torch_dtype)
                    v_q = src_v.float().mul(scale_v).clamp_(-max_val, max_val).round_().to(self.cache_torch_dtype)
                    k_cache[physical_block, :, block_offset, :] = k_q
                    v_cache[physical_block, :, block_offset, :] = v_q
                else:
                    k_cache[physical_block, :, block_offset, :] = src_k
                    v_cache[physical_block, :, block_offset, :] = src_v

        return k_cache, v_cache


OP_MAPPING = {"torch": StoreKVCacheOp}
