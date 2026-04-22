import torch

from core.op import ProviderRegistry
from core.ops.llm_ops import RotaryEmbeddingOp

try:
    import torch._dynamo as _dynamo
    _dynamo.config.cache_size_limit = max(
        getattr(_dynamo.config, "cache_size_limit", 8), 128
    )
except Exception:
    pass


def _rope_inplace(qk, cos_h, sin_h):
    x1, x2 = qk.chunk(2, dim=-1)
    out1 = x1 * cos_h - x2 * sin_h
    out2 = x2 * cos_h + x1 * sin_h
    qk.copy_(torch.cat([out1, out2], dim=-1))


@ProviderRegistry.register_vendor_impl("rotary_embedding", "torch")
class RotaryEmbeddingTorchCompiledOp(RotaryEmbeddingOp):
    def vendor_impl(self):
        super().vendor_impl()

        pos_ids = [
            pos
            for b in range(self.batch_size)
            for pos in range(
                int(self.cache_lens[b]),
                int(self.cache_lens[b]) + int(self.q_lens[b]),
            )
        ]
        assert len(pos_ids) == self.num_tokens
        self._pos_ids_cpu = torch.tensor(pos_ids, dtype=torch.int64)
        self._pos_ids_dev = None

        self._qk_head_end = self.q_head_num + self.kv_head_num
        self._half = self.rope_dim // 2
        self._rope_end = self.rope_offset + self.rope_dim

        self._cossin_cache = {}

        if hasattr(torch, "compile"):
            try:
                self._rope_fn = torch.compile(
                    _rope_inplace, fullgraph=False, dynamic=False
                )
            except Exception:
                self._rope_fn = _rope_inplace
        else:
            self._rope_fn = _rope_inplace

        self._run_func = self.vendor_impl_run

    def _get_cached_cossin(self, cos, sin):
        key = (id(cos), id(sin))
        cached = self._cossin_cache.get(key)
        if cached is not None:
            return cached

        if self._pos_ids_dev is None or self._pos_ids_dev.device != cos.device:
            self._pos_ids_dev = self._pos_ids_cpu.to(
                device=cos.device, non_blocking=True
            )
        cos_h = cos.index_select(0, self._pos_ids_dev)[:, :self._half].unsqueeze(1).contiguous()
        sin_h = sin.index_select(0, self._pos_ids_dev)[:, :self._half].unsqueeze(1).contiguous()
        self._cossin_cache[key] = (cos_h, sin_h)
        return cos_h, sin_h

    def vendor_impl_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        cos = tensor_mapping["cos"]
        sin = tensor_mapping["sin"]

        cos_h, sin_h = self._get_cached_cossin(cos, sin)

        qk_view = packed_qkv[:, :self._qk_head_end, self.rope_offset:self._rope_end]

        self._rope_fn(qk_view, cos_h, sin_h)
        return packed_qkv
