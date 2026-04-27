import torch
from core.op import ProviderRegistry
from core.ops.llm_ops import RotaryEmbeddingOp

# Default cache_size_limit=8 can be too small for all test cases
# Increase it to 64 to avoid unnecessary recompilations in tests.
torch._dynamo.config.cache_size_limit = max(
    getattr(torch._dynamo.config, "cache_size_limit", 8), 64
)

def _rope_inplace(qk, cos_h, sin_h):
    x1, x2 = qk.chunk(2, dim=-1)
    out1 = x1 * cos_h - x2 * sin_h
    out2 = x2 * cos_h + x1 * sin_h
    qk.copy_(torch.cat([out1, out2], dim=-1))

@ProviderRegistry.register_vendor_impl("rotary_embedding", "torch_compiled")
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
        self._pos_ids_cpu = torch.tensor(pos_ids, dtype=torch.int64)
        self._cos_h = None
        self._sin_h = None
        self._rope_fn = torch.compile(_rope_inplace, dynamic=False)

    def vendor_impl_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        cos = tensor_mapping["cos"]
        sin = tensor_mapping["sin"]

        if self._cos_h is None:
            half = self.rope_dim // 2
            pos_ids = self._pos_ids_cpu.to(cos.device, non_blocking=True)
            self._cos_h = cos.index_select(0, pos_ids)[:, :half].unsqueeze(1).contiguous()
            self._sin_h = sin.index_select(0, pos_ids)[:, :half].unsqueeze(1).contiguous()

        qk_end = self.q_head_num + self.kv_head_num
        rope_end = self.rope_offset + self.rope_dim
        qk_view = packed_qkv[:, :qk_end, self.rope_offset:rope_end]
        self._rope_fn(qk_view, self._cos_h, self._sin_h)
        return packed_qkv
