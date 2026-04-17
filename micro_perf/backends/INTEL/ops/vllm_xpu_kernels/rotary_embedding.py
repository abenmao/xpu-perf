import sys
import pathlib
from functools import partial
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import RotaryEmbeddingOp
from core.utils import OpTensorInfo, calc_tensor_size


def _build_cos_sin_cache(max_seq_len, rotary_dim, base=10000.0):
    """Build vllm-format cos_sin_cache: [max_seq_len, rotary_dim].
    
    Layout: [cos_0, cos_1, ..., cos_{d/2-1}, sin_0, sin_1, ..., sin_{d/2-1}]
    where d = rotary_dim.
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, rotary_dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)  # [max_seq_len, rotary_dim]
    return cache


try:
    import vllm_xpu_kernels._C

    @ProviderRegistry.register_vendor_impl("rotary_embedding", "vllm_xpu_kernels")
    class VLLMXPUKernelsRotaryEmbeddingOp(RotaryEmbeddingOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

        def _get_rotary_slice_bytes(self):
            dtype_size = torch.tensor([], dtype=self.torch_dtype).element_size()
            q_bytes = self.num_tokens * self.q_head_num * self.rope_dim * dtype_size
            k_bytes = self.num_tokens * self.kv_head_num * self.rope_dim * dtype_size
            return q_bytes, k_bytes

        def vendor_impl(self):
            """Override vendor_impl to set up vllm_xpu_kernels rotary embedding.
            
            vllm_xpu_kernels::rotary_embedding expects:
              - positions: [num_tokens] int64
              - query: [num_tokens, num_q_heads * head_dim]
              - key: [num_tokens, num_kv_heads * head_dim]  (optional)
              - head_size: int
              - cos_sin_cache: [max_pos, rotary_dim] interleaved cos/sin
              - is_neox: bool
            """
            super().vendor_impl()

            self.torch_dtype = getattr(torch, self.dtype)

            # Build per-token absolute position IDs (cache_len + local_offset)
            positions_list = []
            for batch_idx in range(self.batch_size):
                q_len = self.q_lens[batch_idx]
                cache_len = self.cache_lens[batch_idx]
                positions_list.extend(range(cache_len, cache_len + q_len))
            self._positions = torch.tensor(positions_list, dtype=torch.long)

            # Build interleaved cos/sin cache: [max_pos, rotary_dim]
            # vllm format: [cos_0, ..., cos_{d/2-1}, sin_0, ..., sin_{d/2-1}]
            cos_sin_cache = _build_cos_sin_cache(self.max_kv_len, self.rope_dim)

            # Override input tensors for vllm_xpu_kernels interface
            self.input_tensor_info = {}
            self.output_tensor_info = {}

            self.input_tensor_info["q"] = OpTensorInfo(
                shape=[self.num_tokens, self.q_head_num * self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
            self.input_tensor_info["k"] = OpTensorInfo(
                shape=[self.num_tokens, self.kv_head_num * self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
            self.input_tensor_info["positions"] = OpTensorInfo(
                shape=[self.num_tokens],
                dtype=torch.long,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: self._positions.to(
                    dtype=dtype, device=device
                ),
            )
            self.input_tensor_info["cos_sin_cache"] = OpTensorInfo(
                shape=[self.max_kv_len, self.rope_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: cos_sin_cache.to(
                    dtype=dtype, device=device
                ),
            )

            # calculator
            self.input_tensor_size = sum(
                [calc_tensor_size(info) for info in self.input_tensor_info.values()]
            )
            self.output_tensor_size = 0
            self.tensor_size = self.input_tensor_size

            q_rotary_bytes, k_rotary_bytes = self._get_rotary_slice_bytes()
            positions_bytes = calc_tensor_size(self.input_tensor_info["positions"])
            cache_bytes = calc_tensor_size(self.input_tensor_info["cos_sin_cache"])

            # The kernel only reads and writes the rotary slice of q/k.
            self.read_bytes = positions_bytes + cache_bytes + q_rotary_bytes + k_rotary_bytes
            self.write_bytes = q_rotary_bytes + k_rotary_bytes
            self.io_bytes = self.read_bytes + self.write_bytes

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=False,
            )
            self._run_func = self.rotary_embedding_run

        def rotary_embedding_run(self, tensor_mapping):
            positions = tensor_mapping["positions"]
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            cos_sin_cache = tensor_mapping["cos_sin_cache"]

            if self.rope_offset == 0:
                torch.ops._C.rotary_embedding(
                    positions, q, k, self.head_dim, cos_sin_cache, True
                )
                return q

            q_view = q.view(self.num_tokens, self.q_head_num, self.head_dim)
            k_view = k.view(self.num_tokens, self.kv_head_num, self.head_dim)

            dim_start = self.rope_offset
            dim_end = self.rope_offset + self.rope_dim

            q_rotary_view = q_view[:, :, dim_start:dim_end]
            k_rotary_view = k_view[:, :, dim_start:dim_end]

            torch.ops._C.rotary_embedding(
                positions,
                q_rotary_view,
                k_rotary_view,
                self.rope_dim,
                cos_sin_cache,
                True,
            )
            return q

except Exception:
    pass
