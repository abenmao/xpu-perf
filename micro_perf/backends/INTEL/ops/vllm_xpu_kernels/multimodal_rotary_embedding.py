import sys
import pathlib
from functools import partial
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import MultimodalRotaryEmbeddingOp
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
    import vllm_xpu_kernels._xpu_C

    @ProviderRegistry.register_vendor_impl("multimodal_rotary_embedding", "vllm_xpu_kernels")
    class VLLMXPUKernelsMultimodalRotaryEmbeddingOp(MultimodalRotaryEmbeddingOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

        def vendor_impl(self):
            """Override vendor_impl to set up vllm_xpu_kernels multimodal rotary embedding.
            
            vllm_xpu_kernels::multimodal_rotary_embedding expects:
              - positions: [num_mrope_sections, num_tokens] int64
              - query: [num_tokens, num_q_heads, head_dim]
              - key: [num_tokens, num_kv_heads, head_dim] (optional)
              - head_size: int
              - cos_sin_cache: [max_pos, rotary_dim] interleaved cos/sin
              - is_neox: bool
              - mrope_section: list[int]
            """
            super().vendor_impl()

            self.torch_dtype = getattr(torch, self.dtype)

            # Build per-token absolute position IDs per section
            positions_list = []
            for _ in range(self.num_mrope_sections):
                sec_positions = []
                for batch_idx in range(self.batch_size):
                    q_len = self.q_lens[batch_idx]
                    cache_len = self.cache_lens[batch_idx]
                    sec_positions.extend(range(cache_len, cache_len + q_len))
                positions_list.append(sec_positions)
            self._positions = torch.tensor(positions_list, dtype=torch.long)

            # Build interleaved cos/sin cache: [max_pos, rotary_dim]
            cos_sin_cache = _build_cos_sin_cache(self.max_kv_len, self.rope_dim)

            # Override input tensors for vllm_xpu_kernels interface
            self.input_tensor_info = {}
            self.output_tensor_info = {}

            self.input_tensor_info["q"] = OpTensorInfo(
                shape=[self.num_tokens, self.q_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
            self.input_tensor_info["k"] = OpTensorInfo(
                shape=[self.num_tokens, self.kv_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
            self.input_tensor_info["positions"] = OpTensorInfo(
                shape=[self.num_mrope_sections, self.num_tokens],
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

            self.read_bytes = self.input_tensor_size
            self.write_bytes = (
                calc_tensor_size(self.input_tensor_info["q"])
                + calc_tensor_size(self.input_tensor_info["k"])
            )
            self.io_bytes = self.read_bytes + self.write_bytes

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=False,
            )
            self._run_func = self.multimodal_rotary_embedding_run

        def multimodal_rotary_embedding_run(self, tensor_mapping):
            positions = tensor_mapping["positions"]
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            cos_sin_cache = tensor_mapping["cos_sin_cache"]

            torch.ops._xpu_C.multimodal_rotary_embedding(
                positions, q, k, self.head_dim, cos_sin_cache, True,
                self.mrope_section
            )
            return q

except Exception:
    pass
