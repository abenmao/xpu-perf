import math
import pathlib
import sys
import warnings

import torch
from torch.nn.attention.bias import causal_lower_right

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.utils import OpTensorInfo, calc_tensor_size
from backends.INTEL.ops.torch.flash_attention import FlashAttentionXpuOp
from backends.INTEL.ops.sycl_ext import flash_attention as sycl_ext_flash_attention


try:
    _SYCL_EXT_SO = pathlib.Path(sycl_ext_flash_attention._SYCL_EXT_SO)
    if not _SYCL_EXT_SO.is_file():
        raise FileNotFoundError(
            f"sycl_ext flash attention shared object not found: {_SYCL_EXT_SO}"
        )

    @ProviderRegistry.register_vendor_impl("flash_attention", "sycl_ext")
    class SYCLExtFlashAttentionOp(FlashAttentionXpuOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl_ext"]
            self._provider = "sycl_ext"

        def prepare(self):
            super().prepare()

            if self.cache_dtype == "int8":
                scale_shape = [self.kv_head_num, self.head_dim]
                self.input_tensor_info["k_scale"] = OpTensorInfo(
                    shape=scale_shape,
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones,
                )
                self.input_tensor_info["v_scale"] = OpTensorInfo(
                    shape=scale_shape,
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones,
                )

                scale_bytes = (
                    calc_tensor_size(self.input_tensor_info["k_scale"])
                    + calc_tensor_size(self.input_tensor_info["v_scale"])
                )
                self.input_tensor_size += scale_bytes
                self.tensor_size += scale_bytes
                self.read_bytes += scale_bytes
                self.io_bytes = self.read_bytes + self.write_bytes

            self._run_func = self.flash_attention_run

        def _dequantize_cache(self, cache, scale):
            scale = scale.to(dtype=self.torch_dtype)
            return cache.to(dtype=self.torch_dtype) * scale.view(1, 1, self.kv_head_num, self.head_dim)

        def flash_attention_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k_cache = tensor_mapping["k_cache"]
            k_new = tensor_mapping["k_new"]
            v_cache = tensor_mapping["v_cache"]
            v_new = tensor_mapping["v_new"]

            if k_cache.dtype != q.dtype:
                if "k_scale" not in tensor_mapping or "v_scale" not in tensor_mapping:
                    raise ValueError(
                        "sycl_ext flash_attention requires k_scale/v_scale for quantized KV cache"
                    )
                k_cache = self._dequantize_cache(k_cache, tensor_mapping["k_scale"])
                v_cache = self._dequantize_cache(v_cache, tensor_mapping["v_scale"])

            k = torch.cat([k_cache, k_new], dim=1)
            v = torch.cat([v_cache, v_new], dim=1)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn_mask = None
            if self.prefix_len is not None:
                bias = causal_lower_right(self.q_seq_len, self.kv_seq_len)
                attn_mask = bias._materialize(q.device)

            out = sycl_ext_flash_attention.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=self.is_causal if attn_mask is None else False,
                scale=float(1.0 / math.sqrt(self.head_dim)),
                enable_gqa=True,
                output_dtype="float32",
            )

            tensor_mapping["out"] = out
            return out

except Exception as e:
    warnings.warn(f"Failed to register sycl_ext flash_attention provider: {e}")
