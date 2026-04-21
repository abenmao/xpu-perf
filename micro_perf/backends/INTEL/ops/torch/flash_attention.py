import sys
import math
import pathlib
from functools import partial
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

sys.path.insert(
    0, 
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size

OP_MAPPING = {}


class FlashAttentionXpuOp(FlashAttentionOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["sdpa_flash_attention"]

        self.q_seq_len = args_dict.get("q_len", 1024)
        self.cache_len = args_dict.get("cache_len", 0)
        self.kv_seq_len = self.cache_len + self.q_seq_len
        self.head_dim = args_dict.get("head_dim", 128)
        self.num_head_q = args_dict.get("q_head_num", 96)
        self.num_head_kv = args_dict.get("kv_head_num", 8)
        self.prefix_len = self.cache_len if self.q_seq_len > 1 and self.cache_len != 0 else None
        self.scale = float(1.0 / math.sqrt(self.head_dim))
        self.is_causal = self.q_seq_len == self.kv_seq_len

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm"]:
            raise NotImplementedError

        self.dtype = self.args_dict.get("dtype", "float16")
        self.cache_dtype = self.args_dict["cache_dtype"]
        if self.dtype not in ["float16", "bfloat16"]:
            raise NotImplementedError
        self.torch_dtype = getattr(torch, self.dtype)
        self.cache_torch_dtype = getattr(torch, self.cache_dtype) if self.cache_dtype != "int8" else torch.int8

        self.q_head_num = self.args_dict["q_head_num"]
        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]
        self.batch_size = self.args_dict["batch_size"]
        self.q_seq_len = self.args_dict["q_len"]
        self.cache_len = self.args_dict["cache_len"]
        self.kv_seq_len = self.cache_len + self.q_seq_len
        self.is_causal = self.q_seq_len == self.kv_seq_len
        self.prefix_len = self.cache_len if self.q_seq_len > 1 and self.cache_len != 0 else None
        self.softmax_scale = self.head_dim ** (-0.5)

        self.input_tensor_info = {
            "q": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "k_cache": OpTensorInfo(
                shape=[self.batch_size, self.cache_len, self.kv_head_num, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "k_new": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.kv_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "v_cache": OpTensorInfo(
                shape=[self.batch_size, self.cache_len, self.kv_head_num, self.head_dim],
                dtype=self.cache_torch_dtype,
                device=self.backend.get_torch_device_name()
            ),
            "v_new": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.kv_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )
        }

        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name()
            )
        }

        self.input_tensor_size = sum([
            calc_tensor_size(info) for info in self.input_tensor_info.values()
        ])
        self.output_tensor_size = sum([
            calc_tensor_size(info) for info in self.output_tensor_info.values()
        ])
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes

        self.calc_flops = 0
        for idx in range(self.batch_size):
            q_len = self.q_seq_len
            kv_len = self.kv_seq_len

            gemm_flops = self.q_head_num * q_len * self.head_dim * kv_len * 2

            if self.prefix_len is not None:
                decode_kv_len = kv_len - self.prefix_len
                decode_q_len = min(q_len, decode_kv_len)
                total_valid_flops = q_len * self.prefix_len + (decode_q_len * decode_kv_len - decode_q_len * decode_q_len / 2)
                flops_ratio = total_valid_flops / (q_len * kv_len) if (q_len * kv_len) > 0 else 1.
            elif self.is_causal:
                flops_ratio = (q_len * kv_len - q_len * q_len / 2) / (q_len * kv_len)
            else:
                flops_ratio = 1.

            # QK^T + PV = 2x gemm
            self.calc_flops += gemm_flops * 2 * flops_ratio

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )

        self._run_func = self.flash_attention_run

    def flash_attention_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        k_cache = tensor_mapping["k_cache"]
        k_new = tensor_mapping["k_new"]
        v_cache = tensor_mapping["v_cache"]
        v_new = tensor_mapping["v_new"]

        k = torch.cat([k_cache, k_new], dim=1)
        v = torch.cat([v_cache, v_new], dim=1)

        q = q.transpose(1, 2)  # [batch, q_head_num, q_seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, kv_head_num, kv_seq_len, head_dim]
        v = v.transpose(1, 2)

        attn_mask = None
        if self.prefix_len is not None:
            attn_mask = causal_lower_right(self.q_seq_len, self.kv_seq_len)

        with sdpa_kernel(backends=[SDPBackend.OVERRIDEABLE]):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                is_causal=self.is_causal if attn_mask is None else False,
                scale=self.scale,
                enable_gqa=True
            )

        tensor_mapping["out"] = out
        return out


OP_MAPPING["xpu_flash_attention"] = FlashAttentionXpuOp
OP_MAPPING["torch"] = FlashAttentionXpuOp
