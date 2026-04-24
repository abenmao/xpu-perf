import os
import sys
import math
import pathlib
from functools import partial

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[5])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype


_ark_instance = None


def _resolve_framework_repo() -> pathlib.Path:
    xpu_perf_root = pathlib.Path(__file__).resolve().parents[5]

    env_repo = os.environ.get("SYCL_TLA_REPO", "").strip()
    if env_repo:
        p = pathlib.Path(env_repo)
        if p.is_dir():
            return p

    default_repo = xpu_perf_root.parent / "frameworks.ai.lpot.auto-round"
    if default_repo.is_dir():
        return default_repo

    for p in xpu_perf_root.parent.iterdir():
        name = p.name.lower()
        if p.is_dir() and name.startswith("frameworks") and "auto-round" in name:
            return p

    raise FileNotFoundError(
        "Cannot locate frameworks.ai.lpot.auto-round repo. "
        f"Set SYCL_TLA_REPO or create sibling path {default_repo}"
    )


def _load_ark():
    global _ark_instance
    if _ark_instance is not None:
        return _ark_instance

    repo = _resolve_framework_repo()
    ark_dir = repo / "auto_round_extension" / "ark"
    if not ark_dir.is_dir():
        raise FileNotFoundError(f"ark dir not found: {ark_dir}")

    # Import wrapper package that exposes ARK and internally loads auto_round_kernel_xpu.
    sys.path.insert(0, str(ark_dir))
    import auto_round_kernel

    _ark_instance = auto_round_kernel.ARK()
    if _ark_instance.xpu_lib is None:
        raise RuntimeError("auto_round_kernel loaded but xpu_lib is None")
    return _ark_instance


try:
    @ProviderRegistry.register_vendor_impl("sage_attention_v1", "sycl-tla")
    class SyclTlaSageAttentionV1Op(FlashAttentionOp):
        SUPPORTED_HDIMS = [64, 96, 128, 192]

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sycl-tla"]
            self._ark = _load_ark()

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if self.arg_type not in ["llm"]:
                raise NotImplementedError

            self.dtype = self.args_dict.get("dtype", "float16")
            if self.dtype not in ["float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = get_torch_dtype(self.dtype)

            self.q_head_num = self.args_dict["q_head_num"]
            self.kv_head_num = self.args_dict["kv_head_num"]
            self.head_dim = self.args_dict["head_dim"]
            if self.head_dim not in self.SUPPORTED_HDIMS:
                raise ValueError(
                    f"head_dim {self.head_dim} is not supported, expected one of {self.SUPPORTED_HDIMS}"
                )

            if self.q_head_num % self.kv_head_num != 0:
                raise ValueError(
                    f"q_head_num ({self.q_head_num}) must be divisible by kv_head_num ({self.kv_head_num})"
                )

            self.batch_size = self.args_dict["batch_size"]
            self.attn_mode = self.args_dict.get("attn_mode", self.args_dict.get("mode", "prefill"))
            self.is_causal = self.args_dict.get("is_causal", self.attn_mode == "decode")
            self.block_size = self.args_dict.get("block_size", 64)

            q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len", 1))
            k_seq_len = self.args_dict.get("k_seq_len", self.args_dict.get("kv_seq_len", q_seq_len))
            max_q_len = self.args_dict.get("max_q_len", q_seq_len)
            max_cache_len = self.args_dict.get("max_cache_len", k_seq_len)

            if self.attn_mode == "decode":
                self.q_seq_len = max_q_len
                self.kv_seq_len = max_cache_len
            else:
                self.q_seq_len = q_seq_len
                self.kv_seq_len = k_seq_len

            self.scale = float(self.head_dim ** (-0.5))

            self.input_tensor_info = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_head_num, self.q_seq_len, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "k": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, self.kv_seq_len, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "v": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, self.kv_seq_len, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
            }

            self.output_tensor_info = {
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_head_num, self.q_seq_len, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                )
            }

            self.input_tensor_size = sum(calc_tensor_size(info) for info in self.input_tensor_info.values())
            self.output_tensor_size = sum(calc_tensor_size(info) for info in self.output_tensor_info.values())
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self.skip_profiling = True

            base_flops = 4 * self.batch_size * self.q_head_num * self.q_seq_len * self.kv_seq_len * self.head_dim
            if self.is_causal:
                ratio = (
                    self.q_seq_len * self.kv_seq_len - self.q_seq_len * self.q_seq_len / 2
                ) / (self.q_seq_len * self.kv_seq_len)
                self.calc_flops = int(base_flops * ratio)
            else:
                self.calc_flops = int(base_flops)

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.sage_attention_v1_run

        def sage_attention_v1_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            v = tensor_mapping["v"]
            out = self._ark.sagev1(
                q,
                k,
                v,
                scale=self.scale,
                is_causal=self.is_causal,
                quant_block_size=self.block_size,
            )
            torch.xpu.synchronize(q.device)
            if "out" in tensor_mapping:
                tensor_mapping["out"].copy_(out)
                return tensor_mapping["out"]
            return out

except Exception as e:
    print(f"[SyclTlaSageAttentionV1Op] Failed to register: {e}")

