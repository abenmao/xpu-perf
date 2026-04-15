import sys
import os
import re
import pathlib
import subprocess

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[5])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp

try:
    # Resolve sycl-tla path relative to xpu-perf root (xpu-perf/../sycl-tla)
    # flash_attention.py -> ops -> INTEL -> backends -> micro_perf -> xpu-perf
    _XPU_PERF_ROOT = pathlib.Path(__file__).resolve().parents[5]
    SYCL_TLA_DIR = str(_XPU_PERF_ROOT.parent / "sycl-tla")
    SYCL_TLA_BUILD_DIR = os.path.join(
        SYCL_TLA_DIR, "build/examples/06_bmg_flash_attention"
    )

    if not os.path.isdir(SYCL_TLA_BUILD_DIR):
        print(
            f"[WARNING] sycl-tla build dir not found at {SYCL_TLA_BUILD_DIR}. "
            f"sycl_tla_flash_attention provider will NOT be available. "
            f"Expected sycl-tla repo at {SYCL_TLA_DIR} (sibling of {_XPU_PERF_ROOT})."
        )
        raise FileNotFoundError(SYCL_TLA_BUILD_DIR)


    if not os.path.isdir(SYCL_TLA_BUILD_DIR):
        print(
            f"[WARNING] sycl-tla build dir not found at {SYCL_TLA_BUILD_DIR}. "
            f"sycl_tla_flash_attention provider will NOT be available."
        )
        raise FileNotFoundError(SYCL_TLA_BUILD_DIR)

    @ProviderRegistry.register_vendor_impl("flash_attention", "sycl_tla_flash_attention")
    class SyclTlaFAOp(FlashAttentionOp):
        # Supported head dims per the sycl-tla CMakeLists.txt
        SUPPORTED_HDIMS = [64, 96, 128, 192]

        # Dtype mapping from framework names to sycl-tla binary name components
        DTYPE_MAP = {
            "bfloat16": "bfloat16",
            "float8": "float_e4m3",
        }

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.head_dim not in self.SUPPORTED_HDIMS:
                raise ValueError(
                    f"SyclTlaFAOp only supports head_dim in {self.SUPPORTED_HDIMS}, got {self.head_dim}"
                )

            # Run sycl-tla binary and store parsed results
            self._sycl_tla_result = self._run_sycl_tla()

            # Override run to no-op; benchmarking was done by the binary
            self._run_func = lambda tensor_mapping: None
            self._create_tensors_func = lambda instance_num: [{}] * max(instance_num, 1)

        def _get_binary_path(self):
            binary_dtype = self.DTYPE_MAP.get(self.cache_dtype, "bfloat16")
            binary_name = (
                f"06_xe_fmha_fwd_{self.attn_mode}_{binary_dtype}_t_hdim{self.head_dim}"
            )
            return os.path.join(SYCL_TLA_BUILD_DIR, binary_name)

        def _run_sycl_tla(self):
            binary_path = self._get_binary_path()
            if not os.path.isfile(binary_path):
                raise FileNotFoundError(f"sycl-tla binary not found: {binary_path}")

            if self.attn_mode == "prefill":
                # Prefill: q and kv have the same sequence length per batch element
                seq_qo = self.q_lens[0]
                seq_kv = self.kv_lens[0]
                cmd = (
                    f"{binary_path} "
                    f"--iterations=100 --batch={self.batch_size} --verify=0 "
                    f"--num_heads_q={self.q_head_num} --num_heads_kv={self.kv_head_num} "
                    f"--head_size_qk={self.head_dim} --head_size_vo={self.head_dim} "
                    f"--seq_len_qo={seq_qo} --seq_len_kv={seq_kv}"
                )
            elif self.attn_mode == "decode":
                # Decode: small q_len (new tokens), large kv_cache (previously cached)
                seq_qo = self.max_q_len
                seq_kv = self.max_q_len  # new KV tokens = new query tokens
                cache_len = self.max_cache_len
                cmd = (
                    f"{binary_path} "
                    f"--iterations=100 --batch={self.batch_size} --verify=0 "
                    f"--num_heads_q={self.q_head_num} --num_heads_kv={self.kv_head_num} "
                    f"--head_size_qk={self.head_dim} --head_size_vo={self.head_dim} "
                    f"--seq_len_qo={seq_qo} --seq_len_kv={seq_kv} "
                    f"--seq_len_kv_cache={cache_len}"
                )
            else:
                raise ValueError(f"Unsupported attn_mode: {self.attn_mode}")

            if self.is_causal:
                cmd += " --is_causal"

            print(f"[SyclTlaFAOp] Running: {cmd}")

            result = subprocess.run(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                start_new_session=True,
                close_fds=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"sycl-tla binary failed (rc={result.returncode}):\n{result.stdout}"
                )

            print(f"[SyclTlaFAOp] Output:\n{result.stdout}")

            perf_match = re.search(
                r"Performance:\s+([\d\.]+)\s+GB/s,\s+([\d\.]+)\s+TFlop/s,\s+([\d\.]+)\s+ms",
                result.stdout,
            )
            if not perf_match:
                raise RuntimeError(
                    f"Failed to parse sycl-tla output:\n{result.stdout}"
                )

            return {
                "gb_s": float(perf_match.group(1)),
                "tflops": float(perf_match.group(2)),
                "latency_ms": float(perf_match.group(3)),
            }

        def summary(self, latency_us, kernel_mapping={}):
            if self._sycl_tla_result:
                latency_us = self._sycl_tla_result["latency_ms"] * 1000.0
            return super().summary(latency_us, kernel_mapping)


except Exception as e:
    print(f"[SyclTlaFAOp] Failed to register: {e}")
    pass
