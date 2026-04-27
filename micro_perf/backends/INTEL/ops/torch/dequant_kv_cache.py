import torch
from core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype, get_attn_info
from core.op import BasicOp
import sys
import pathlib
from functools import partial

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)


# ── Triton fused dequant kernel ──────────────────────────────────────────────
if HAS_TRITON:
    _DEQUANT_CONFIGS = [
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [1, 2, 4, 8]
        for s in [1, 2, 3]
    ]

    @triton.autotune(configs=_DEQUANT_CONFIGS, key=['B', 'H', 'L'])
    @triton.jit
    def _fused_dequant_kernel(
        src_ptr, scale_ptr, dst_ptr,
        B, H, L,
        D: tl.constexpr,
        s_src_b, s_src_h, s_src_l, s_src_d,
        s_dst_b, s_dst_h, s_dst_l, s_dst_d,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        l_idx = pid % L
        tmp = pid // L
        h_idx = tmp % H
        b_idx = tmp // H

        d_offs = tl.arange(0, BLOCK_D)
        mask = d_offs < D

        scale = tl.load(scale_ptr + h_idx * D + d_offs, mask=mask).to(tl.bfloat16)
        src_off = b_idx * s_src_b + h_idx * s_src_h + l_idx * s_src_l + d_offs * s_src_d
        x = tl.load(src_ptr + src_off, mask=mask).to(tl.bfloat16)

        y = x * scale

        dst_off = b_idx * s_dst_b + h_idx * s_dst_h + l_idx * s_dst_l + d_offs * s_dst_d
        tl.store(dst_ptr + dst_off, y, mask=mask)

    def triton_fused_dequant(src, scale, dst, kv_len):
        """src: [B,H,max_seq,D] int8, scale: [H,D] bf16, dst: [B,H,max_seq,D] bf16"""
        B, H = src.shape[0], src.shape[1]
        D = src.shape[3]
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * H * kv_len,)
        _fused_dequant_kernel[grid](
            src, scale, dst,
            B, H, kv_len, D,
            src.stride(0), src.stride(1), src.stride(2), src.stride(3),
            dst.stride(0), dst.stride(1), dst.stride(2), dst.stride(3),
            BLOCK_D=BLOCK_D,
        )


class DequantKVCacheOp(BasicOp):
    """Standalone INTEL dequant_kv_cache implementation.

    Upstream removed this op class; we keep it here for int8/float8
    KV-cache dequantization benchmarking on Intel GPUs.
    """

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)

    def prepare(self):
        self.arg_type = self.args_dict["arg_type"]
        if self.arg_type not in ["llm", "batch_llm"]:
            raise ValueError

        self.attn_mode = self.args_dict.get("attn_mode", "prefill")
        if self.attn_mode not in ["prefill", "decode"]:
            raise ValueError
        get_attn_info(self.arg_type, self.attn_mode, self.args_dict, self)

        # src (quant) dtype
        self.dtype = self.args_dict.get("dtype", "int8")
        if self.dtype not in ["int8", "float8"]:
            raise ValueError
        self.torch_dtype = get_torch_dtype(self.dtype)

        # dequant target dtype
        self.dst_dtype = self.args_dict.get("dst_dtype", "bfloat16")
        if self.dst_dtype not in ["bfloat16"]:
            raise ValueError
        self.dst_torch_dtype = get_torch_dtype(self.dst_dtype)

        self.kv_head_num = self.args_dict["kv_head_num"]
        self.head_dim = self.args_dict["head_dim"]

        self.quant_mode = self.args_dict.get("quant_mode", "static")
        if self.quant_mode not in ["static"]:
            raise ValueError

        # all tokens with same head/head_dim element pos share one scale
        if self.quant_mode == "static":
            self.quant_scale_shape = [self.kv_head_num, self.head_dim]

        self.input_tensor_info = {
            "kv_lens": OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.kv_lens, dtype=dtype, device=device
                ),
            ),
            "k_scale": OpTensorInfo(
                shape=self.quant_scale_shape,
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
            "v_scale": OpTensorInfo(
                shape=self.quant_scale_shape,
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            ),
        }
        self.output_tensor_info = {}

        if self.cache_type == "linear":
            self.input_tensor_info["slot_mapping"] = OpTensorInfo(
                shape=[self.batch_size],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.slot_mapping, dtype=dtype, device=device
                ),
            )
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )

        elif self.cache_type == "paged":
            self.input_tensor_info["block_table"] = OpTensorInfo(
                shape=[self.batch_size, self.max_block_num_per_seq],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=lambda size, dtype, device: torch.tensor(
                    self.block_table, dtype=dtype, device=device
                ),
            )
            self.input_tensor_info["k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.input_tensor_info["v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_k_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )
            self.output_tensor_info["dequant_v_cache"] = OpTensorInfo(
                shape=[self.num_kv_blocks, self.kv_head_num, self.block_size, self.head_dim],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
                creator=torch.empty,
            )

        self.input_tensor_size = sum(
            [calc_tensor_size(info) for info in self.input_tensor_info.values()]
        )
        self.output_tensor_size = sum(
            [calc_tensor_size(info) for info in self.output_tensor_info.values()]
        )
        self.tensor_size = self.input_tensor_size + self.output_tensor_size

        self.read_bytes = (
            calc_tensor_size(self.input_tensor_info["kv_lens"])
            + calc_tensor_size(self.input_tensor_info["k_scale"])
            + calc_tensor_size(self.input_tensor_info["v_scale"])
        )

        if self.cache_type == "linear":
            self.read_bytes += (
                calc_tensor_size(self.input_tensor_info["slot_mapping"])
                + calc_tensor_size(self.input_tensor_info["k_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
                + calc_tensor_size(self.input_tensor_info["v_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
            )
            self.write_bytes = (
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
                + calc_tensor_size(self.output_tensor_info["dequant_v_cache"])
                / self.batch_size
                / self.max_kv_len
                * self.num_kv_tokens
            )

        elif self.cache_type == "paged":
            self.read_bytes += (
                calc_tensor_size(self.input_tensor_info["block_table"])
                / self.batch_size
                / self.max_block_num_per_seq
                * self.num_kv_blocks
                + calc_tensor_size(self.input_tensor_info["k_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
                + calc_tensor_size(self.input_tensor_info["v_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
            )
            self.write_bytes = (
                calc_tensor_size(self.output_tensor_info["dequant_k_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
                + calc_tensor_size(self.output_tensor_info["dequant_v_cache"])
                / self.num_kv_blocks
                / self.block_size
                * self.num_kv_tokens
            )

        self.io_bytes = self.read_bytes + self.write_bytes

        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )

        self._run_func = self.dequant_kv_cache_run

    def dequant_kv_cache_run(self, tensor_mapping):
        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        dequant_k_cache = tensor_mapping["dequant_k_cache"]
        dequant_v_cache = tensor_mapping["dequant_v_cache"]

        kv_lens = tensor_mapping["kv_lens"]
        k_scale = tensor_mapping["k_scale"]
        v_scale = tensor_mapping["v_scale"]

        if self.cache_type == "paged":
            block_table = tensor_mapping["block_table"]
            raise NotImplementedError(
                "DequantKVCacheOp paged cache not implemented yet."
            )

        if self.cache_type == "linear":
            kv_len = self.kv_lens[0]

            if HAS_TRITON and self.kv_head_num > 4:
                k_scale_bf16 = k_scale.to(self.dst_torch_dtype)
                v_scale_bf16 = v_scale.to(self.dst_torch_dtype)
                triton_fused_dequant(k_cache, k_scale_bf16, dequant_k_cache, kv_len)
                triton_fused_dequant(v_cache, v_scale_bf16, dequant_v_cache, kv_len)
            else:
                k_s = k_scale.to(self.dst_torch_dtype).unsqueeze(0).unsqueeze(2)
                v_s = v_scale.to(self.dst_torch_dtype).unsqueeze(0).unsqueeze(2)
                if self.batch_size <= 4:
                    # Vectorized: good for small batch
                    src_k = k_cache[:, :, :kv_len, :].to(self.dst_torch_dtype)
                    src_v = v_cache[:, :, :kv_len, :].to(self.dst_torch_dtype)
                    dequant_k_cache[:, :, :kv_len, :] = src_k * k_s
                    dequant_v_cache[:, :, :kv_len, :] = src_v * v_s
                else:
                    # Per-batch: avoids huge temporaries for large batch
                    for b in range(self.batch_size):
                        src_k = k_cache[b:b+1, :, :kv_len, :].to(self.dst_torch_dtype)
                        src_v = v_cache[b:b+1, :, :kv_len, :].to(self.dst_torch_dtype)
                        dequant_k_cache[b:b+1, :, :kv_len, :] = src_k * k_s
                        dequant_v_cache[b:b+1, :, :kv_len, :] = src_v * v_s

        return dequant_k_cache, dequant_v_cache


OP_MAPPING = {"dequant_kv_cache": DequantKVCacheOp}
