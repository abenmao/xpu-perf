import sys
import pathlib
import os
from functools import partial

import torch
import triton
import triton.language as tl
from triton.language.extra.intel import libdevice as _libd

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import ScaleDynamicQuantOp


# Triton single-pass kernel
# amax / normalize 共享同一份 fp32 buffer，DRAM 上 hidden_states 只读 1 次
INT8_MAX = 127.0
_INT8_MAX_T = tl.constexpr(127.0)


@triton.jit
def _smooth_dyn_quant_kernel(
    X_ptr, S_ptr, Q_ptr, PS_ptr,
    stride_xn, stride_qn,
    H,
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    x = tl.load(X_ptr + pid * stride_xn + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xs = x * s  # [BLOCK_H] fp32, 留在寄存器

    amax = tl.max(tl.abs(xs), axis=0)
    inv_scale = _INT8_MAX_T / amax
    scale = amax / _INT8_MAX_T

    q = xs * inv_scale
    q = tl.clamp(q, -_INT8_MAX_T, _INT8_MAX_T)
    q = _libd.rint(q).to(tl.int8)

    tl.store(Q_ptr + pid * stride_qn + offs, q, mask=mask)
    tl.store(PS_ptr + pid, scale)


def _smooth_dyn_quant_triton(hidden_states, smooth_scale, out_q, out_s):
    N, H = hidden_states.shape
    BLOCK_H = triton.next_power_of_2(H)
    # 测过 (4,2)/(8,2)/(16,2)/(16,3)/(32,3)，(32,3) 整体最优
    _smooth_dyn_quant_kernel[(N,)](
        hidden_states, smooth_scale, out_q, out_s,
        hidden_states.stride(0), out_q.stride(0),
        H,
        BLOCK_H=BLOCK_H,
        num_warps=32,
        num_stages=3,
    )


# 备选 torch.compile 路径，可通过 SDQ_USE_TRITON=0 切换
_COMPILED_CACHE = {}


def _make_compiled():
    def _fn(hidden_states, smooth_scale, out_q, out_s):
        x = hidden_states.float() * smooth_scale
        amax = x.abs().amax(dim=-1)
        scale = amax / INT8_MAX
        out_s.copy_(scale)
        q = (x / scale.unsqueeze(-1)).clamp(-INT8_MAX, INT8_MAX).round()
        out_q.copy_(q.to(torch.int8))

    return torch.compile(_fn, dynamic=False, fullgraph=True, mode="reduce-overhead")


def _get_compiled():
    fn = _COMPILED_CACHE.get("default")
    if fn is None:
        fn = _make_compiled()
        _COMPILED_CACHE["default"] = fn
    return fn


@ProviderRegistry.register_vendor_impl("scale_dynamic_quant", "torch")
class ScaleDynamicQuantTorchOp(ScaleDynamicQuantOp):
    def vendor_impl(self):
        super().vendor_impl()
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )
        self._use_triton = os.environ.get("SDQ_USE_TRITON", "1") != "0"
        if not self._use_triton:
            self._compiled_fn = _get_compiled()

    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]
        quant_tokens = tensor_mapping["quant_tokens"]
        per_token_scale = tensor_mapping["per_token_scale"]

        if self._use_triton:
            _smooth_dyn_quant_triton(
                hidden_states, smooth_scale, quant_tokens, per_token_scale,
            )
        else:
            self._compiled_fn(hidden_states, smooth_scale, quant_tokens, per_token_scale)
        return quant_tokens, per_token_scale
