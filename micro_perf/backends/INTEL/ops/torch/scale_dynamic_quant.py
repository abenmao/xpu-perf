import sys
import pathlib

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.llm_ops import ScaleDynamicQuantOp

OP_MAPPING = {}


def smooth_per_token_dynamic_quant_opt(
    hidden_states: torch.Tensor,
    smooth_scale: torch.Tensor,
    dst_torch_dtype=torch.int8
):
    max_dtype_val = 127.0 if dst_torch_dtype == torch.int8 else 448.0

    ori_shape = hidden_states.shape
    smoothed_input = (hidden_states.view(ori_shape[0], -1) * smooth_scale.view(1, -1)).float()

    per_token_scale = smoothed_input.abs().amax(dim=-1, keepdim=True) / max_dtype_val

    quant_tokens_fp32 = (smoothed_input / per_token_scale).clamp(-max_dtype_val, max_dtype_val)
    if dst_torch_dtype == torch.int8:
        quant_tokens_fp32 = quant_tokens_fp32.round()

    quant_tokens = quant_tokens_fp32.to(dst_torch_dtype).view(ori_shape)
    per_token_scale = per_token_scale.squeeze(-1)

    return quant_tokens, per_token_scale

smooth_per_token_dynamic_quant_compiled = torch.compile(smooth_per_token_dynamic_quant_opt)


class ScaleDynamicQuantTorchCompiledOp(ScaleDynamicQuantOp):
    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]

        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant_compiled(
            hidden_states, smooth_scale, self.dst_torch_dtype
        )
        return quant_tokens, per_token_scale


OP_MAPPING["torch_compiled"] = ScaleDynamicQuantTorchCompiledOp
