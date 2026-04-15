import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.llm_ops import ScaleDynamicQuantOp
from core.utils import smooth_per_token_dynamic_quant_compiled

OP_MAPPING = {}


class ScaleDynamicQuantTorchCompiledOp(ScaleDynamicQuantOp):
    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        smooth_scale = tensor_mapping["smooth_scale"]

        quant_tokens, per_token_scale = smooth_per_token_dynamic_quant_compiled(
            hidden_states, smooth_scale, self.dst_torch_dtype
        )
        return quant_tokens, per_token_scale


OP_MAPPING["torch_compiled"] = ScaleDynamicQuantTorchCompiledOp
