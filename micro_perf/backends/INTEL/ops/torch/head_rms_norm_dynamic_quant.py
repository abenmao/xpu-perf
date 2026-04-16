import sys
import pathlib

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.ops.llm_ops import HeadRMSNormDynamicQuantOp as BaseHeadRMSNormDynamicQuantOp


class HeadRMSNormDynamicQuantTorchOp(BaseHeadRMSNormDynamicQuantOp):
    pass


OP_MAPPING = {"torch": HeadRMSNormDynamicQuantTorchOp}
