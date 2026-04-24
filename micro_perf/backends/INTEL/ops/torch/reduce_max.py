from core.ops.vector_reduction_ops import ReduceMaxOp


OP_MAPPING = {}


class INTELReduceMaxTorchOp(ReduceMaxOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._provider = "torch"


OP_MAPPING["torch"] = INTELReduceMaxTorchOp
