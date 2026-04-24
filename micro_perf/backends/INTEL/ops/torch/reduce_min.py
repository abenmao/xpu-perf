from core.ops.vector_reduction_ops import ReduceMinOp


OP_MAPPING = {}


class INTELReduceMinTorchOp(ReduceMinOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self._provider = "torch"


# Expose reduce_min torch implementation only via provider name "torch".
OP_MAPPING["torch"] = INTELReduceMinTorchOp
