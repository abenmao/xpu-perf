import pathlib
import sys
import warnings

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[5])
)

from core.op import ProviderRegistry

try:
    from backends.INTEL.ops.sycl_ext.flash_attention_provider import (
        SYCLExtFlashAttentionOp,
    )

    @ProviderRegistry.register_vendor_impl("flash_attention", "sycl_tla_flash_attention")
    class SyclTlaFAOp(SYCLExtFlashAttentionOp):
        """Compatibility provider that keeps the sycl_tla name on top of sycl_ext."""

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self._provider = "sycl_tla_flash_attention"
            self.extra_providers = ["sycl_tla_flash_attention"]


except Exception as e:
    warnings.warn(
        "[SyclTlaFAOp] Failed to register sycl_tla_flash_attention with the "
        f"sycl_ext backend: {e}"
    )
