import importlib.metadata
import traceback

INTEL_PROVIDER = {}


# https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    INTEL_PROVIDER["flash_attn_v2"] = {
        "flash_attn_v2": importlib.metadata.version("flash_attn")
    }
except:
    pass


# https://github.com/Dao-AILab/flash-attention
try:
    from flash_attn_interface import flash_attn_func, flash_attn_with_kvcache
    INTEL_PROVIDER["flash_attn_v3"] = {
        "flash_attn_v3": importlib.metadata.version("flash_attn"),
    }    
except:
    pass


# https://github.com/vllm-project/vllm-xpu-kernels
try:
    import vllm_xpu_kernels._C
    INTEL_PROVIDER["vllm_xpu_kernels"] = {
        "vllm_xpu_kernels": importlib.metadata.version("vllm-xpu-kernels"),
    }
except:
    pass

# flashmla-xpu (MLA attention kernels — built from sycl_ext/flashmla/)
try:
    import os, importlib.util as _ilu
    _mla_so = os.path.join(
        os.path.dirname(__file__), "ops", "sycl_ext", "flashmla",
        "flash_attn_xpu.abi3.so")
    if os.path.isfile(_mla_so):
        _spec = _ilu.spec_from_file_location("flash_attn_xpu", _mla_so)
        _ext = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_ext)
        INTEL_PROVIDER["flashmla_xpu"] = {"flashmla_xpu": "1.0"}
except:
    pass
