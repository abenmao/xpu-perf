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


# https://github.com/oneapi-src/oneDNN  (locally-built under <xpu-perf>/../oneDNN/build)
try:
    import os as _os
    import pathlib as _pathlib
    _ONEDNN_DIR = str(_pathlib.Path(__file__).resolve().parents[3].parent / "oneDNN")
    _ONEDNN_BENCHDNN = _os.path.join(_ONEDNN_DIR, "build", "tests", "benchdnn", "benchdnn")
    _ONEDNN_LIB = _os.path.join(_ONEDNN_DIR, "build", "src")
    if _os.path.isfile(_ONEDNN_BENCHDNN) and _os.path.isdir(_ONEDNN_LIB):
        INTEL_PROVIDER["onednn"] = {
            "onednn": f"local-build ({_ONEDNN_DIR}/build)",
        }
except:
    pass
