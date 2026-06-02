import importlib.util
import os
import pathlib
import ctypes

import torch


_XPU_PERF_ROOT = pathlib.Path(__file__).resolve().parents[5]
_SYCL_EXT_SO = os.path.join(
    str(_XPU_PERF_ROOT),
    "micro_perf/backends/INTEL/ops/sycl_ext/flash_attention_sycl.so",
)
_sycl_ext = None


def _preload_torch_shared_libs():
    if os.name != "posix":
        return

    torch_lib_dir = pathlib.Path(torch.__file__).resolve().parent / "lib"
    if not torch_lib_dir.is_dir():
        return

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in ("libc10.so", "libtorch.so", "libtorch_python.so", "libc10_xpu.so"):
        lib_path = torch_lib_dir / lib_name
        if lib_path.is_file():
            ctypes.CDLL(str(lib_path), mode=mode)


def _load_extension():
    if not os.path.isfile(_SYCL_EXT_SO):
        raise FileNotFoundError(
            f"sycl_ext shared object not found: {_SYCL_EXT_SO}. "
            "Build it with micro_perf/backends/INTEL/ops/sycl_ext/build.sh first."
        )

    _preload_torch_shared_libs()

    spec = importlib.util.spec_from_file_location("flash_attention_sycl", _SYCL_EXT_SO)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {_SYCL_EXT_SO}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_extension():
    global _sycl_ext
    if _sycl_ext is None:
        _sycl_ext = _load_extension()
    return _sycl_ext


def scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    output_dtype=None,
    output=None,
    plan_cache_len=None,
    plan_kv_new_len=None,
    plan_is_causal=None,
    plan_is_decode=None,
    plan_use_sycl_tla=None,
):
    return _get_extension().scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        output_dtype,
        output,
        plan_cache_len,
        plan_kv_new_len,
        plan_is_causal,
        plan_is_decode,
        plan_use_sycl_tla,
    )


def prepare_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    output_dtype=None,
    output=None,
    plan_cache_len=None,
    plan_kv_new_len=None,
    plan_is_causal=None,
    plan_is_decode=None,
    plan_use_sycl_tla=None,
):
    return _get_extension().prepare_scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        output_dtype,
        output,
        plan_cache_len,
        plan_kv_new_len,
        plan_is_causal,
        plan_is_decode,
        plan_use_sycl_tla,
    )


def run_prepared_scaled_dot_product_attention(prepared):
    return _get_extension().run_prepared_scaled_dot_product_attention(prepared)