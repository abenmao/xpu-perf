# Flash Attention SYCL Ext

This document describes the current flash attention implementation under `micro_perf/backends/INTEL/ops/sycl_ext`, including the Python-facing interface, build methods, and run instructions.

## Overview

The current implementation adds a C++/pybind extension layer for a flash attention style operator with an interface aligned to `torch.nn.functional.scaled_dot_product_attention`.

Relevant files:

- `flash_attention.cpp`: C++ extension entrypoint and operator implementation.
- `flash_attention.py`: Python wrapper that loads `flash_attention_sycl.so` and exposes `scaled_dot_product_attention(...)`.
- `build.sh`: shared build script for all `sycl_ext` extensions, including `flash_attention_sycl.so`.
- `build-fa.sh`: standalone build script for compiling only `flash_attention_sycl.so`.

Current scope:

- The Python interface is SDPA-compatible.
- The extension can be compiled as a `.so` and imported from Python.
- The current implementation uses sycl-tla FMHA kernels for validated dense no-cache bf16 XPU cases.
- Cached-KV cases currently fall back to the ATen implementation path to preserve correctness.

This means the extension now has a real sycl-tla-backed execution path, but the full cached-KV replacement is not finished yet.

## Python Interface

The Python wrapper exports:

```python
scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
)
```

### Input layout

The current C++ implementation expects 4D tensors in the same layout used by the existing torch flash attention path in this repo:

- `query`: `[batch, heads_q, q_len, head_dim]`
- `key`: `[batch, heads_kv, kv_len, head_dim]`
- `value`: `[batch, heads_kv, kv_len, head_dim]`

### Supported semantics

- `attn_mask=None` or a provided mask tensor.
- `is_causal=True/False`.
- `scale=None` or explicit scaling factor.
- `enable_gqa=True` for grouped query attention, with KV heads expanded to query head count.
- `dropout_p` in `[0, 1)`.

### Current backend routing

The implementation chooses between two backends at runtime:

- sycl-tla FMHA path:
    - device is XPU,
    - dtype is bf16,
    - head dimension is one of `64/96/128/192`,
    - `dropout_p == 0`,
    - no KV cache is inferred from the inputs.
- ATen fallback path:
    - CPU execution,
    - unsupported dtype or head dimension,
    - non-zero dropout,
    - cached-KV cases such as decode-with-cache or prefix causal mask.

This routing is intentional. The dense no-cache sycl-tla path is validated. Cached-KV wiring in the current remote checkout still requires a dedicated integration based on the 06 legacy cached-KV implementation.

### Validation behavior

The extension performs basic checks before running:

- `query`, `key`, and `value` must all be 4D.
- All tensors must be on the same device.
- Dtypes must match.
- Batch sizes and head dimensions must match.
- If `enable_gqa=False`, query heads must equal key/value heads.
- If `enable_gqa=True`, query heads must be divisible by key/value heads.
- `attn_mask` and `is_causal` cannot both be set.

## Build Methods

Prerequisites:

- `icpx` must already be available in the environment.
- `python3` or `python` must be available.
- `sycl-tla` must exist as a sibling checkout of `xpu-perf`, so that `${XPU_PERF_ROOT}/../sycl-tla` resolves correctly.

### Method 1: Build through `build.sh`

This is the recommended method if you are already building other `sycl_ext` operators.

```bash
cd /home/mjc/xpu-perf/micro_perf/backends/INTEL/ops/sycl_ext
bash build.sh
```

For flash attention, `build.sh` now compiles with the sycl-tla include and link stack, including:

```bash
icpx -shared -fPIC -O3 -DNDEBUG -std=c++17 \
    -DTORCH_EXTENSION_NAME=flash_attention_sycl \
    $SYCL_TLA_COMPILE_FLAGS \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    $SYCL_TLA_INCLUDES \
    flash_attention.cpp \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu \
    $SYCL_TLA_LINK_FLAGS \
    -Xsycl-target-backend=spir64_gen "-device bmg-g21" \
    -Xspirv-translator \
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    "${SYCL_TLA_RUNTIME_PATHS[@]}" \
    -L/lib64/stubs \
    -o flash_attention_sycl.so \
    $SYCL_TLA_LINK_LIBS
```

Notes:

- `build.sh` now auto-selects `python3` first, and falls back to `python` if needed.
- Torch include and library paths are resolved from the active Python environment.

### Method 2: Build only flash attention

If you only want to validate this operator without compiling all other `sycl_ext` targets, use the dedicated script:

```bash
cd /home/mjc/xpu-perf/micro_perf/backends/INTEL/ops/sycl_ext
bash build-fa.sh
```

`build-fa.sh` uses the same validated sycl-tla include, link, runtime-path, and ABI-detection logic as `build.sh`, but only builds `flash_attention_sycl.so`.

Validated result in container `be03ca73faf9`:

- artifact: `/home/mjc/xpu-perf/micro_perf/backends/INTEL/ops/sycl_ext/flash_attention_sycl.so`
- timestamp: `2026-05-28 05:05:43 +0000`
- size: `4455976` bytes

## Run Methods

### Method 1: Direct Python import

After `flash_attention_sycl.so` is built, import the wrapper through the repo package path:

```bash
cd /home/mjc/xpu-perf

/opt/venv/bin/python3 - <<'PY'
from micro_perf.backends.INTEL.ops.sycl_ext.flash_attention import scaled_dot_product_attention
print("import ok", callable(scaled_dot_product_attention))
PY
```

Validated result in container `be03ca73faf9`:

- `import ok True`

### Method 2: Compare against torch SDPA

This is the recommended smoke test for semantics alignment:

```bash
cd /home/mjc/xpu-perf

/opt/venv/bin/python3 - <<'PY'
import torch
import torch.nn.functional as F
from micro_perf.backends.INTEL.ops.sycl_ext import flash_attention as fa

torch.manual_seed(0)
q = torch.randn(2, 4, 3, 8, dtype=torch.float32)
k = torch.randn(2, 2, 5, 8, dtype=torch.float32)
v = torch.randn(2, 2, 5, 8, dtype=torch.float32)

out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
out_ext = fa.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)

print("max_abs_diff", (out_ref - out_ext).abs().max().item())
print("shape", tuple(out_ext.shape))
PY
```

Verified result on the remote container used for this implementation:

- `max_abs_diff = 2.980232238769531e-07`
- `shape = (2, 4, 3, 8)`

Additional validated bf16 XPU cases after switching to the sycl-tla-backed path:

- `prefill_causal`: `max_abs_diff = 0.0`
- `decode_cached`: `max_abs_diff = 0.0078125`
- `prefix_causal_mask`: `max_abs_diff = 0.0084228515625`

The latter two cases currently run through the fallback path because cached-KV integration is not yet enabled by default.

### Method 3: Validate API constraint behavior

The current implementation follows the SDPA rule that `attn_mask` and `is_causal` cannot both be set:

```bash
cd /home/mjc/xpu-perf

/opt/venv/bin/python3 - <<'PY'
import torch
from micro_perf.backends.INTEL.ops.sycl_ext import flash_attention as fa

q = torch.randn(2, 4, 3, 8, dtype=torch.float32)
k = torch.randn(2, 2, 5, 8, dtype=torch.float32)
v = torch.randn(2, 2, 5, 8, dtype=torch.float32)

try:
    fa.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=torch.ones(3, 5, dtype=torch.bool),
        is_causal=True,
        enable_gqa=True,
    )
except RuntimeError as exc:
    print(exc)
PY
```

## Remote Build And Run Workflow

The current remote workflow validated in this task is:

- SSH target: `root@10.239.11.59`
- Container: `be03ca73faf9`
- Repo path in container: `/home/mjc/xpu-perf`
- sycl-tla path in container: `/home/mjc/sycl-tla`
- Python executable: `/opt/venv/bin/python3`

Example workflow:

```bash
ssh root@10.239.11.59
docker exec -it be03ca73faf9 bash

cd /home/mjc/xpu-perf/micro_perf/backends/INTEL/ops/sycl_ext
bash build.sh
```

Or build only flash attention and validate import:

```bash
ssh root@10.239.11.59
docker exec -it be03ca73faf9 bash

cd /home/mjc/xpu-perf/micro_perf/backends/INTEL/ops/sycl_ext
bash build-fa.sh

cd /home/mjc/xpu-perf
/opt/venv/bin/python3 -c "from micro_perf.backends.INTEL.ops.sycl_ext.flash_attention import scaled_dot_product_attention; print('import ok', callable(scaled_dot_product_attention))"
```

## Current Limitations

- The current implementation is not yet wired into the `micro_perf` provider path as a standalone benchmark provider.
- Cached-KV execution is not yet enabled on the sycl-tla path by default.
- The full cached-KV replacement should be based on the dedicated 06 legacy cached-KV implementation rather than the current benchmark configuration shortcut.

## Recommended Next Step

To replace the current external sycl-tla binary workflow completely, the next step should be finishing the cached-KV integration around the 06 legacy cached-KV implementation so that:

- accepts external tensor/device memory instead of allocating benchmark-owned buffers internally,
- exposes a callable C++ API suitable for `.so` packaging,
- maps SDPA concepts such as mask, causal, scale, and GQA into the underlying kernel configuration,
- replaces the current fallback path for decode-with-cache and prefix-causal cases.