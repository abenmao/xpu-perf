#!/bin/bash
# Build SYCL KV cache kernels as PyTorch extension
# Usage: source /opt/intel/oneapi/setvars.sh && bash build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get torch include/lib paths
TORCH_INCLUDES=$(python3 -c "
import torch.utils.cpp_extension as ext
for p in ext.include_paths():
    print(f'-I{p}', end=' ')
")

TORCH_LIBS=$(python3 -c "
import torch.utils.cpp_extension as ext
for p in ext.library_paths():
    print(f'-L{p}', end=' ')
")

# Python include
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

echo "Building store_kv_cache SYCL extension..."
icpx -fsycl -shared -fPIC -O2 -std=c++17 \
    -DTORCH_EXTENSION_NAME=store_kv_cache_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    store_kv_cache_kernel.cpp \
    -o store_kv_cache_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10

echo "Built: $SCRIPT_DIR/store_kv_cache_sycl.so"
ls -la store_kv_cache_sycl.so

echo ""
echo "Building dequant_kv_cache SYCL extension..."
icpx -fsycl -shared -fPIC -O2 -std=c++17 \
    -DTORCH_EXTENSION_NAME=dequant_kv_cache_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    dequant_kv_cache_kernel.cpp \
    -o dequant_kv_cache_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10

echo "Built: $SCRIPT_DIR/dequant_kv_cache_sycl.so"
ls -la dequant_kv_cache_sycl.so
