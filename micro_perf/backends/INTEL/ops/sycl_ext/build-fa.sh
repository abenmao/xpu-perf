#!/bin/bash
# Build only the flash_attention SYCL extension.
# Usage: bash build-fa.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icpx not found. Please source oneAPI setvars first."
    exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "ERROR: neither python3 nor python was found in PATH."
        exit 1
    fi
fi

TORCH_INCLUDES=$($PYTHON_BIN -c "
import torch.utils.cpp_extension as ext
for p in ext.include_paths():
    print(f'-I{p}', end=' ')
")

TORCH_LIBS=$($PYTHON_BIN -c "
import torch.utils.cpp_extension as ext
for p in ext.library_paths():
    print(f'-L{p}', end=' ')
")

PYTHON_INCLUDE=$($PYTHON_BIN -c "import sysconfig; print(sysconfig.get_path('include'))")

XPU_PERF_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SYCL_TLA_ROOT="$(cd "$XPU_PERF_ROOT/../sycl-tla" && pwd)"

MKLROOT=${MKLROOT:-/opt/intel/oneapi/mkl/latest}
TBBROOT=${TBBROOT:-/opt/intel/oneapi/tbb/latest}
CMPLR_ROOT=${CMPLR_ROOT:-/opt/intel/oneapi/compiler/latest}
SYCL_INTEL_TARGET=${SYCL_INTEL_TARGET:-35}

SYCL_TLA_INCLUDES="-I$SYCL_TLA_ROOT -I$SYCL_TLA_ROOT/include -I$SYCL_TLA_ROOT/tools/util/include -I$SYCL_TLA_ROOT/examples/common -I$SYCL_TLA_ROOT/examples/06_bmg_flash_attention -I$SYCL_TLA_ROOT/applications -isystem $MKLROOT/include"
SYCL_TLA_COMPILE_FLAGS="-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET=$SYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -DMKL_ILP64 -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen -Wall -Wno-unused-variable -Wno-unused-local-typedef -Wno-unused-but-set-variable -Wno-uninitialized -Wno-reorder-ctor -Wno-logical-op-parentheses -Wno-unused-function -Wno-unknown-pragmas"
SYCL_TLA_LINK_FLAGS="-fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen"
SYCL_TLA_LIB_DIRS="-L$MKLROOT/lib -L$TBBROOT/lib/intel64/gcc4.8"
SYCL_TLA_LINK_LIBS="$MKLROOT/lib/libmkl_intel_thread.so $CMPLR_ROOT/lib/libiomp5.so $MKLROOT/lib/libmkl_intel_ilp64.so $MKLROOT/lib/libmkl_core.so -fsycl $MKLROOT/lib/libmkl_sycl_blas.so $MKLROOT/lib/libmkl_tbb_thread.so $SYCL_TLA_LIB_DIRS -ltbb -lsycl -lOpenCL -lm -ldl -lpthread"
SYCL_TLA_RUNTIME_PATHS=(-Wl,-rpath,/lib64/stubs -Wl,-rpath,"$MKLROOT/lib" -Wl,-rpath,"$TBBROOT/lib/intel64/gcc4.8")

echo "Building flash_attention SYCL extension..."
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

echo "Built: $SCRIPT_DIR/flash_attention_sycl.so"
ls -la flash_attention_sycl.so