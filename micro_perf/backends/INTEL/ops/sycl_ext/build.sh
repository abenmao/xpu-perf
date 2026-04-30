#!/bin/bash
# Build SYCL kernels used by xpu-perf sycl_ext ops.
# Usage: source /opt/intel/oneapi/setvars.sh && bash build.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v icpx >/dev/null 2>&1; then
    echo "ERROR: icpx not found. Please source oneAPI setvars first."
    exit 1
fi

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

XPU_PERF_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
SYCL_TLA_ROOT="$(cd "$XPU_PERF_ROOT/../sycl-tla" && pwd)"

MKLROOT=${MKLROOT:-/opt/intel/oneapi/mkl/latest}
TBBROOT=${TBBROOT:-/opt/intel/oneapi/tbb/latest}
CMPLR_ROOT=${CMPLR_ROOT:-/opt/intel/oneapi/compiler/latest}

SYCL_TLA_INCLUDES="-I$SYCL_TLA_ROOT/include -I$SYCL_TLA_ROOT/tools/util/include -I$SYCL_TLA_ROOT/examples/common -isystem $MKLROOT/include"

SYCL_TLA_COMPILE_FLAGS="-DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DCUTLASS_VERSIONS_GENERATED -DMKL_ILP64 -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen -Wall -Wno-unused-variable -Wno-unused-local-typedef -Wno-unused-but-set-variable -Wno-uninitialized -Wno-reorder-ctor -Wno-logical-op-parentheses -Wno-unused-function -Wno-unknown-pragmas"
SYCL_TLA_LINK_FLAGS="-fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen"
SYCL_TLA_LIB_DIRS="-L$MKLROOT/lib -L$TBBROOT/lib/intel64/gcc4.8"
SYCL_TLA_LINK_LIBS="$MKLROOT/lib/libmkl_intel_thread.so $CMPLR_ROOT/lib/libiomp5.so $MKLROOT/lib/libmkl_intel_ilp64.so $MKLROOT/lib/libmkl_core.so -fsycl $MKLROOT/lib/libmkl_sycl_blas.so $MKLROOT/lib/libmkl_tbb_thread.so $SYCL_TLA_LIB_DIRS -ltbb -lsycl -lOpenCL -lm -ldl -lpthread"
SYCL_TLA_RUNTIME_PATHS=(-Wl,-rpath,/lib64/stubs -Wl,-rpath,"$MKLROOT/lib" -Wl,-rpath,"$TBBROOT/lib/intel64/gcc4.8")
BMG_09_LINK_FLAGS=(-Xs "-options \"-igc_opts 'VectorAliasBBThreshold=10000'\"")
BMG_10_LINK_FLAGS=(-Xs "-options \"-igc_opts 'allowDecompose2DBlockFuncs=0'\"")

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

echo ""
echo "Building reduce_min SYCL extension..."
icpx -fsycl -shared -fPIC -O3 -std=c++17 \
    -DTORCH_EXTENSION_NAME=reduce_min_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    reduce_min_kernel.cpp \
    -o reduce_min_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu

echo "Built: $SCRIPT_DIR/reduce_min_sycl.so"
ls -la reduce_min_sycl.so

echo ""
echo "Building reduce_max SYCL extension..."
icpx -fsycl -shared -fPIC -O3 -std=c++17 \
    -DTORCH_EXTENSION_NAME=reduce_max_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    reduce_max_kernel.cpp \
    -o reduce_max_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu

echo "Built: $SCRIPT_DIR/reduce_max_sycl.so"
ls -la reduce_max_sycl.so

echo ""
echo "Building moe_softmax_topk SYCL extension..."
icpx -fsycl -shared -fPIC -O3 -std=c++17 \
    -DTORCH_EXTENSION_NAME=moe_softmax_topk_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    moe_softmax_topk_kernel.cpp \
    -o moe_softmax_topk_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu

echo "Built: $SCRIPT_DIR/moe_softmax_topk_sycl.so"
ls -la moe_softmax_topk_sycl.so

echo ""
echo "Building scatter SYCL extension..."
icpx -fsycl -shared -fPIC -O3 -std=c++17 \
    -DTORCH_EXTENSION_NAME=scatter_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    scatter_kernel.cpp \
    -o scatter_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu

echo "Built: $SCRIPT_DIR/scatter_sycl.so"
ls -la scatter_sycl.so

echo ""
echo "Building head_rms_norm SYCL extension..."
icpx -fsycl -shared -fPIC -O3 -std=c++17 \
    -DTORCH_EXTENSION_NAME=head_rms_norm_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    head_rms_norm.cpp \
    -o head_rms_norm_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu

echo "Built: $SCRIPT_DIR/head_rms_norm_sycl.so"
ls -la head_rms_norm_sycl.so

echo ""
echo "Building scale_dynamic_quant SYCL extension..."
icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg-g21" \
    -shared -fPIC -O3 -std=c++17 \
    -DTORCH_EXTENSION_NAME=scale_dynamic_quant_sycl \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    scale_dynamic_quant_kernel.cpp \
    -o scale_dynamic_quant_sycl.so \
    $TORCH_LIBS \
    -ltorch -ltorch_python -lc10 -lc10_xpu

echo "Built: $SCRIPT_DIR/scale_dynamic_quant_sycl.so"
ls -la scale_dynamic_quant_sycl.so

echo "Building bmg_moe_gating_gemm_sycl SYCL extension..."
icpx -shared -fPIC -O3 -DNDEBUG -std=c++17 \
    -DTORCH_EXTENSION_NAME=bmg_moe_gating_gemm_sycl \
    $SYCL_TLA_COMPILE_FLAGS \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    $SYCL_TLA_INCLUDES \
    00_bmg_moe_gating_gemm.cpp \
    $SYCL_TLA_LINK_FLAGS \
    -Xsycl-target-backend=spir64_gen "-device bmg-g21" \
    -Xspirv-translator \
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    "${SYCL_TLA_RUNTIME_PATHS[@]}" \
    -L/lib64/stubs \
    -o bmg_moe_gating_gemm_sycl.so \
    $SYCL_TLA_LINK_LIBS

echo "Built: $SCRIPT_DIR/bmg_moe_gating_gemm_sycl.so"
ls -la bmg_moe_gating_gemm_sycl.so

echo ""
echo "Building bmg_moe_quant_grouped_gemm_fp8_sycl SYCL extension..."
icpx -shared -fPIC -O3 -DNDEBUG -std=c++17 \
    -DTORCH_EXTENSION_NAME=bmg_moe_quant_grouped_gemm_fp8_sycl \
    $SYCL_TLA_COMPILE_FLAGS \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    $SYCL_TLA_INCLUDES \
    09_bmg_moe_quant_grouped_gemm.cpp \
    $SYCL_TLA_LINK_FLAGS \
    -Xsycl-target-backend=spir64_gen "-device bmg-g21" \
    "${BMG_09_LINK_FLAGS[@]}" \
    -Xspirv-translator \
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    "${SYCL_TLA_RUNTIME_PATHS[@]}" \
    -L/lib64/stubs \
    -o bmg_moe_quant_grouped_gemm_fp8_sycl.so \
    $SYCL_TLA_LINK_LIBS

echo "Built: $SCRIPT_DIR/bmg_moe_quant_grouped_gemm_fp8_sycl.so"
ls -la bmg_moe_quant_grouped_gemm_fp8_sycl.so

echo ""
echo "Building bmg_moe_quant_grouped_gemm_int8_sycl SYCL extension..."
icpx -shared -fPIC -O3 -DNDEBUG -std=c++17 \
    -DTORCH_EXTENSION_NAME=bmg_moe_quant_grouped_gemm_int8_sycl \
    $SYCL_TLA_COMPILE_FLAGS \
    $TORCH_INCLUDES \
    -I"$PYTHON_INCLUDE" \
    $SYCL_TLA_INCLUDES \
    10_bmg_moe_quant_grouped_gemm.cpp \
    $SYCL_TLA_LINK_FLAGS \
    -Xsycl-target-backend=spir64_gen "-device bmg-g21" \
    "${BMG_10_LINK_FLAGS[@]}" \
    -Xspirv-translator \
    -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
    "${SYCL_TLA_RUNTIME_PATHS[@]}" \
    -L/lib64/stubs \
    -o bmg_moe_quant_grouped_gemm_int8_sycl.so \
    $SYCL_TLA_LINK_LIBS

echo "Built: $SCRIPT_DIR/bmg_moe_quant_grouped_gemm_int8_sycl.so"
ls -la bmg_moe_quant_grouped_gemm_int8_sycl.so
