cmake_minimum_required(VERSION 3.18)
project(flash_attn_xpu)

SET(CMAKE_C_COMPILER icx)
SET(CMAKE_CXX_COMPILER icpx)

find_package(Python3 COMPONENTS Interpreter Development.Module
                                 Development.SABIModule)

SET(SOURCE
    csrc/flash_attn/flash_api.cpp
    csrc/xpu/attn/attn_interface.cpp
    csrc/xpu/attn/xe_2/mla_dense_decode_xe2.cpp
    csrc/xpu/attn/xe_2/mla_sparse_prefill_xe2.cpp
    csrc/xpu/attn/xe_2/mla_sparse_decode_xe2.cpp
    csrc/xpu/attn/xe_2/mla_dense_prefill_xe2.cpp)

python3_add_library(flash_attn_xpu MODULE WITH_SOABI USE_SABI 3.12
    ${SOURCE}
)

# sycl-tla headers — search env var, sibling dir, then common dev path
if(DEFINED ENV{SYCL_TLA_DIR})
    set(_SYCL_TLA_ROOT "$ENV{SYCL_TLA_DIR}")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../sycl-tla")
    # sibling of xpu-perf repo root
    get_filename_component(_SYCL_TLA_ROOT
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../../sycl-tla" ABSOLUTE)
elseif(EXISTS "/root/software/src/sycl-tla")
    set(_SYCL_TLA_ROOT "/root/software/src/sycl-tla")
else()
    message(FATAL_ERROR
        "sycl-tla not found. Set SYCL_TLA_DIR env var or place sycl-tla as sibling of xpu-perf.")
endif()
message(STATUS "Using sycl-tla at: ${_SYCL_TLA_ROOT}")

SET(SYCL_TLA_INCLUDE_DIRS
    ${_SYCL_TLA_ROOT}/include
    ${_SYCL_TLA_ROOT}/tools/util/include
    ${_SYCL_TLA_ROOT}/applications
)

# Torch cmake
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)

target_link_libraries(flash_attn_xpu PRIVATE ${TORCH_LIBRARIES})
target_include_directories(flash_attn_xpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/xpu/attn/xe_2
    ${TORCH_INCLUDE_DIRS}
    ${SYCL_TLA_INCLUDE_DIRS}
)
target_compile_definitions(flash_attn_xpu PRIVATE
    -DVLLM_XPU_ENABLE_XE2
    "-DTORCH_EXTENSION_NAME=flash_attn_xpu"
)

# compile flags
list(APPEND SYCL_COMPILE_FLAGS "-fsycl")
list(APPEND SYCL_COMPILE_FLAGS "-fsycl-targets=spir64_gen")
list(APPEND SYCL_COMPILE_FLAGS "-O3")
list(APPEND SYCL_COMPILE_FLAGS "-DNDEBUG")
list(APPEND SYCL_COMPILE_FLAGS "-DCUTLASS_ENABLE_HEADERS_ONLY")
list(APPEND SYCL_COMPILE_FLAGS "-DCUTLASS_ENABLE_SYCL")
list(APPEND SYCL_COMPILE_FLAGS "-DSYCL_INTEL_TARGET")
list(APPEND SYCL_COMPILE_FLAGS "-DCUTLASS_VERSIONS_GENERATED")
list(APPEND SYCL_COMPILE_FLAGS "-ftemplate-backtrace-limit=0")
list(APPEND SYCL_COMPILE_FLAGS "-fdiagnostics-color=always")
target_compile_options(flash_attn_xpu PRIVATE ${SYCL_COMPILE_FLAGS})

# link flags
list(APPEND SYCL_LINK_FLAGS "-fsycl")
list(APPEND SYCL_LINK_FLAGS "-fsycl-targets=spir64_gen")
list(APPEND SYCL_LINK_FLAGS "-fsycl-max-parallel-link-jobs=16")
list(APPEND SYCL_LINK_FLAGS "-Xspirv-translator;-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate")
list(APPEND SYCL_LINK_FLAGS -Xsycl-target-backend=spir64_gen
    "-device pvc,bmg-g21-a0,bmg-g31-a0 -internal_options -cl-intel-256-GRF-per-thread")
target_link_options(flash_attn_xpu PRIVATE ${SYCL_LINK_FLAGS})
