#pragma once

#include <sycl/sycl.hpp>
#include "csrc/xpu/attn/xe_2/fmha_utils.hpp"

struct paged_decode_args_t;

void cutlass_mla_dense_decode_xe2(
    sycl::queue& queue,
    CutlassDType cuType,
    const paged_decode_args_t& args);
