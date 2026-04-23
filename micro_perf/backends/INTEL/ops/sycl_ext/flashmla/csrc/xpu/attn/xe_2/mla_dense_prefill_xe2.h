#pragma once

#include <sycl/sycl.hpp>

struct dense_prefill_args_t;

// Dense prefill kernel entry point.
void cutlass_mla_dense_prefill_xe2(
    sycl::queue& queue,
    const dense_prefill_args_t& args);
