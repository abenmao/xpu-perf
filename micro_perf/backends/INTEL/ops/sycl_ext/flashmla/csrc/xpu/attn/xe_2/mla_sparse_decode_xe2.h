#pragma once

#include <sycl/sycl.hpp>

struct sparse_decode_args_t;

void cutlass_mla_sparse_decode_xe2(
    sycl::queue& queue,
    const sparse_decode_args_t& args);
