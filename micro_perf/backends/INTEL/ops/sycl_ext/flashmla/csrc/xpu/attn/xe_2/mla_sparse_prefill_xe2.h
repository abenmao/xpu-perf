#pragma once

#include <sycl/sycl.hpp>

struct sparse_prefill_args_t;

// Sparse prefill kernel entry point.
// Internally converts prefill args (s_q as batch) and delegates
// to the shared decode tiled kernel.  This indirection allows
// future replacement with a prefill-specific kernel for optimization.
void cutlass_mla_sparse_prefill_xe2(
    sycl::queue& queue,
    const sparse_prefill_args_t& args);
