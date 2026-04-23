#include "mla_dense_decode_xe2.h"
#include "paged_decode.hpp"

using namespace cute;

// MLA policy: head_dim_k=576, page_size=64, q_packed=8
// 576 = 9 * 64, so QK GEMM tiles 64 wide iterate 9 times over head_dim
// PV GEMM produces 576-dim output (caller slices to head_dim_v=512)
// VTiles = 576 / 32 = 18
using mla_decode_policy_q8_h576_p64 =
    decode_policy_qpacked_head<_8, cute::Int<576>, _64>;

void cutlass_mla_dense_decode_xe2(
    sycl::queue& queue,
    CutlassDType cuType,
    const paged_decode_args_t& args) {
  if (args.is_causal) {
    decode_policy_dispatch_impl<
        mla_decode_policy_q8_h576_p64, true, false, false>(
        queue, cuType, args);
  } else {
    decode_policy_dispatch_impl<
        mla_decode_policy_q8_h576_p64, false, false, false>(
        queue, cuType, args);
  }
}
