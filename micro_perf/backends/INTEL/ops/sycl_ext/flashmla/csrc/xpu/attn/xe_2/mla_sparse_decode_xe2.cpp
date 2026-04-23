#include "mla_sparse_decode_xe2.h"
#include "sparse_decode_tiled.hpp"

using namespace cute;

// Sparse decode policy for MLA: d_qk=576, q_packed=8, TILE_K=64
// Matches dense decode tile config exactly (4 SGs, page_size=64).
// Device SLM = 128KB, TILE_K=64 needs 64*576*2 = 72KB < 128KB.
using sparse_mla_decode_ShapeQK = Shape<_8, _64, _64>;
using sparse_mla_decode_ShapePV = Shape<_8, _32, _64>;
using sparse_mla_decode_ShapeOut = Shape<_8, cute::Int<576>>;
using sparse_mla_decode_SGLayoutQK = Layout<Shape<_1, _4, _1>>;

void cutlass_mla_sparse_decode_xe2(
    sycl::queue& queue,
    const sparse_decode_args_t& args) {

  if (args.attn_sink) {
    // With attn_sink (Sink=true)
    SparseDecodeConfig<
        sparse_mla_decode_ShapeQK,
        sparse_mla_decode_ShapePV,
        sparse_mla_decode_ShapeOut,
        sparse_mla_decode_SGLayoutQK,
        void,  // SubgroupLayoutPV (auto)
        1,     // PipelineStages
        true,  // Sink
        64,    // TILE_K
        bfloat16_t,
        float>::kernel_dispatch(queue, args);
  } else {
    // Without attn_sink (Sink=false)
    SparseDecodeConfig<
        sparse_mla_decode_ShapeQK,
        sparse_mla_decode_ShapePV,
        sparse_mla_decode_ShapeOut,
        sparse_mla_decode_SGLayoutQK,
        void,
        1,
        false,
        64,
        bfloat16_t,
        float>::kernel_dispatch(queue, args);
  }
}
