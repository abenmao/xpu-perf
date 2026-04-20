// Sparse prefill kernel — dedicated kernel with no Split-K.
// Single kernel launch handles all s_q × num_groups × h_kv work items.
// Direct bf16 output (no float accumulators, no reduce kernel).

#include "mla_sparse_prefill_xe2.h"
#include "sparse_prefill_tiled.hpp"

using namespace cute;

// Same tile shapes as sparse decode (shared mainloop/epilogue)
using sparse_mla_prefill_ShapeQK = Shape<_8, _64, _64>;
using sparse_mla_prefill_ShapePV = Shape<_8, _32, _64>;
using sparse_mla_prefill_ShapeOut = Shape<_8, cute::Int<576>>;
using sparse_mla_prefill_SGLayoutQK = Layout<Shape<_1, _4, _1>>;

void cutlass_mla_sparse_prefill_xe2(
    sycl::queue& queue,
    const sparse_prefill_args_t& args) {

  // Prefill always uses Sink=false. Attn sink is applied host-side
  // so that the returned lse/max_logits remain token-only values.
  SparsePrefillConfig<
      sparse_mla_prefill_ShapeQK,
      sparse_mla_prefill_ShapePV,
      sparse_mla_prefill_ShapeOut,
      sparse_mla_prefill_SGLayoutQK,
      void,         // SubgroupLayoutPV (auto)
      1,            // PipelineStages
      false,        // Sink = false (always)
      64,           // TILE_K
      bfloat16_t,
      bfloat16_t    // ElementO = bf16 (direct output)
  >::kernel_dispatch(queue, args);
}
