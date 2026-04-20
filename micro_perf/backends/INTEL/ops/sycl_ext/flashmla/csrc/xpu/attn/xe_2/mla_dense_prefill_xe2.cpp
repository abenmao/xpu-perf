// Dense prefill kernel — dedicated kernel for MLA dense attention prefill.
// Single kernel launch handles all total_q × num_groups × h_kv work items.
// Direct bf16 output (no float accumulators, no reduce kernel).

#include "mla_dense_prefill_xe2.h"
#include "mla_dense_prefill_tiled.hpp"

using namespace cute;

// Q-token tiled: M=64 (64 Q tokens per WG, each WG handles one h_q head).
// 8 SGs split M → each SG handles 8 Q tokens.
// Compute intensity: 64 FLOP/byte (vs 8 with old M=8 GQA packing).
using dense_mla_prefill_ShapeQK = Shape<_64, _64, _32>;
using dense_mla_prefill_ShapePV = Shape<_64, _32, _64>;
using dense_mla_prefill_ShapeOut = Shape<_64, cute::Int<576>>;
using dense_mla_prefill_SGLayoutQK = Layout<Shape<_8, _1, _1>>;

void cutlass_mla_dense_prefill_xe2(
    sycl::queue& queue,
    const dense_prefill_args_t& args) {

  DensePrefillConfig<
      dense_mla_prefill_ShapeQK,
      dense_mla_prefill_ShapePV,
      dense_mla_prefill_ShapeOut,
      dense_mla_prefill_SGLayoutQK,
      void,         // SubgroupLayoutPV (auto)
      1,            // PipelineStages
      false,        // Sink = false (always)
      32,           // TILE_K (D-loop step size)
      bfloat16_t,
      bfloat16_t    // ElementO = bf16 (direct output)
  >::kernel_dispatch(queue, args);
}
