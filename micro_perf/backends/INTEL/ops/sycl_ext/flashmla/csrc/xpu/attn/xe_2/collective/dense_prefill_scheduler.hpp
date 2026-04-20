/***************************************************************************************************
 * Dense Prefill Tile Scheduler for XPU (Q-token tiled)
 * Each WG handles ONE h_q head across kQTokens consecutive Q tokens.
 * Q tiles are assigned per-batch: each batch's Q tokens are independently tiled.
 * Grid Z = total_q_tiles × h_q (overestimated, invalid tiles early-exit).
 * Variable-length sequences via cu_seqlens_q / cu_seqlens_k.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cute/tensor.hpp>

namespace cutlass::fmha::kernel {

using namespace cute;

// Grid: (V_tiles, 1, max_q_tiles * h_q)
struct DensePrefillTileScheduler {
  struct Params {
    dim3 grid;
    int kQTokens;                      // M dimension = Q tokens per WG
    FastDivmod divmod_h_q;             // divides by h_q
    int h_q_per_kv;                    // h_q / h_kv
    const int* cu_seqlens_q;           // [batch+1] int32
    const int* cu_seqlens_k;           // [batch+1] int32
    int batch_size;
    int total_q;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  DensePrefillTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      int num_groups,
      const int* cu_seqlens_q,
      const int* cu_seqlens_k,
      int batch_size) {
    int kQTokens = get<0>(tile_shape);  // M from TileShapeO
    int h_q = shape.num_heads_q;
    int h_kv = shape.num_heads_kv;

    // Upper bound on total Q tiles across all batches:
    // Each batch boundary can add at most 1 extra tile.
    int max_q_tiles = (shape.total_q + kQTokens - 1) / kQTokens + batch_size;

    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),  // V tiles
        1,
        max_q_tiles * h_q);                                     // Q tiles × heads
    return Params{
        grid,
        kQTokens,
        {h_q},
        h_q / h_kv,
        cu_seqlens_q,
        cu_seqlens_k,
        batch_size,
        shape.total_q};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE bool is_valid() { return valid_; }

  // Returns (blk_v, head_q, head_kv, q_abs_start, actual_q_count,
  //          batch_q_start, k_start, k_end)
  CUTLASS_DEVICE auto get_block_coord() {
    int z = BlockIdxZ();
    int linear_tile_idx, head_q;
    params.divmod_h_q(linear_tile_idx, head_q, z);

    int head_kv = head_q / params.h_q_per_kv;

    // Walk through batches to find which batch this tile belongs to
    int running_tiles = 0;
    for (int b = 0; b < params.batch_size; b++) {
      int batch_q_start = params.cu_seqlens_q[b];
      int batch_q_end = params.cu_seqlens_q[b + 1];
      int seqlen_q = batch_q_end - batch_q_start;
      int tiles_in_batch = (seqlen_q + params.kQTokens - 1) / params.kQTokens;

      if (linear_tile_idx < running_tiles + tiles_in_batch) {
        int within_batch_tile = linear_tile_idx - running_tiles;
        int q_start_in_batch = within_batch_tile * params.kQTokens;
        int q_abs_start = batch_q_start + q_start_in_batch;
        int actual_q_count = cute::min(
            q_start_in_batch + params.kQTokens, seqlen_q) - q_start_in_batch;

        int k_start = params.cu_seqlens_k[b];
        int k_end = params.cu_seqlens_k[b + 1];

        return make_coord(
            int(BlockIdxX()),   // blk_v
            head_q,
            head_kv,
            q_abs_start,
            actual_q_count,
            batch_q_start,
            k_start,
            k_end);
      }
      running_tiles += tiles_in_batch;
    }

    // Invalid tile (grid overestimate)
    valid_ = false;
    return make_coord(0, 0, 0, 0, 0, 0, 0, 0);
  }

  CUTLASS_DEVICE DensePrefillTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
