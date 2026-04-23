/***************************************************************************************************
 * Sparse Decode Tile Schedulers for XPU
 * Maps work-groups to (batch, head, kv_split) for sparse attention decode.
 * Follows DecodeTileScheduler in chunk_prefill_scheduler.hpp.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cute/tensor.hpp>

namespace cutlass::fmha::kernel {

using namespace cute;

// Tile scheduler for sparse decode main kernel
// Grid: (V_tiles, 1, batch * num_heads_kv * num_kv_splits)
struct SparseDecodeTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    FastDivmod divmod_batch;
    int num_kv_splits_;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  SparseDecodeTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      const int& num_kv_splits = 1) {
    int num_head = shape.num_heads_kv;
    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),
        1,
        size(shape.batch * num_head * num_kv_splits));
    return Params{
        grid,
        {num_head},
        {shape.batch * num_head},
        num_kv_splits};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE bool is_valid() { return valid_; }

  // Returns (blk_q=0, blk_v, head_kv, idx_b, idx_kv_split)
  CUTLASS_DEVICE auto get_block_coord() {
    int idx_kv_split = BlockIdxZ();
    int head, idx_b;
    // z = idx_b * num_heads_kv * num_kv_splits + head * num_kv_splits + idx_kv_split
    // divmod_batch divides by (batch * num_heads_kv) to extract split
    params.divmod_batch(idx_kv_split, idx_b, idx_kv_split);
    // divmod_num_heads divides by num_heads_kv to extract head & batch
    params.divmod_num_heads(idx_b, head, idx_b);
    return make_coord(0, BlockIdxX(), head, idx_b, idx_kv_split);
  }

  CUTLASS_DEVICE SparseDecodeTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

// Tile scheduler for sparse decode reduce kernel
// Grid: (s_q, num_heads_q, batch)
struct SparseReduceSplitKTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_heads;
    int num_kv_splits;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  SparseReduceSplitKTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      const int& num_kv_splits = 1) {
    dim3 grid(shape.seq_len_qo, shape.num_heads_q, shape.batch);
    return Params{grid, {shape.num_heads_q}, num_kv_splits};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE bool is_valid() { return valid_; }

  // Returns (seq_idx, head_q, idx_b)
  CUTLASS_DEVICE auto get_block_coord() {
    return make_coord(BlockIdxX(), BlockIdxY(), BlockIdxZ());
  }

  CUTLASS_DEVICE SparseReduceSplitKTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
