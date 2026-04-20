/***************************************************************************************************
 * Sparse Prefill Tile Scheduler for XPU
 * Maps work-groups to (s_q, gqa_group, head_kv) for sparse attention prefill.
 * No Split-K: each WG handles one query's full topk range.
 * GQA groups are fused into the grid Z dimension for single-launch.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include <cute/tensor.hpp>

namespace cutlass::fmha::kernel {

using namespace cute;

// Grid: (V_tiles, 1, s_q * num_groups * h_kv)
struct SparsePrefillTileScheduler {
  struct Params {
    dim3 grid;
    FastDivmod divmod_num_groups_hkv;  // divides by (num_groups * h_kv)
    FastDivmod divmod_hkv;             // divides by h_kv
    int num_groups_;
  };

  bool valid_ = true;
  Params params;

  CUTLASS_DEVICE
  SparsePrefillTileScheduler(Params const& params) : params(params) {}

  template <class ProblemShape, class TileShape>
  static Params to_underlying_arguments(
      ProblemShape const& shape,
      KernelHardwareInfo hw_info,
      TileShape const& tile_shape,
      int num_groups) {
    dim3 grid(
        size(ceil_div(shape.head_size_vo, get<1>(tile_shape))),
        1,
        size(shape.s_q * num_groups * shape.num_heads_kv));
    return Params{
        grid,
        {num_groups * shape.num_heads_kv},
        {shape.num_heads_kv},
        num_groups};
  }

  template <int Num_SGs>
  static dim3 get_grid_shape(Params const& params) {
    return params.grid;
  }

  CUTLASS_DEVICE bool is_valid() { return valid_; }

  // Returns (blk_q=0, blk_v, head_kv, idx_sq, group_idx)
  CUTLASS_DEVICE auto get_block_coord() {
    int z = BlockIdxZ();
    int idx_sq, group_hkv;
    // z = idx_sq * (num_groups * h_kv) + group_idx * h_kv + head_kv
    params.divmod_num_groups_hkv(idx_sq, group_hkv, z);
    int group_idx, head_kv;
    params.divmod_hkv(group_idx, head_kv, group_hkv);
    return make_coord(0, BlockIdxX(), head_kv, idx_sq, group_idx);
  }

  CUTLASS_DEVICE SparsePrefillTileScheduler& operator++() {
    valid_ = false;
    return *this;
  }
};

}  // namespace cutlass::fmha::kernel
