/***************************************************************************************************
 * Dense Prefill Forward Mainloop for XPU (Q-token tiled)
 * Uses TiledCopy 2D block loads for Q, K, V (no SLM staging).
 * K and V are different layout views over the same KV cache memory:
 *   K: (seqlen_k, d_qk), stride (h_kv*d_qk, 1)  — row-major tokens
 *   V: (d_output, seqlen_k), stride (1, h_kv*d_qk) — transposed view
 * Both use the same underlying 2D surface for hardware block loads.
 * Per-element causal masking for multi-Q-token tiles.
 **************************************************************************************************/
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include <cute/tensor.hpp>
#include "flash_attention_v2/collective/fmha_fusion.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    int TILE_K_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,
    class TiledCopyK_ = void,
    class TiledCopyV_ = void>
struct DensePrefillFwdMainloop {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  static constexpr int TILE_K = TILE_K_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  //
  // Accumulator types
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;
  using FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>;
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  // Dense prefill arguments (minimal — KV tensors constructed by kernel)
  struct Arguments {
    ElementS const scale;
    bool is_causal;
  };

  using Params = Arguments;

  // No SLM needed — 2D block loads go directly to registers
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //
  DensePrefillFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void*) {
    constexpr double kLog2e = 1.4426950408889634074;
    return Arguments{
        static_cast<ElementS>(static_cast<double>(args.scale) * kLog2e),
        args.is_causal};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Main operator: processes multiple Q tokens against KV range via 2D block loads
  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,   // (actual_q_count, d_qk) — may be partial tile
      TensorK2D const& K_2D,   // (seqlen_k, d_qk) — K view of KV cache
      TensorV2D const& V_2D,   // (d_output, seqlen_k) — V view (transposed)
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int first_q_pos,          // position of first Q token in batch
      int actual_q_count,       // number of valid Q tokens (may be < M)
      int blk_k0,               // start K block index
      int blk_k1,               // end K block index (exclusive)
      int thr_id,
      int seq_len_k) {          // total seqlen_k for remainder masking
    using namespace sycl::ext::oneapi::this_work_item;

    // M dimension (compile-time Q tokens per WG tile)
    constexpr int kQTokens = get<0>(TileShapeQK{});

    // ---- Coordinate tensors ----
    // Pad Q identity to full tile size for local_tile (even if actual_q_count < kQTokens).
    // Hardware 2D block load zero-fills OOB rows.
    constexpr auto padded_d_v = get<1>(TileShapePV{}) * C<VTiles>{};
    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    Tensor cQ = make_identity_tensor(
        make_shape(C<kQTokens>{}, size<1>(Q_2D.shape())));               // padded (M, d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(
        make_shape(padded_d_v, size<0>(K_2D.shape())));           // (v,k) padded
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{})); // (q,k)

    // ---- Tile coordinate tensors ----
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});   // (q,d,D)
    Tensor gK = local_tile(
        cK, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _)); // (v,k,K)
    Tensor gV_split = local_tile(
        gV, TileShapePV{}, make_coord(_, _, 0), Step<X, _1, _1>{});  // (v,k,VV,K)

    // ---- Create TiledCopy objects from actual tensors ----
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D};
    TiledCopyV copy_v{V_2D};

    // ---- Create MMAs ----
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    // ---- Thread-level slices ----
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    // ---- Partition coordinate tensors for copy ----
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    // ---- Register fragments for MMA and copies ----
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    // ---- Prefetch objects ----
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D);

    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ---- Initial prefetches: Q (all D blocks) + K block 0 (all D blocks) ----
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }
    for (int D = 0; D < size<4>(pKgK); D++) {
      prefetch(prefetch_k, pKgK(_, _, _, blk_k0, D));
    }

    // ---- Initialize accumulators ----
    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len_k % get<1>(TileShapeQK{}) != 0);

    // Precompute causal block thresholds for multi-Q token tile
    constexpr int k_tile = get<1>(TileShapeQK{});
    int max_q_pos = first_q_pos + actual_q_count - 1;  // last Q token position

    if (params.is_causal) {
      // Skip K blocks fully masked for ALL Q tokens (based on last/highest Q pos)
      blk_k1 = cute::min(blk_k1, cute::ceil_div(max_q_pos + 1, k_tile));
    }
    // First block that might need partial causal masking (based on first/lowest Q pos)
    int blk_k_causal = params.is_causal ? (first_q_pos / k_tile) : blk_k1;

    // ---- Main loop over K blocks ----
    for (int K = blk_k0; K < blk_k1; K++) {

      // ---- GEMM 1: S = Q × K^T ----
      clear(tSrS);
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK(_, _, _, K, D), tKrK);
        reorder(tQrQ, tSrQ);
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      // V prefetch for GEMM 2
      prefetch(prefetch_v, pVgV(_, _, _, K));

      // Per-element causal masking: each Q token has a different causal boundary
      if (params.is_causal && K >= blk_k_causal) {
        // Create local identity tensor for tile-local (row, col) coordinates
        Tensor cP_local = make_identity_tensor(take<0, 2>(TileShapeQK{}));
        auto cS_coord = thr_mma_qk.partition_C(cP_local);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          int row_local = get<0>(cS_coord(i));  // 0..M-1 within tile
          int col_local = get<1>(cS_coord(i));  // 0..N-1 within tile
          int q_pos = first_q_pos + row_local;
          int k_pos = K * k_tile + col_local;
          if (k_pos > q_pos) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      // Remainder masking for last partial K block (column-wise, all rows share same bound)
      if (check_remainder_k && K == blk_k1 - 1) {
        int lane_id = get_sub_group().get_local_id()[0];
        FragSCol k_rem_mask;
        int k = K * k_tile + lane_id;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len_k) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      // Online softmax with fused scaling
      softmax(K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      // ---- GEMM 2: O += P × V, split over V dimension ----
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV(_, _, _, VV, K), tVrV);
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      // K prefetch for next iteration
      if (K + 1 < blk_k1) {
        for (int D = 0; D < size<4>(pKgK); D++) {
          prefetch(prefetch_k, pKgK(_, _, _, K + 1, D));
        }
      }
    }
  }

  // Online softmax with scale fused into exp2 (FMHAFwdMainloop style)
  CUTLASS_DEVICE
  void softmax(
      bool first_block,
      FragS& tS,
      FragSRow& tS_max,
      FragSRow& tS_sum,
      FragA& tA) {

    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), params.scale * tS_bmax(i));
    }

    // Scale and exponentiate: exp2(scale * S - max)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      tS(i) = sycl::native::exp2(
          params.scale * tS(i) - broadcast<0>(tS_max, tS, i));
    }

    // Rescale existing accumulators
    if (!first_block) {
      FragSRow rescale;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    // Update sums
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

}  // namespace cutlass::fmha::collective
