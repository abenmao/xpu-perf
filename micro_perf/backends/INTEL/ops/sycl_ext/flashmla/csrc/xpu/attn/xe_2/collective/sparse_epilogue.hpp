/***************************************************************************************************
 * Sparse Attention Epilogue for XPU
 * Handles output normalization (softmax divide), TiledCopy output write,
 * cross-SG reduction (ReduceK), Split-K exp_sums/max_logits storage, and attn_sink.
 * Closely follows DecodeFwdEpilogue in chunk_prefill_epilogue.hpp.
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include <cute/tensor.hpp>

#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/collective/copy_block_slm.hpp"

namespace cutlass::fmha::collective {

using namespace cute;

template <
    class CollectiveMainloop_,
    class TileShapeO_,
    class TensorO_,
    class TensorLSE_,
    class TiledCopyO_ = void,
    bool Sink_ = false>
class SparseFwdEpilogue {
 public:
  //
  // Type Aliases (matching DecodeFwdEpilogue exactly)
  //
  using TiledMMAPV = typename CollectiveMainloop_::TiledMMAPV;
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  using TileShapeO = TileShapeO_;
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAPV::ThrLayoutVMNK{}))));

  using TensorO = TensorO_;
  using TensorO2D =
      decltype(TensorO_{}(append<rank_v<TensorO_>>(make_coord(_, _), 0)));
  using ElementO = typename TensorO_::value_type;

  using TensorLSE = TensorLSE_;
  using TensorLSE2D = conditional_t<
      is_void_v<TensorLSE_>,
      void,
      decltype(TensorLSE_{}(append<rank_v<TensorLSE_>>(make_coord(_, _), 0)))>;
  using ElementLSE = conditional_t<
      is_void_v<TensorLSE_>,
      void,
      typename TensorLSE_::value_type>;

  using FragA = typename CollectiveMainloop_::FragA;
  using FragARow = typename CollectiveMainloop_::FragARow;
  using ElementA = typename FragA::value_type;

  static constexpr bool Sink = Sink_;
  using ElementSink = typename CollectiveMainloop_::TensorQ::value_type;

  // Cross-SG reduction (same as dense)
  using ReduceK = decltype(size<3>(typename TiledMMAPV::ThrLayoutVMNK{}));

  using SGTileShapeA = decltype(atuple_coshape(FragA{}.tv_layout()));
  using ReduceSGQ = decltype(cute::gcd(get<0>(SGTileShapeA{}), ReduceK{}));

  static auto reduce_sg_v_helper() {
    constexpr auto v_total_sg = get<1>(SGTileShapeA{}) / intel::_SGSize{};
    constexpr auto v_avail_sg = ReduceK{} / ReduceSGQ{};
    return Int<
        (v_total_sg > v_avail_sg) ? cute::gcd(v_total_sg, v_avail_sg)
                                  : v_total_sg>{};
  }
  using ReduceSGV = decltype(reduce_sg_v_helper());
  using ReduceSGLayout =
      decltype(make_identity_layout(Shape<ReduceSGQ, ReduceSGV>{}));
  using SGTileShapeO =
      decltype(shape_div(take<0, 2>(SGTileShapeA{}), shape(ReduceSGLayout{})));

  using ReduceFragA = decltype(make_subgroup_tensor<ElementA>(
      make_layout(select<1, 0>(SGTileShapeO{}), Stride<E<1>, E<0>>{})));
  using ReduceFragARow = decltype(reduce<1>(ReduceFragA{}, sycl::plus<void>{}));

  static auto default_tiled_copy_O_helper() {
    if constexpr (ReduceK{} == _1{})
      return make_block_2d_copy_D(TiledMMAPV{}, TensorO2D{});
    else
      return make_block_2d_copy_D_subtiled(
          TiledMMAPV{},
          ReduceFragA{}.tv_layout(),
          ReduceSGLayout{},
          TensorO2D{});
  }

  using DefaultTiledCopyO = decltype(default_tiled_copy_O_helper());
  using TiledCopyO =
      conditional_t<is_void_v<TiledCopyO_>, DefaultTiledCopyO, TiledCopyO_>;

  struct Arguments {};
  struct Params {};

  // Shared memory storage (same as dense)
  using AlignedSGTileA_Q =
      C<((size<0>(SGTileShapeA{}) + intel::sg_size - 1) / intel::sg_size) *
        intel::sg_size>;

  struct SharedStorageNone {};
  struct SharedStorageReduceK {
    cute::array<ElementA, size(SGTileShapeA{}) * SGPerWG{}> a_data;
    cute::array<ElementA, AlignedSGTileA_Q{} * SGPerWG{}> a_sum_data,
        a_max_data;
  };

  using SharedStorage = conditional_t<
      (ReduceK{} > _1{}),
      SharedStorageReduceK,
      SharedStorageNone>;

 private:
  SharedStorage& shared;

 public:
  static constexpr Params
  to_underlying_arguments(Arguments const&, void*) { return {}; }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) { return true; }

  CUTLASS_HOST_DEVICE
  SparseFwdEpilogue(Params const&, SharedStorage& shared_) : shared(shared_) {}

  // Non-SplitK: direct output
  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int thr_id) {

    auto [rA, rA_max_unused, rA_sum, active] =
        reduce_A(tArA, tA_max, tA_sum, thr_id);

    if (!active) return;

    // Softmax normalization: divide by sum (guard against sum=0 for all-invalid splits)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      rA_sum(i) = (rA_sum(i) == ElementA(0)) ? ElementA(0) : ElementA(1) / rA_sum(i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);

    // Tile output and write (use TileShapeO for identity to handle partial Q tiles)
    Tensor cO = make_identity_tensor(TileShapeO{});
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);

    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);
    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // SplitK: store partial results
  template <typename QVCoord, class TensorSink>
  CUTLASS_DEVICE void operator()(
      TensorO2D const& O,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int thr_id,
      const TensorLSE2D& exp_sums,
      const TensorLSE2D& max_logits,
      int idx_kv_split,
      int head_group_q,
      TensorSink& tSink,
      int num_kv_splits) {

    int sg_id = thr_id / intel::sg_size;

    // Attn sink: add exp(sink) contribution to the softmax denominator for
    // the first split. The sink is a virtual token with no V contribution — it
    // only inflates the denominator, damping the attention output.
    //
    // We do NOT update tA_max here (unlike mainloop rescaling) because:
    //   (1) max_logits returned to Python should reflect token-only scores,
    //   (2) if sink >> max, exp2(sink-max) may overflow to inf, but that is
    //       fine: normalization gives 1/inf=0, yielding O_accum=0. The reduce
    //       kernel guards against 0*inf=NaN.
    //
    // In DPAS, each WI holds ALL 8 head values in fragments. All WIs in SG 0
    // must apply sink identically to maintain fragment coherence.
    if constexpr (Sink) {
      constexpr double kLog2e = 1.4426950408889634074;
      if (idx_kv_split == 0 && sg_id == 0) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < cute::min(int(tA_sum.size()), head_group_q); i++) {
          tA_sum(i) += sycl::native::exp2(
              static_cast<ElementA>(tSink(i) * kLog2e) - tA_max(i));
        }
      }
    }

    auto [rA, rA_max, rA_sum, active] = reduce_A(tArA, tA_max, tA_sum, thr_id);

    // Store exp_sums and max_logits for Split-K reduce
    if (thr_id < head_group_q) {
      exp_sums(thr_id, idx_kv_split) = rA_sum(0);
      max_logits(thr_id, idx_kv_split) = rA_max(0);
    }

    if (!active) return;

    // Softmax normalization (guard against sum=0 for all-invalid splits)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA_sum.size(); i++)
      rA_sum(i) = (rA_sum(i) == ElementA(0)) ? ElementA(0) : ElementA(1) / rA_sum(i);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < rA.size(); i++)
      rA(i) *= broadcast<0>(rA_sum, rA, i);

    // Write output (use TileShapeO for identity to handle partial Q tiles)
    Tensor cO = make_identity_tensor(TileShapeO{});
    Tensor gO = local_tile(cO, TileShapeO{}, blk_qv);

    TiledCopyO copy_o{O};
    auto thr_copy_o = copy_o.get_slice(thr_id);
    auto tOrO = thr_copy_o.partition_sg_fragment_S(gO);
    auto tOgO = thr_copy_o.partition_D(gO);

    reorder(rA, tOrO);
    copy(copy_o, tOrO, tOgO);
  }

  // Cross-SG k-block reduction (copied faithfully from DecodeFwdEpilogue)
  template <typename FragA_t, typename FragARow_t>
  CUTLASS_DEVICE decltype(auto) reduce_A(
      FragA_t& tArA,
      FragARow_t& tA_max,
      FragARow_t& tA_sum,
      int thr_id) {

    using namespace sycl::ext::oneapi::this_work_item;

    if constexpr (ReduceK{} == _1{}) {
      ReduceFragARow rA_max;
      return std::make_tuple(tArA, rA_max, tA_sum, true);
    } else {
      // Identify A tile ID and k block for this subgroup
      auto thr_vak = group<1, 3>(TiledMMAPV{}.get_thr_layout_vmnk())
                         .get_flat_coord(assert_uniform(thr_id));
      auto a_tile = get<1>(thr_vak);
      auto k_blk = get<2>(thr_vak);

      // Set up SLM tensors (same layout as dense epilogue)
      auto shape_A =
          append(append(SGTileShapeA{}, ReduceK{}), SGPerWG{} / ReduceK{});
      auto shape_A_row = make_shape(
          get<0>(SGTileShapeO{}),
          shape(ReduceSGLayout{}),
          ReduceK{},
          SGPerWG{} / ReduceK{});

      // Physical layouts, with subtile modes broken out
      auto sA_layout = group<2, 4>(flat_divide(
          make_ordered_layout(shape_A, Step<_1, _0, _2, _3>{}),
          SGTileShapeO{}));
      auto sA_row_stride = make_stride(
          _1{},
          make_stride(get<0>(shape_A_row), _0{}),
          AlignedSGTileA_Q{},
          AlignedSGTileA_Q{} * ReduceK{});
      auto sA_row_layout = make_layout(shape_A_row, sA_row_stride);

      // Coordinate layouts, with subtile modes broken out
      auto basis2 = make_basis_like(SGTileShapeO{});
      auto sA_coords = make_layout(
          append(SGTileShapeO{}, shape(ReduceSGLayout{})),
          append(basis2, product_each(zip(SGTileShapeO{}, basis2))));

      auto sA = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_data),
          sA_layout);
      auto sA_max = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_max_data),
          sA_row_layout);
      auto sA_sum = make_tensor(
          make_smem_ptr<ElementA>(&shared.a_sum_data),
          sA_row_layout);

      // Write contributions to SLM
      copy_block_r2s(tA_max, sA_max(_, _, k_blk, a_tile));
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);
      copy_block_r2s(tA_sum, sA_sum(_, _, k_blk, a_tile));
      copy_block_r2s(tArA, sA(_, _, _, k_blk, a_tile), sA_coords);

      bool active = (k_blk < size(ReduceSGLayout{})) ||
                    (ReduceK{} == size(ReduceSGLayout{}));

      // Wait for maxima, signal other data available
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);
      barrier_arrive(ScopeWorkgroup, SemanticsRelease | SemanticsWGMemory);

      ReduceFragA rA;
      ReduceFragARow rA_sum_out, rA_max_out, rA_kmax[ReduceK{}];

      if (active) {
        // Read A_max back from SLM and reduce
        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          copy_block_s2r(sA_max(_, k_blk, kr, a_tile), rA_kmax[kr]);
        }

        rA_max_out = rA_kmax[0];
        for (int kr = 1; kr < ReduceK{}; kr++)
          cute::transform(rA_max_out, rA_kmax[kr], rA_max_out, cute::max_fn{});

        // Calculate scale factors
        for (int kr = 0; kr < ReduceK{}; kr++) {
          cute::transform(
              rA_max_out, rA_kmax[kr], rA_kmax[kr], [](auto gmax, auto kmax) {
                return sycl::native::exp2(kmax - gmax);
              });
        }
      }

      // Wait for A/A_sum data
      barrier_wait(ScopeWorkgroup, SemanticsAcquire | SemanticsWGMemory);

      if (active) {
        // Read A/A_sum back from SLM, align scaling, and reduce
        clear(rA_sum_out);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragARow rA_sum_read;
          copy_block_s2r(sA_sum(_, k_blk, kr, a_tile), rA_sum_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_sum_read.size(); i++) {
            rA_sum_out(i) += rA_sum_read(i) * rA_kmax[kr](i);
          }
        }

        clear(rA);

        CUTLASS_PRAGMA_UNROLL
        for (int kr = 0; kr < ReduceK{}; kr++) {
          ReduceFragA rA_read;
          copy_block_s2r(
              sA(_, _, k_blk, kr, a_tile), sA_coords(_, _, 0), rA_read);

          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < rA_read.size(); i++) {
            rA(i) += rA_read(i) * broadcast<0>(rA_kmax[kr], rA, i);
          }
        }
      }
      return std::make_tuple(rA, rA_max_out, rA_sum_out, active);
    }
  }
};

}  // namespace cutlass::fmha::collective
