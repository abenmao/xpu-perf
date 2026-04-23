/***************************************************************************************************
 * Sparse Decode Forward Mainloop for XPU
 * SLM-staged sparse token gathering + TiledMMA QK/PV computation.
 * Follows the same code structure as DecodeFwdMainloop in chunk_prefill_mainloop.hpp.
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

// FP8 dequant helpers (device-side)
CUTLASS_DEVICE float fp8_e4m3fn_to_float_dev(uint8_t val) {
  uint32_t sign = static_cast<uint32_t>(val >> 7);
  uint32_t biased_exp = static_cast<uint32_t>((val >> 3) & 0xF);
  uint32_t mantissa = static_cast<uint32_t>(val & 0x7);
  if (biased_exp == 0 && mantissa == 0) return sign ? -0.0f : 0.0f;
  if (biased_exp == 0) {
    float fval = static_cast<float>(mantissa) * (1.0f / 512.0f);
    return sign ? -fval : fval;
  }
  uint32_t f32_bits = (sign << 31) | ((biased_exp + 120) << 23) | (mantissa << 20);
  float result;
  __builtin_memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

CUTLASS_DEVICE float e8m0fnu_to_float_dev(uint8_t val) {
  if (val == 0) return 0.0f;
  uint32_t f32_bits = static_cast<uint32_t>(val) << 23;
  float result;
  __builtin_memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    int TILE_K_,
    class TensorQ_,
    class TiledCopyQ_>
struct SparseFwdMainloop {
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
  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;

  //
  // Accumulator types (same as DecodeFwdMainloop)
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

  // Sparse-specific arguments
  struct Arguments {
    ElementS const scale;
    const int* indices;          // [batch, s_q, topk]
    int topk;
    const int* topk_length;     // [batch] or nullptr
    void const* kv_ptr;         // KV cache base pointer
    int h_kv;
    int d_qk;
    int d_v;
    int total_tokens;
    int fp8_model;              // 0=BF16, 1=V32, 2=MODEL1
    int bytes_per_token;
    int page_block_size;
  };

  using Params = Arguments;

  // SLM storage for gathered KV tokens
  static constexpr int MAX_D_QK = 576;
  struct SharedStorage {
    sycl::ext::oneapi::bfloat16 kv_slm[TILE_K * MAX_D_QK];
  };

  Params params;

  //
  // Methods
  //
  SparseFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void*) {
    constexpr double kLog2e = 1.4426950408889634074;
    // Need to create a copy with modified scale
    return Arguments{
        static_cast<ElementS>(static_cast<double>(args.scale) * kLog2e),
        args.indices,
        args.topk,
        args.topk_length,
        args.kv_ptr,
        args.h_kv,
        args.d_qk,
        args.d_v,
        args.total_tokens,
        args.fp8_model,
        args.bytes_per_token,
        args.page_block_size};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  // Gather one KV token from global memory to SLM (handles BF16 + FP8)
  CUTLASS_DEVICE void gather_token_to_slm(
      sycl::ext::oneapi::bfloat16* slm,
      int slm_offset,
      int token_idx,
      int hkv_idx,
      int local_tid,
      int wg_size) const {
    using bf16 = sycl::ext::oneapi::bfloat16;
    const int d_qk = params.d_qk;

    if (params.fp8_model == 0) {
      auto* kv_bf16 = reinterpret_cast<const bf16*>(params.kv_ptr);
      for (int d = local_tid; d < d_qk; d += wg_size) {
        slm[slm_offset + d] = kv_bf16[
            static_cast<long>(token_idx) * params.h_kv * d_qk + hkv_idx * d_qk + d];
      }
    } else if (params.fp8_model == 1) {
      auto* raw = reinterpret_cast<const uint8_t*>(params.kv_ptr);
      const uint8_t* tok = raw + static_cast<long>(token_idx) * 656;
      for (int d = local_tid; d < d_qk; d += wg_size) {
        if (d < 512) {
          float scale = reinterpret_cast<const float*>(tok + 512)[d >> 7];
          slm[slm_offset + d] = static_cast<bf16>(
              fp8_e4m3fn_to_float_dev(tok[d]) * scale);
        } else {
          slm[slm_offset + d] = reinterpret_cast<const bf16*>(tok + 528)[d - 512];
        }
      }
    } else {
      auto* raw = reinterpret_cast<const uint8_t*>(params.kv_ptr);
      int block = token_idx / params.page_block_size;
      int tok = token_idx % params.page_block_size;
      const uint8_t* blk = raw + static_cast<long>(block) * params.bytes_per_token;
      for (int d = local_tid; d < d_qk; d += wg_size) {
        if (d < 448) {
          float scale = e8m0fnu_to_float_dev(
              blk[params.page_block_size * 576 + tok * 8 + d / 64]);
          slm[slm_offset + d] = static_cast<bf16>(
              fp8_e4m3fn_to_float_dev(blk[tok * 576 + d]) * scale);
        } else {
          slm[slm_offset + d] = reinterpret_cast<const bf16*>(
              blk + tok * 576 + 448)[d - 448];
        }
      }
    }
  }

  template <typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,
      FragA& tArA,
      FragARow& tA_max,
      FragARow& tA_sum,
      QVCoord blk_qv,
      int const& idx_b,
      int const& idx_sq,
      int const& head_kv,
      int const& head_group_q,
      int tk_start,
      int tk_end,
      int thr_id,
      SharedStorage& shared_storage) {
    using namespace sycl::ext::oneapi::this_work_item;
    using bf16 = sycl::ext::oneapi::bfloat16;

    const int d_qk = params.d_qk;
    const int d_v = params.d_v;
    const int wg_size = SGPerWG::value * intel::sg_size;
    const int local_tid = thr_id;

    // Get sparse indices for this (batch, seq) pair
    const int* idx_row = params.indices +
        (idx_b * 1 + idx_sq) * params.topk;

    bf16* kv_slm = shared_storage.kv_slm;

    // Proxy tensors for MMA partitioning
    Tensor cQ = make_identity_tensor(Q_2D.shape());
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));

    // WG tile for Q
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});

    // Create TiledCopy for Q from global memory
    TiledCopyQ copy_q{Q_2D};

    // Create MMAs
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    // Q copy/MMA fragments
    auto tQgQ = thr_copy_q.partition_S(gQ);
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    // S (QK result) fragment
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);

    // P→A reorder (softmax result → PV input)
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    // Initialize accumulators
    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    // K fragment from SLM: we create an SLM tensor and partition it with MMA
    constexpr int d_chunk_qk = get<2>(TileShapeQK{});
    constexpr int v_chunk_pv = get<1>(TileShapePV{});

    // Number of SGs splitting the N dimension of QK GEMM (= K dim of PV GEMM)
    constexpr int SGs_in_N = get<1>(shape(SubgroupLayoutQK{}));
    constexpr int N_per_sg = TILE_K / SGs_in_N;
    constexpr int K_per_sg_pv = TILE_K / SGs_in_N;

    // Create MMA B fragment for QK GEMM (from identity tensor for type/layout)
    auto tSrK = make_fragment_like(thr_mma_qk.partition_sg_fragment_B(
        make_identity_tensor(take<1, 3>(TileShapeQK{}))));

    // Precompute SG/lane info (constant per thread)
    auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    int lane = sg.get_local_id()[0];
    int sg_id_local = thr_id / intel::sg_size;
    int sg_id_for_N = sg_id_local % SGs_in_N;

    // Main loop over token blocks
    for (int tk_block = tk_start; tk_block < tk_end; tk_block += TILE_K) {
      int actual_tile_k = cute::min(TILE_K, tk_end - tk_block);

      // Phase 1: Cooperative gather of TILE_K tokens into SLM
      // Use MAX_D_QK stride to match QK/PV GEMM read patterns
      for (int t = 0; t < actual_tile_k; t++) {
        int token_idx = idx_row[tk_block + t];
        bool valid = (token_idx >= 0 && token_idx < params.total_tokens);
        if (valid) {
          gather_token_to_slm(kv_slm, t * MAX_D_QK, token_idx, head_kv, local_tid, wg_size);
        } else {
          for (int d = local_tid; d < d_qk; d += wg_size) {
            kv_slm[t * MAX_D_QK + d] = bf16(0.0f);
          }
        }
      }
      for (int t = actual_tile_k; t < TILE_K; t++) {
        for (int d = local_tid; d < d_qk; d += wg_size) {
          kv_slm[t * MAX_D_QK + d] = bf16(0.0f);
        }
      }
      sycl::group_barrier(get_work_group<3>());

      // Phase 2: QK GEMM — S[q_packed, TILE_K] = Q[q_packed, d_qk] × K^T[d_qk, TILE_K]
      clear(tSrS);
      int d_chunks = d_qk / d_chunk_qk;
      for (int D = 0; D < d_chunks; D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        reorder(tQrQ, tSrQ);

        // Fill B fragment from SLM using BLayout wi_interleave mapping.
        // DPAS B operand layout interleaves (N, K) positions across threads:
        //   flat = lane + 16 * v_atom
        //   N_in_atom = (flat / 2) % 16,  K_in_atom = (flat / 32) * 2 + flat % 2
        {
          int sg_base = sg_id_for_N * N_per_sg;
          int d_offset = D * d_chunk_qk;
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < (int)tSrK.size(); v++) {
            int v_atom = v % 16;
            int k_iter = v / 16;
            int flat = lane + 16 * v_atom;
            int N_in_atom = (flat >> 1) & 15;
            int K_in_atom = (flat >> 5) * 2 + (flat & 1);
            int kv_token = sg_base + N_in_atom;
            int dim = d_offset + k_iter * 16 + K_in_atom;
            bf16 val = (kv_token < TILE_K && dim < d_qk) ?
                kv_slm[kv_token * MAX_D_QK + dim] : bf16(0.0f);
            tSrK(v) = static_cast<typename decltype(tSrK)::value_type>(val);
          }
        }
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      // Handle remainder d_qk dimensions
      int d_rem = d_qk % d_chunk_qk;
      if (d_rem > 0) {
        int D = d_chunks;
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        reorder(tQrQ, tSrQ);
        {
          int sg_base = sg_id_for_N * N_per_sg;
          int d_offset = D * d_chunk_qk;
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < (int)tSrK.size(); v++) {
            int v_atom = v % 16;
            int k_iter = v / 16;
            int flat = lane + 16 * v_atom;
            int N_in_atom = (flat >> 1) & 15;
            int K_in_atom = (flat >> 5) * 2 + (flat & 1);
            int kv_token = sg_base + N_in_atom;
            int dim = d_offset + k_iter * 16 + K_in_atom;
            bf16 val = (kv_token < TILE_K && dim < d_qk) ?
                kv_slm[kv_token * MAX_D_QK + dim] : bf16(0.0f);
            tSrK(v) = static_cast<typename decltype(tSrK)::value_type>(val);
          }
        }
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      // Apply scale
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tSrS.size(); i++) {
        tSrS(i) *= params.scale;
      }

      // Remainder masking: mask positions beyond actual_tile_k
      if (actual_tile_k < TILE_K) {
        FragSCol k_rem_mask;
        int k_start_in_tile = sg_id_for_N * N_per_sg;
        int k = k_start_in_tile + lane;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++) {
          k_rem_mask(i) = (k < actual_tile_k) ?
              ElementS(0) : ElementS(-INFINITY);
          k += intel::sg_size;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) += broadcast<1>(k_rem_mask, tSrS, i);
        }
      }

      // Mask invalid tokens (negative indices)
      {
        FragSCol validity_mask;
        int k_start_in_tile = sg_id_for_N * N_per_sg;
        int k = k_start_in_tile + lane;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < validity_mask.size(); i++) {
          int global_tk = tk_block + k;
          bool valid = (k < actual_tile_k) &&
                       (global_tk < params.topk) &&
                       (idx_row[global_tk] >= 0) &&
                       (idx_row[global_tk] < params.total_tokens);
          validity_mask(i) = valid ? ElementS(0) : ElementS(-INFINITY);
          k += intel::sg_size;
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), tSrS(i) + broadcast<1>(validity_mask, tSrS, i));
        }
      }

      // Online softmax
      bool first_block = (tk_block == tk_start);
      softmax(first_block, tSrS, tA_max, tA_sum, tArA);

      // Reorder S→P for PV GEMM
      reorder(tSrS, tArP);

      // Phase 3: PV GEMM — O[q_packed, d_v] += P[q_packed, TILE_K] × V[TILE_K, d_v]
      // SGLayoutPV K=SGs_in_N: each SG handles K_per_sg_pv kv_tokens
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        int v_base = VV * v_chunk_pv;

        auto tArV = make_fragment_like(thr_mma_pv.partition_sg_fragment_B(
            make_identity_tensor(take<1, 3>(TileShapePV{}))));

        // Fill B fragment from SLM using BLayout wi_interleave mapping.
        // For PV: B[N=v_dim, K=kv_token]
        {
          int kv_token_base = sg_id_local * K_per_sg_pv;

          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < (int)tArV.size(); v++) {
            int v_atom = v % 16;
            int n_iter = v / 16;
            int flat = lane + 16 * v_atom;
            int N_in_atom = (flat >> 1) & 15;
            int K_in_atom = (flat >> 5) * 2 + (flat & 1);
            int v_dim = v_base + n_iter * 16 + N_in_atom;
            int kv_token = kv_token_base + K_in_atom;
            bf16 val = (kv_token < TILE_K && v_dim < d_v) ?
                kv_slm[kv_token * MAX_D_QK + v_dim] : bf16(0.0f);
            tArV(v) = static_cast<typename decltype(tArV)::value_type>(val);
          }
        }

        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      // Barrier before next iteration overwrites SLM
      sycl::group_barrier(get_work_group<3>());
    }
  }

  // Online softmax (identical to DecodeFwdMainloop)
  CUTLASS_DEVICE
  void softmax(
      bool first_block,
      FragS& tS,
      FragSRow& tS_max,
      FragSRow& tS_sum,
      FragA& tA) {
    // Row-wise max for this block
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    // Update global max
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), tS_bmax(i));
    }

    // Exponentiate: exp2(S - max)
    // Guard against -inf - (-inf) = NaN when all tokens are masked.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      auto diff = tS(i) - broadcast<0>(tS_max, tS, i);
      tS(i) = (diff != diff) ? ElementS(0) : sycl::native::exp2(diff);
    }

    // Rescale existing accumulators
    if (!first_block) {
      FragSRow rescale;
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        auto rdiff = tS_prev_max(i) - tS_max(i);
        rescale(i) = (rdiff != rdiff) ? ElementS(0) : sycl::native::exp2(rdiff);
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
