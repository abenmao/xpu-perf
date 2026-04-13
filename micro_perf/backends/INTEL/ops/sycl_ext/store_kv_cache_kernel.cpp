// store_kv_cache_kernel.cpp — v3: vec8 + sub-group optimized
//
// Optimizations:
//   1. nd_range: one work-group per (token, head) pair
//   2. vec8 loads/stores: 8 bf16 = 16 bytes per transaction (2x cache line utilization)
//   3. Sub-group size 16: 16 threads × 8 elements = 128 = head_dim in one pass
//   4. __restrict__ + pre-computed strides
//   5. reqd_sub_group_size attribute for compiler to use native SIMD width

#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

// --- Helper: bf16 (uint16_t) -> fp32 via bit shift ---
static inline float bf16_to_fp32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

// --- Vec types for vectorized memory access ---
struct alignas(16) bf16x8_t { uint16_t d[8]; };
struct alignas(8)  int8x8_t { int8_t d[8]; };
struct alignas(8)  bf16x4_t { uint16_t x, y, z, w; };
struct alignas(4)  int8x4_t { int8_t x, y, z, w; };

// Sub-group size for Intel Xe2 (BMG/B60): 16 threads
// 16 threads × 8 elements/thread = 128 = head_dim
constexpr int SUBGROUP_SIZE = 16;

// ============================================================
// INT8 store: bf16 -> fp32 -> scale -> clamp -> round -> int8
// ============================================================

void store_kv_cache_int8_sycl(
    torch::Tensor key,          // [num_tokens, kv_head, head_dim] bf16
    torch::Tensor value,        // [num_tokens, kv_head, head_dim] bf16
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor k_scale,      // [kv_head, head_dim] fp32
    torch::Tensor v_scale,      // [kv_head, head_dim] fp32
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int kv_head = static_cast<int>(key.size(1));
    const int head_dim = static_cast<int>(key.size(2));
    const int max_kv_len = static_cast<int>(k_cache.size(2));
    const int num_tokens = static_cast<int>(key.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int src_token_stride = kv_head * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ key_ptr = reinterpret_cast<const uint16_t*>(key.data_ptr());
    const uint16_t* __restrict__ val_ptr = reinterpret_cast<const uint16_t*>(value.data_ptr());
    int8_t* __restrict__ k_cache_ptr = k_cache.data_ptr<int8_t>();
    int8_t* __restrict__ v_cache_ptr = v_cache.data_ptr<int8_t>();
    const float* __restrict__ k_scale_ptr = k_scale.data_ptr<float>();
    const float* __restrict__ v_scale_ptr = v_scale.data_ptr<float>();

    const int num_groups = num_tokens * kv_head;
    // Work-group = one sub-group = SUBGROUP_SIZE threads
    // Each thread handles vec8 elements in a strided loop
    const int block_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * block_size, block_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_idx = item.get_group(0);
            const int lid = item.get_local_id(0);

            const int token_idx = group_idx / kv_head;
            const int head_idx = group_idx % kv_head;
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const uint16_t* __restrict__ k_src = key_ptr + token_idx * src_token_stride + head_idx * head_dim;
            const uint16_t* __restrict__ v_src = val_ptr + token_idx * src_token_stride + head_idx * head_dim;
            const float* __restrict__ ks = k_scale_ptr + head_idx * head_dim;
            const float* __restrict__ vs = v_scale_ptr + head_idx * head_dim;

            const int cache_offset = batch_idx * cache_batch_stride
                                   + head_idx * cache_head_stride
                                   + (cache_start_i + local_q_pos) * head_dim;
            int8_t* __restrict__ k_dst = k_cache_ptr + cache_offset;
            int8_t* __restrict__ v_dst = v_cache_ptr + cache_offset;

            // Vec8 path: only when head_dim is a multiple of 8 (alignment)
            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int num_vec = head_dim / VEC;
                const auto* k_src_v8 = reinterpret_cast<const bf16x8_t*>(k_src);
                const auto* v_src_v8 = reinterpret_cast<const bf16x8_t*>(v_src);

                for (int vi = lid; vi < num_vec; vi += block_size) {
                    bf16x8_t kv8 = k_src_v8[vi];
                    bf16x8_t vv8 = v_src_v8[vi];
                    int base = vi * VEC;

                    int8x8_t kout, vout;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        float kf = bf16_to_fp32(kv8.d[j]) * ks[base + j];
                        kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                        kout.d[j] = static_cast<int8_t>(kf);

                        float vf = bf16_to_fp32(vv8.d[j]) * vs[base + j];
                        vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                        vout.d[j] = static_cast<int8_t>(vf);
                    }
                    reinterpret_cast<int8x8_t*>(k_dst)[vi] = kout;
                    reinterpret_cast<int8x8_t*>(v_dst)[vi] = vout;
                }
            } else {
                // Scalar fallback for unaligned head_dim
                for (int i = lid; i < head_dim; i += block_size) {
                    float kf = bf16_to_fp32(k_src[i]) * ks[i];
                    kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                    k_dst[i] = static_cast<int8_t>(kf);

                    float vf = bf16_to_fp32(v_src[i]) * vs[i];
                    vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                    v_dst[i] = static_cast<int8_t>(vf);
                }
            }
        });
    });
}

// ============================================================
// BF16 store: bf16 -> bf16 permute+copy (vec8 + sub-group)
// ============================================================

void store_kv_cache_bf16_sycl(
    torch::Tensor key,          // [num_tokens, kv_head, head_dim] bf16
    torch::Tensor value,        // [num_tokens, kv_head, head_dim] bf16
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] bf16
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] bf16
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int kv_head = static_cast<int>(key.size(1));
    const int head_dim = static_cast<int>(key.size(2));
    const int max_kv_len = static_cast<int>(k_cache.size(2));
    const int num_tokens = static_cast<int>(key.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int src_token_stride = kv_head * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ key_ptr = reinterpret_cast<const uint16_t*>(key.data_ptr());
    const uint16_t* __restrict__ val_ptr = reinterpret_cast<const uint16_t*>(value.data_ptr());
    uint16_t* __restrict__ k_cache_ptr = reinterpret_cast<uint16_t*>(k_cache.data_ptr());
    uint16_t* __restrict__ v_cache_ptr = reinterpret_cast<uint16_t*>(v_cache.data_ptr());

    const int num_groups = num_tokens * kv_head;
    const int block_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * block_size, block_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_idx = item.get_group(0);
            const int lid = item.get_local_id(0);

            const int token_idx = group_idx / kv_head;
            const int head_idx = group_idx % kv_head;
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const uint16_t* __restrict__ k_src = key_ptr + token_idx * src_token_stride + head_idx * head_dim;
            const uint16_t* __restrict__ v_src = val_ptr + token_idx * src_token_stride + head_idx * head_dim;

            const int cache_offset = batch_idx * cache_batch_stride
                                   + head_idx * cache_head_stride
                                   + (cache_start_i + local_q_pos) * head_dim;
            uint16_t* __restrict__ k_dst = k_cache_ptr + cache_offset;
            uint16_t* __restrict__ v_dst = v_cache_ptr + cache_offset;

            // Vec8: 8 x uint16_t = 16 bytes per load/store
            // Only use vec8 when head_dim is a multiple of 8 (alignment)
            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int num_vec = head_dim / VEC;
                const auto* k_src_v8 = reinterpret_cast<const bf16x8_t*>(k_src);
                const auto* v_src_v8 = reinterpret_cast<const bf16x8_t*>(v_src);
                auto* k_dst_v8 = reinterpret_cast<bf16x8_t*>(k_dst);
                auto* v_dst_v8 = reinterpret_cast<bf16x8_t*>(v_dst);

                for (int vi = lid; vi < num_vec; vi += block_size) {
                    k_dst_v8[vi] = k_src_v8[vi];
                    v_dst_v8[vi] = v_src_v8[vi];
                }
            } else {
                // Scalar fallback for unaligned head_dim
                for (int i = lid; i < head_dim; i += block_size) {
                    k_dst[i] = k_src[i];
                    v_dst[i] = v_src[i];
                }
            }
        });
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("store_kv_cache_int8", &store_kv_cache_int8_sycl,
          "Fused bf16->int8 quantized store to linear KV cache (SYCL v3)");
    m.def("store_kv_cache_bf16", &store_kv_cache_bf16_sycl,
          "Fused bf16->bf16 store to linear KV cache (SYCL v3)");
}
