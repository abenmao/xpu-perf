// store_kv_cache_kernel.cpp — v5: per-token work-group + packed_qkv direct read
//
// v5 changes vs v4:
//   - Accept packed_qkv directly instead of pre-sliced key/value tensors
//   - Kernel reads K/V via head offsets into packed_qkv — zero Python-side copy
//   - Source stride = total_heads × head_dim (skips Q heads between tokens)
//
// Optimizations:
//   1. nd_range: one work-group per token, kv_head sub-groups per group
//   2. vec8 loads/stores: 8 bf16 = 16 bytes per transaction
//   3. Sub-group size 16: 16 threads × 8 elements = 128 = head_dim in one pass
//   4. __restrict__ + pre-computed strides
//   5. reqd_sub_group_size attribute for compiler to use native SIMD width
//   6. No slice/contiguous in Python — kernel reads packed_qkv with stride

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
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor k_scale,      // [kv_head, head_dim] fp32
    torch::Tensor v_scale,      // [kv_head, head_dim] fp32
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int max_kv_len = static_cast<int>(k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int n = kv_head * head_dim;  // KV elements per token
    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    int8_t* __restrict__ k_cache_ptr = k_cache.data_ptr<int8_t>();
    int8_t* __restrict__ v_cache_ptr = v_cache.data_ptr<int8_t>();
    const float* __restrict__ k_scale_ptr = k_scale.data_ptr<float>();
    const float* __restrict__ v_scale_ptr = v_scale.data_ptr<float>();

    const int num_groups = num_tokens;
    const int block_size = kv_head * SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * block_size, block_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int token_idx = item.get_group(0);
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int src_base = token_idx * src_token_stride;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = n / VEC;
                for (int vi = lid; vi < total_vec; vi += block_size) {
                    const int flat_elem = vi * VEC;
                    const int head_idx = flat_elem / head_dim;
                    const int dim_in_head = flat_elem % head_dim;

                    const int k_src_off = src_base + k_offset + flat_elem;
                    const int v_src_off = src_base + v_offset + flat_elem;
                    const int dst_off = batch_idx * cache_batch_stride
                                      + head_idx * cache_head_stride
                                      + (cache_start_i + local_q_pos) * head_dim
                                      + dim_in_head;

                    bf16x8_t kv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_off);
                    bf16x8_t vv8 = *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_off);

                    const float* __restrict__ ks = k_scale_ptr + head_idx * head_dim + dim_in_head;
                    const float* __restrict__ vs = v_scale_ptr + head_idx * head_dim + dim_in_head;

                    int8x8_t kout, vout;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        float kf = bf16_to_fp32(kv8.d[j]) * ks[j];
                        kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                        kout.d[j] = static_cast<int8_t>(kf);

                        float vf = bf16_to_fp32(vv8.d[j]) * vs[j];
                        vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                        vout.d[j] = static_cast<int8_t>(vf);
                    }
                    *reinterpret_cast<int8x8_t*>(k_cache_ptr + dst_off) = kout;
                    *reinterpret_cast<int8x8_t*>(v_cache_ptr + dst_off) = vout;
                }
            } else {
                for (int i = lid; i < n; i += block_size) {
                    const int head_idx = i / head_dim;
                    const int dim_in_head = i % head_dim;

                    const int k_src_off = src_base + k_offset + i;
                    const int v_src_off = src_base + v_offset + i;
                    const int dst_off = batch_idx * cache_batch_stride
                                      + head_idx * cache_head_stride
                                      + (cache_start_i + local_q_pos) * head_dim
                                      + dim_in_head;

                    const float ks = k_scale_ptr[head_idx * head_dim + dim_in_head];
                    const float vs = v_scale_ptr[head_idx * head_dim + dim_in_head];

                    float kf = bf16_to_fp32(qkv_ptr[k_src_off]) * ks;
                    kf = sycl::clamp(sycl::rint(kf), -127.0f, 127.0f);
                    k_cache_ptr[dst_off] = static_cast<int8_t>(kf);

                    float vf = bf16_to_fp32(qkv_ptr[v_src_off]) * vs;
                    vf = sycl::clamp(sycl::rint(vf), -127.0f, 127.0f);
                    v_cache_ptr[dst_off] = static_cast<int8_t>(vf);
                }
            }
        });
    });
}

// ============================================================
// BF16 store: bf16 -> bf16 permute+copy (vec8 + sub-group)
// ============================================================

void store_kv_cache_bf16_sycl(
    torch::Tensor packed_qkv,   // [num_tokens, total_heads, head_dim] bf16
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] bf16
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] bf16
    int64_t k_head_start,
    int64_t kv_head_num,
    int64_t batch_size,
    int64_t q_len,
    int64_t cache_start
) {
    const int total_heads = static_cast<int>(packed_qkv.size(1));
    const int head_dim = static_cast<int>(packed_qkv.size(2));
    const int kv_head = static_cast<int>(kv_head_num);
    const int max_kv_len = static_cast<int>(k_cache.size(2));
    const int num_tokens = static_cast<int>(packed_qkv.size(0));
    const int q_len_i = static_cast<int>(q_len);
    const int cache_start_i = static_cast<int>(cache_start);

    const int n = kv_head * head_dim;  // KV elements per token
    const int src_token_stride = total_heads * head_dim;
    const int k_offset = static_cast<int>(k_head_start) * head_dim;
    const int v_offset = (static_cast<int>(k_head_start) + kv_head) * head_dim;
    const int cache_head_stride = max_kv_len * head_dim;
    const int cache_batch_stride = kv_head * cache_head_stride;

    const uint16_t* __restrict__ qkv_ptr = reinterpret_cast<const uint16_t*>(packed_qkv.data_ptr());
    uint16_t* __restrict__ k_cache_ptr = reinterpret_cast<uint16_t*>(k_cache.data_ptr());
    uint16_t* __restrict__ v_cache_ptr = reinterpret_cast<uint16_t*>(v_cache.data_ptr());

    const int num_groups = num_tokens;
    const int block_size = kv_head * SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * block_size, block_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int token_idx = item.get_group(0);
            const int lid = item.get_local_id(0);
            const int batch_idx = token_idx / q_len_i;
            const int local_q_pos = token_idx % q_len_i;

            const int src_base = token_idx * src_token_stride;

            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int total_vec = n / VEC;
                for (int vi = lid; vi < total_vec; vi += block_size) {
                    const int flat_elem = vi * VEC;
                    const int head_idx = flat_elem / head_dim;
                    const int dim_in_head = flat_elem % head_dim;

                    const int k_src_off = src_base + k_offset + flat_elem;
                    const int v_src_off = src_base + v_offset + flat_elem;
                    const int dst_off = batch_idx * cache_batch_stride
                                      + head_idx * cache_head_stride
                                      + (cache_start_i + local_q_pos) * head_dim
                                      + dim_in_head;

                    *reinterpret_cast<bf16x8_t*>(k_cache_ptr + dst_off) =
                        *reinterpret_cast<const bf16x8_t*>(qkv_ptr + k_src_off);
                    *reinterpret_cast<bf16x8_t*>(v_cache_ptr + dst_off) =
                        *reinterpret_cast<const bf16x8_t*>(qkv_ptr + v_src_off);
                }
            } else {
                for (int i = lid; i < n; i += block_size) {
                    const int head_idx = i / head_dim;
                    const int dim_in_head = i % head_dim;

                    const int k_src_off = src_base + k_offset + i;
                    const int v_src_off = src_base + v_offset + i;
                    const int dst_off = batch_idx * cache_batch_stride
                                      + head_idx * cache_head_stride
                                      + (cache_start_i + local_q_pos) * head_dim
                                      + dim_in_head;

                    k_cache_ptr[dst_off] = qkv_ptr[k_src_off];
                    v_cache_ptr[dst_off] = qkv_ptr[v_src_off];
                }
            }
        });
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("store_kv_cache_int8", &store_kv_cache_int8_sycl,
          "Fused bf16->int8 quantized store to linear KV cache (SYCL v5)");
    m.def("store_kv_cache_bf16", &store_kv_cache_bf16_sycl,
          "Fused bf16->bf16 store to linear KV cache (SYCL v5)");
}
