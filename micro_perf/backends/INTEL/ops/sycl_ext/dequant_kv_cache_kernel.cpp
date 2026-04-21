// dequant_kv_cache_kernel.cpp — SYCL fused int8→bf16 dequantization
//
// Fuses copy_(int8→bf16) + mul_(scale) into a single kernel,
// cutting memory traffic ~2x vs the two-kernel torch path.
//
// Same optimizations as store_kv_cache v3:
//   1. nd_range: one work-group per (batch, head, kv_pos) triple
//   2. vec8 loads/stores: 8 elements per transaction
//   3. Sub-group size 16: 16 threads × 8 elements = 128 = head_dim
//   4. __restrict__ + pre-computed strides

#include <sycl/sycl.hpp>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>

// --- Helper: bf16 (uint16_t) <-> fp32 via bit manipulation ---
static inline float bf16_to_fp32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f;
    __builtin_memcpy(&f, &bits, sizeof(float));
    return f;
}

static inline uint16_t fp32_to_bf16(float f) {
    // Round-to-nearest-even (matches PyTorch / hardware bf16 conversion)
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(float));
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

// --- Vec types for vectorized memory access ---
struct alignas(16) bf16x8_t { uint16_t d[8]; };
struct alignas(8)  int8x8_t { int8_t d[8]; };

constexpr int SUBGROUP_SIZE = 16;

// ============================================================
// Dequant: int8 cache → bf16 output (fused cast + scale)
//
// For each element:
//   output_bf16[b,h,t,d] = bf16( fp32(cache_int8[b,h,t,d]) * fp32(scale_bf16[h,d]) )
//
// Scale is passed as bf16 (pre-cast from fp32 in Python) to match
// torch's `.to(bfloat16)` truncation.
// ============================================================

void dequant_kv_cache_sycl(
    torch::Tensor k_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor v_cache,      // [batch, kv_head, max_kv_len, head_dim] int8
    torch::Tensor dequant_k,    // [batch, kv_head, max_kv_len, head_dim] bf16 (output)
    torch::Tensor dequant_v,    // [batch, kv_head, max_kv_len, head_dim] bf16 (output)
    torch::Tensor k_scale,      // [kv_head, head_dim] bf16
    torch::Tensor v_scale,      // [kv_head, head_dim] bf16
    int64_t batch_size,
    int64_t kv_len              // actual kv length to process (≤ max_kv_len)
) {
    const int kv_head = static_cast<int>(k_cache.size(1));
    const int max_kv_len_dim = static_cast<int>(k_cache.size(2));
    const int head_dim = static_cast<int>(k_cache.size(3));
    const int bs = static_cast<int>(batch_size);
    const int kv_len_i = static_cast<int>(kv_len);

    // Strides for [batch, kv_head, max_kv_len, head_dim] layout
    const int head_stride = max_kv_len_dim * head_dim;
    const int batch_stride = kv_head * head_stride;

    const int8_t* __restrict__ k_cache_ptr = k_cache.data_ptr<int8_t>();
    const int8_t* __restrict__ v_cache_ptr = v_cache.data_ptr<int8_t>();
    uint16_t* __restrict__ dk_ptr = reinterpret_cast<uint16_t*>(dequant_k.data_ptr());
    uint16_t* __restrict__ dv_ptr = reinterpret_cast<uint16_t*>(dequant_v.data_ptr());
    const uint16_t* __restrict__ ks_ptr = reinterpret_cast<const uint16_t*>(k_scale.data_ptr());
    const uint16_t* __restrict__ vs_ptr = reinterpret_cast<const uint16_t*>(v_scale.data_ptr());

    const int num_groups = bs * kv_head * kv_len_i;
    const int block_size = SUBGROUP_SIZE;

    auto& queue = c10::xpu::getCurrentXPUStream().queue();

    queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(num_groups * block_size, block_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SUBGROUP_SIZE)]] {
            const int group_idx = item.get_group(0);
            const int lid = item.get_local_id(0);

            // Decompose group_idx → (batch, head, kv_pos)
            const int batch_idx = group_idx / (kv_head * kv_len_i);
            const int rem = group_idx % (kv_head * kv_len_i);
            const int head_idx = rem / kv_len_i;
            const int kv_pos = rem % kv_len_i;

            // Element offset (same for int8 src and bf16 dst, in elements)
            const int offset = batch_idx * batch_stride
                             + head_idx * head_stride
                             + kv_pos * head_dim;

            const int8_t* __restrict__ k_src = k_cache_ptr + offset;
            const int8_t* __restrict__ v_src = v_cache_ptr + offset;
            uint16_t* __restrict__ k_dst = dk_ptr + offset;
            uint16_t* __restrict__ v_dst = dv_ptr + offset;

            // Scale: [kv_head, head_dim] in bf16
            const uint16_t* __restrict__ ks = ks_ptr + head_idx * head_dim;
            const uint16_t* __restrict__ vs = vs_ptr + head_idx * head_dim;

            // Vec8 path: only when head_dim is a multiple of 8 (alignment)
            constexpr int VEC = 8;
            if (head_dim % VEC == 0) {
                const int num_vec = head_dim / VEC;
                const auto* k_src_v8 = reinterpret_cast<const int8x8_t*>(k_src);
                const auto* v_src_v8 = reinterpret_cast<const int8x8_t*>(v_src);

                for (int vi = lid; vi < num_vec; vi += block_size) {
                    int8x8_t kv8 = k_src_v8[vi];
                    int8x8_t vv8 = v_src_v8[vi];
                    int base = vi * VEC;

                    bf16x8_t kout, vout;
                    #pragma unroll
                    for (int j = 0; j < 8; j++) {
                        float kf = static_cast<float>(kv8.d[j]) * bf16_to_fp32(ks[base + j]);
                        kout.d[j] = fp32_to_bf16(kf);

                        float vf = static_cast<float>(vv8.d[j]) * bf16_to_fp32(vs[base + j]);
                        vout.d[j] = fp32_to_bf16(vf);
                    }
                    reinterpret_cast<bf16x8_t*>(k_dst)[vi] = kout;
                    reinterpret_cast<bf16x8_t*>(v_dst)[vi] = vout;
                }
            } else {
                // Scalar fallback for unaligned head_dim
                for (int i = lid; i < head_dim; i += block_size) {
                    float kf = static_cast<float>(k_src[i]) * bf16_to_fp32(ks[i]);
                    k_dst[i] = fp32_to_bf16(kf);

                    float vf = static_cast<float>(v_src[i]) * bf16_to_fp32(vs[i]);
                    v_dst[i] = fp32_to_bf16(vf);
                }
            }
        });
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequant_kv_cache", &dequant_kv_cache_sycl,
          "Fused int8->bf16 dequantization from linear KV cache (SYCL)");
}
