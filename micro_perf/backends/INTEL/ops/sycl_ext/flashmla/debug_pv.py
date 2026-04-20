import torch
torch.ops.load_library("flash_attn_xpu.abi3.so")

batch, h_q, d_qk, d_v, topk = 1, 8, 576, 512, 64
h_kv, total_tokens = 1, topk

# q: [batch, seqlen_q=1, num_heads_q, head_dim_k]
Q = torch.randn(batch, 1, h_q, d_qk, dtype=torch.bfloat16, device='xpu')

# kcache: [num_blocks, page_block_size=1, num_heads_k, head_dim_k]
kv_cache = torch.zeros(total_tokens, 1, h_kv, d_qk, dtype=torch.bfloat16, device='xpu')

# V[:,0]=1, rest=0
kv_cache[:, :, :, 0] = 1.0

# indices: [batch, seqlen_q=1, topk]
sparse_indices = torch.arange(topk, dtype=torch.int32, device='xpu').unsqueeze(0).unsqueeze(0)
sm_scale = d_qk ** -0.55

result = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    Q, kv_cache, sparse_indices, d_v, sm_scale, False, None, None)
O = result[0]

print(f"V[:,0]=1: O[0,0,0,:8] = {O[0,0,0,:8].tolist()}")
print(f"O[0,0,0,0] should be 1.0, got {O[0,0,0,0].item():.6f}")
print(f"O[0,0,0,16] should be 0.0, got {O[0,0,0,16].item():.6f}")
