"""Minimal diagnostic test for sparse decode TiledMMA kernel."""
import torch
torch.ops.load_library("flash_attn_xpu.abi3.so")

torch.manual_seed(42)
device = "xpu"
dtype = torch.bfloat16

# Minimal config
batch, h_q, d_qk, d_v = 1, 8, 576, 512
topk = 64  # exactly 1 tile (TILE_K=64)
total_tokens = 256
sm_scale = d_qk ** (-0.55)  # match test's convention

# Create Q: [batch, 1, h_q, d_qk]
q = (torch.randn(batch, 1, h_q, d_qk, device=device, dtype=dtype) / 10).clamp(-1, 1)

# Create KV cache: [num_blocks, block_size, h_kv=1, d_qk] for BF16 path
block_size = 64
num_blocks = total_tokens // block_size
kv = (torch.randn(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype) / 10).clamp(-1, 1)

# Indices: [batch, 1, topk] — first 64 tokens (first block)
indices = torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0).unsqueeze(0)

# Reference: Q @ K^T * scale, softmax, attn @ V[:d_v]
q_ref = q[0, 0].float()  # [h_q, d_qk]
k_flat = kv.reshape(-1, 1, d_qk)  # [total_tokens, 1, d_qk]
k_ref = k_flat[indices[0, 0].long(), 0, :].float()  # [topk, d_qk]
scores = torch.matmul(q_ref, k_ref.t()) * sm_scale  # [h_q, topk]
attn = torch.softmax(scores, dim=-1)  # [h_q, topk]
v_ref = k_ref[:, :d_v]  # [topk, d_v]
out_ref = torch.matmul(attn, v_ref).to(dtype)  # [h_q, d_v]
lse_ref = torch.logsumexp(scores, dim=-1)  # [h_q]

# Run kernel (BF16 path: is_fp8_kvcache=False)
result = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv, indices, d_v, sm_scale, False, None, None)
kern_out = result[0][0, 0]  # [h_q, d_v]
kern_lse = result[1][0, :, 0]  # [h_q]

print(f"=== BF16 Sparse Decode Diagnostic ===")
print(f"Config: batch={batch}, h_q={h_q}, d_qk={d_qk}, d_v={d_v}, topk={topk}")

# LSE comparison
lse_diff = (lse_ref - kern_lse.float()).abs()
print(f"\nLSE max diff: {lse_diff.max():.6f}")
print(f"LSE: ref={lse_ref[:4].tolist()}")
print(f"LSE: kern={kern_lse[:4].float().tolist()}")

# Output comparison
cos_sim = torch.nn.functional.cosine_similarity(
    out_ref.float().flatten().unsqueeze(0),
    kern_out.float().flatten().unsqueeze(0))
diff = (out_ref.float() - kern_out.float()).abs()
print(f"\nOutput cos_sim: {cos_sim.item():.6f}")
print(f"Output max abs diff: {diff.max():.6f}")
print(f"Output mean abs diff: {diff.mean():.6f}")

# Per-head
for h in range(min(h_q, 8)):
    ref_h = out_ref[h].float()
    kern_h = kern_out[h].float()
    cos_h = torch.nn.functional.cosine_similarity(ref_h.unsqueeze(0), kern_h.unsqueeze(0))
    diff_h = (ref_h - kern_h).abs()
    print(f"Head {h}: cos_sim={cos_h.item():.6f}, max_diff={diff_h.max():.6f}, ref_norm={ref_h.norm():.4f}, kern_norm={kern_h.norm():.4f}")

# First few values
for h in range(min(2, h_q)):
    print(f"\nHead {h} ref:  {out_ref[h, :8].float().tolist()}")
    print(f"Head {h} kern: {kern_out[h, :8].float().tolist()}")

# Test 2: V[t,d] = d (same for all tokens) → output O[h,d] = d*sum(P) = d
print(f"\n=== V=dim test (V[t,d]=d for all t) ===")
kv_dim = torch.zeros(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype)
for d in range(d_v):
    kv_dim[:, :, :, d] = float(d) / d_v  # scale to avoid overflow
result_dim = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv_dim, indices, d_v, sm_scale, False, None, None)
kern_dim = result_dim[0][0, 0]  # [h_q, d_v]
expected_dim = torch.arange(d_v, device=device, dtype=dtype).float() / d_v
# Expected: O[h,d] = (d/d_v) * sum(P[h,t]) = d/d_v since sum(P)=1
diff_dim = (kern_dim[0].float() - expected_dim).abs()
print(f"V=dim: max diff: {diff_dim.max():.6f}")
print(f"V=dim: head 0 first 8: {kern_dim[0, :8].float().tolist()}")
print(f"V=dim: expected first 8: {expected_dim[:8].tolist()}")
# Check if the output follows the d pattern (monotonically increasing)
is_monotonic = all(kern_dim[0, i].float() <= kern_dim[0, i+1].float() for i in range(d_v-1))
print(f"V=dim: output monotonic? {is_monotonic}")

# Test 3: V[t,d] = t (same for all dims) → output O[h,d] = sum(P[h,t]*t) (same for all d)
print(f"\n=== V=token test (V[t,d]=t for all d) ===")
kv_tok = torch.zeros(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype)
for t in range(topk):
    block_idx = t // block_size
    tok_in_block = t % block_size
    kv_tok[block_idx, tok_in_block, :, :d_v] = float(t) / topk
result_tok = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv_tok, indices, d_v, sm_scale, False, None, None)
kern_tok = result_tok[0][0, 0]  # [h_q, d_v]
# All dims should have same value for each head
per_head_std = kern_tok.float().std(dim=-1)
print(f"V=token: per-head std (should be ~0): {per_head_std.tolist()}")
print(f"V=token: head 0 first 8: {kern_tok[0, :8].float().tolist()}")

# Test 4: V = identity-like: V[t,d] = 1 if d==t else 0 (for first topk dims)
# → O[h,d] = P[h,d] for d < topk, 0 otherwise
print(f"\n=== V=identity test ===")
kv_id = torch.zeros(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype)
for t in range(topk):
    block_idx = t // block_size
    tok_in_block = t % block_size
    if t < d_v:
        kv_id[block_idx, tok_in_block, :, t] = 1.0
# Reference: compute softmax weights
q_ref = q[0, 0].float()
k_id_flat = kv_id.reshape(-1, 1, d_qk)
k_ref = k_id_flat[indices[0, 0].long(), 0, :].float()
scores_id = torch.matmul(q_ref, k_ref.t()) * sm_scale
softmax_id = torch.softmax(scores_id, dim=-1)
result_id = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv_id, indices, d_v, sm_scale, False, None, None)
kern_id = result_id[0][0, 0]  # [h_q, d_v]
# O[h,d] should = P[h,d] for d < topk
cos_id = torch.nn.functional.cosine_similarity(
    kern_id[:, :topk].float().flatten().unsqueeze(0),
    softmax_id.flatten().unsqueeze(0))
print(f"V=identity: cos_sim(O[:,:topk], P) = {cos_id.item():.6f}")
print(f"V=identity: head 0 O[:8]: {kern_id[0, :8].float().tolist()}")
print(f"V=identity: head 0 P[:8]: {softmax_id[0, :8].tolist()}")
# Test 5: Single-dimension test: V[t,0]=1, V[t,d>0]=0
# Expected: O[h,0] = 1.0, O[h,d>0] = 0.0
print(f"\n=== Single-dim V test (V[:,0]=1, rest=0) ===")
kv_single = torch.zeros(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype)
kv_single[:, :, :, 0] = 1.0
result_s = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv_single, indices, d_v, sm_scale, False, None, None)
kern_s = result_s[0][0, 0]
print(f"Single-dim: O[0,:8] = {kern_s[0, :8].float().tolist()}")
print(f"Single-dim: O[0,0] should be 1.0, got {kern_s[0, 0].float().item():.6f}")
print(f"Single-dim: O[0,1] should be 0.0, got {kern_s[0, 1].float().item():.6f}")
print(f"Single-dim: O[0,16] = {kern_s[0, 16].float().item():.6f}")

# Test 6: V[:,1]=1, rest=0
print(f"\n=== V[:,1]=1 test ===")
kv_d1 = torch.zeros(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype)
kv_d1[:, :, :, 1] = 1.0
result_d1 = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv_d1, indices, d_v, sm_scale, False, None, None)
kern_d1 = result_d1[0][0, 0]
print(f"V[:,1]=1: O[0,:8] = {kern_d1[0, :8].float().tolist()}")
print(f"V[:,1]=1: O[0,0] should be 0, got {kern_d1[0, 0].float().item():.6f}")
print(f"V[:,1]=1: O[0,1] should be 1, got {kern_d1[0, 1].float().item():.6f}")

# Test 7: V[:,16]=1
print(f"\n=== V[:,16]=1 test ===")
kv_d16 = torch.zeros(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype)
kv_d16[:, :, :, 16] = 1.0
result_d16 = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv_d16, indices, d_v, sm_scale, False, None, None)
kern_d16 = result_d16[0][0, 0]
print(f"V[:,16]=1: O[0,:20] = {kern_d16[0, :20].float().tolist()}")
print(f"V[:,16]=1: O[0,16] should be 1, got {kern_d16[0, 16].float().item():.6f}")

# Test 8: V[:,32]=1 (second VTile)
print(f"\n=== V[:,32]=1 test ===")
kv_d32 = torch.zeros(num_blocks, block_size, 1, d_qk, device=device, dtype=dtype)
kv_d32[:, :, :, 32] = 1.0
result_d32 = torch.ops.flash_attn_xpu.sparse_mla_decode_fwd(
    q, kv_d32, indices, d_v, sm_scale, False, None, None)
kern_d32 = result_d32[0][0, 0]
print(f"V[:,32]=1: O[0,28:40] = {kern_d32[0, 28:40].float().tolist()}")
print(f"V[:,32]=1: O[0,32] should be 1, got {kern_d32[0, 32].float().item():.6f}")

