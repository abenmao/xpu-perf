import sys
import pathlib
from functools import partial

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import MoeQuantGroupGemmCombineOp

try:
    import torch
    import intel_extension_for_pytorch as ipex
    torch.ops.torch_ipex.mm_w8a8

    @ProviderRegistry.register_vendor_impl("moe_quant_group_gemm_combine", "ipex")
    class MoeQuantGroupGemmCombineIpexOp(MoeQuantGroupGemmCombineOp):
        """
        IPEX-optimized moe_quant_group_gemm_combine using mm_w8a8 for per-expert GEMM
        followed by index_add_ for the combine/gather step.
        """

        def vendor_impl(self):
            # Call base to set up tensors, sizes, run func
            super().vendor_impl()
            # Override run func
            self._run_func = self.moe_quant_group_gemm_combine_ipex_run

        def moe_quant_group_gemm_combine_ipex_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]
            experts_weight = tensor_mapping["experts_weight"]
            experts_scale = tensor_mapping["experts_scale"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_offset = tensor_mapping["experts_token_offset"]
            scatter_token_id = tensor_mapping["scatter_token_id"]
            scatter_token_weight = tensor_mapping["scatter_token_weight"]
            residual_tokens = tensor_mapping["residual_tokens"]
            convergent_tokens = tensor_mapping["convergent_tokens"]

            # Add residual
            convergent_tokens[self.res_token_start:self.res_token_end] += residual_tokens * self.res_scale

            # Allocate output for per-expert GEMM
            new_scatter_tokens = torch.empty(
                size=[self.dispatch_tokens, self.new_hidden_size],
                dtype=self.dst_torch_dtype,
                device=self.backend.get_torch_device_name(),
            )

            # Per-expert quantized GEMM using mm_w8a8
            # experts_weight: [num_experts_per_rank, new_hidden_size, hidden_size]
            # Transpose to [num_experts_per_rank, hidden_size, new_hidden_size]
            # mm_w8a8 expects column-major per-expert weight (non-contiguous transpose view)
            weight_transposed = experts_weight.transpose(1, 2)

            zero_points = torch.zeros(1, dtype=torch.int8, device=scatter_tokens.device)

            for expert_idx in range(self.num_experts_per_rank):
                cur_token_start = experts_token_offset[expert_idx].item()
                cur_token_count = experts_token_count[expert_idx].item()
                if cur_token_count == 0:
                    continue

                cur_tokens = scatter_tokens[cur_token_start:cur_token_start + cur_token_count]
                cur_scale = per_token_scale[cur_token_start:cur_token_start + cur_token_count].unsqueeze(-1)
                cur_weight = weight_transposed[expert_idx]
                cur_weight_scale = experts_scale[expert_idx].unsqueeze(0)

                torch.ops.torch_ipex.mm_w8a8(
                    cur_tokens,
                    cur_scale,
                    None,
                    cur_weight,
                    cur_weight_scale,
                    zero_points,
                    new_scatter_tokens[cur_token_start:cur_token_start + cur_token_count]
                )

            # Combine: weighted gather via index_add_
            convergent_tokens.index_add_(
                0, scatter_token_id,
                (new_scatter_tokens * scatter_token_weight.unsqueeze(-1)).to(self.dst_torch_dtype)
            )

            return convergent_tokens



except Exception:
    pass

OP_MAPPING = {"torch": MoeQuantGroupGemmCombineOp}
