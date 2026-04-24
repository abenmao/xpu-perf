from functools import partial
import pathlib
import sys
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeGatherOp
from core.utils import OpTensorInfo, calc_tensor_size, create_from_list


try:
    import vllm_xpu_kernels._moe_C
    @ProviderRegistry.register_vendor_impl("moe_gather", "vllm_xpu_kernels")
    class VLLMXPUKernelsMoeGatherOp(MoeGatherOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.extra_providers = ["vllm_xpu_kernels"]
            
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.vendor_impl_run

        def vendor_parser(self):
            if self.dtype in ["bfloat16","float16"]:
                pass
            else:
                raise ValueError(
                    f"MoeGatherOp base impl only support bfloat16 float16 dtype, but got {self.dtype}"
                )
        
        def vendor_impl(self):
            super().vendor_impl()

            flat_topk_weights, unpermuted_row_to_permuted_row, expert_first_token_offset = \
                self._build_vllm_moe_gather_inputs()

            self.input_tensor_info["topk_weights"] = OpTensorInfo(
                shape=[self.num_tokens, self.topk],
                dtype=torch.float32,
                device=self.backend.get_torch_device_name(),
                creator=partial(create_from_list, data=flat_topk_weights),
            )
            self.input_tensor_info["unpermuted_row_to_permuted_row"] = OpTensorInfo(
                shape=[self.num_tokens * self.topk],
                dtype=torch.int32,
                device=self.backend.get_torch_device_name(),
                creator=partial(
                    create_from_list,
                    data=unpermuted_row_to_permuted_row,
                ),
            )
            self.input_tensor_info["expert_first_token_offset"] = OpTensorInfo(
                shape=[self.num_experts_per_rank + 1],
                dtype=torch.int64,
                device=self.backend.get_torch_device_name(),
                creator=partial(
                    create_from_list,
                    data=expert_first_token_offset,
                ),
            )

            self.input_tensor_size = sum(
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            )
            self.output_tensor_size = sum(
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            )
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

        def _build_vllm_moe_gather_inputs(self):
            flat_topk_weights = [0.0] * (self.num_tokens * self.topk)
            unpermuted_row_to_permuted_row = [-1] * (self.num_tokens * self.topk)

            expert_first_token_offset = [0]
            for token_count in self.expert_dispatch_token_count:
                expert_first_token_offset.append(
                    expert_first_token_offset[-1] + token_count
                )

            token_fill_count = [0] * self.num_tokens
            for row_idx, token_idx in enumerate(self.scatter_token_id):
                slot_idx = token_fill_count[token_idx]
                if slot_idx >= self.topk:
                    continue

                target_idx = token_idx * self.topk + slot_idx
                flat_topk_weights[target_idx] = self.scatter_token_weight[row_idx]
                unpermuted_row_to_permuted_row[target_idx] = row_idx
                token_fill_count[token_idx] += 1

            return (
                flat_topk_weights,
                unpermuted_row_to_permuted_row,
                expert_first_token_offset,
            )

            

        def vendor_impl_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            residual_tokens = tensor_mapping["residual_tokens"]
            topk_weights = tensor_mapping["topk_weights"]
            unpermuted_row_to_permuted_row = tensor_mapping[
                "unpermuted_row_to_permuted_row"
            ]
            expert_first_token_offset = tensor_mapping[
                "expert_first_token_offset"
            ]

            convergent_tokens = tensor_mapping["convergent_tokens"]

            torch.ops._moe_C.moe_gather(
                convergent_tokens,
                scatter_tokens,
                topk_weights,
                unpermuted_row_to_permuted_row,
                expert_first_token_offset,
                self.num_experts_per_rank,
            )
            # convergent_tokens[
            #     self.res_token_start:self.res_token_end
            # ] += residual_tokens * self.res_scale
            return convergent_tokens

except Exception:
    pass

