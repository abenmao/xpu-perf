import importlib.util
import os
import pathlib
import sys
import torch
sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry 
from core.ops.llm_ops import MoeGatherOp as MoeGatherBaseOp
from core.utils import OpTensorInfo, calc_tensor_size
from functools import partial
import pathlib
import sys
from functools import partial

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.llm_ops import MoeGatherOp as MoeGatherBaseOp
from core.utils import OpTensorInfo, calc_tensor_size, create_from_list


try:
    import vllm_xpu_kernels._moe_C

    torch.ops._moe_C.moe_gather

    @ProviderRegistry.register_vendor_impl("moe_gather", "vllm_xpu_kernels")
    class VLLMXPUKernelsMoeGatherOp(MoeGatherBaseOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

        def vendor_parser(self):
            if self.dtype not in ["float16", "bfloat16", "float32"]:
                raise ValueError(
                    "VLLMXPUKernelsMoeGatherOp only supports float16, "
                    f"bfloat16, and float32, but got {self.dtype}"
                )

        def vendor_impl(self):
            self.torch_dtype = getattr(torch, self.dtype)
            flat_topk_weights = [
                weight
                for token_weights in self.all_select_weights
                for weight in token_weights
            ]

            expert_first_token_offset = [0]
            for token_count in self.expert_dispatch_token_count:
                expert_first_token_offset.append(
                    expert_first_token_offset[-1] + token_count
                )

            unpermuted_row_to_permuted_row = []
            expert_local_row_offsets = [0] * self.num_experts_per_rank
            for token_experts in self.all_select_experts:
                for expert_idx in token_experts:
                    if self.experts_start_idx <= expert_idx < self.experts_end_idx:
                        local_expert_idx = expert_idx - self.experts_start_idx
                        row_idx = (
                            expert_first_token_offset[local_expert_idx]
                            + expert_local_row_offsets[local_expert_idx]
                        )
                        unpermuted_row_to_permuted_row.append(row_idx)
                        expert_local_row_offsets[local_expert_idx] += 1
                    else:
                        unpermuted_row_to_permuted_row.append(-1)

            self.input_tensor_info = {
                "scatter_tokens": OpTensorInfo(
                    shape=[self.dispatch_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "topk_weights": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=partial(create_from_list, data=flat_topk_weights),
                ),
                "unpermuted_row_to_permuted_row": OpTensorInfo(
                    shape=[self.num_tokens * self.topk],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=partial(
                        create_from_list,
                        data=unpermuted_row_to_permuted_row,
                    ),
                ),
                "expert_first_token_offset": OpTensorInfo(
                    shape=[self.num_experts_per_rank + 1],
                    dtype=torch.int64,
                    device=self.backend.get_torch_device_name(),
                    creator=partial(
                        create_from_list,
                        data=expert_first_token_offset,
                    ),
                ),
            }
            self.output_tensor_info = {
                "convergent_tokens": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.zeros,
                ),
            }

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

            self.algo_size = 0
            self.bus_size = 0

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )
            self._run_func = self.vendor_impl_run

        def vendor_impl_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
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
            return convergent_tokens

except Exception:
    pass

