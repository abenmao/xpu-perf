import sys
import pathlib
from functools import partial

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import MoeGatherOp as _MoeGatherBaseOp

try:
    import torch
    torch.ops.torch_ipex.moe_gather_function

    @ProviderRegistry.register_vendor_impl("moe_gather", "ipex")
    class MoeGatherOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError

            # src_dtype
            self.dtype = self.args_dict["dtype"]
            if not self.dtype in ["float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            # predefined attrs
            self.world_size = self.args_dict.get("world_size", 1)
            self.rank = self.args_dict.get("rank", 0)
            self.ep_size = self.args_dict.get("ep_size", 1)
            self.dp_size = self.args_dict.get("dp_size", 1)
            self.sp_size = self.args_dict.get("sp_size", 1)

            self.num_shared_experts = self.args_dict.get("num_shared_experts", 0)
            self.num_experts = self.args_dict["num_experts"]
            self.topk = self.args_dict["topk"]
            self.num_tokens = self.args_dict["num_tokens"]
            self.hidden_size = self.args_dict["hidden_size"]

            """
            select shared experts based on dp_size/dp_rank
            """
            self.dp_rank = self.rank // self.sp_size
            self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

            """
            select tokens based on sp_size/sp_rank
            """
            self.sp_rank = self.rank % self.sp_size
            self.shared_tokens_per_sp = self.num_tokens // self.sp_size
            self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
            self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

            """
            select experts based on ep_rank
            """
            self.experts_per_rank = self.num_experts // self.ep_size
            self.ep_rank = self.rank
            self.expert_idx_start = self.ep_rank * self.experts_per_rank
            self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
            self.other_experts_set = \
                set(range(self.num_experts)) - \
                set(range(self.expert_idx_start, self.expert_idx_end))

            """
            for convinience, we also split num_tokens to ep_size parts to generate selected_experts
            """
            self.tokens_per_ep = self.num_tokens // self.ep_size
            self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
            self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

            # [tokens_per_ep, topk]
            self.allocated_tokens = self.tokens_per_ep * self.topk
            self.allocated_tokens_per_expert = self.allocated_tokens // self.experts_per_rank
            self.allocated_tokens_per_expert_remainder = self.allocated_tokens % self.experts_per_rank

            self.token_offset_list = []

            # for shared experts
            for _ in range(self.shared_experts_per_rank):
                self.token_offset_list.extend(
                    range(self.shared_token_sp_start, self.shared_token_sp_end)
                )

            # for experts
            for expert_idx in range(self.experts_per_rank):
                cur_select_token = expert_idx
                while cur_select_token // self.topk < self.tokens_per_ep:
                    self.token_offset_list.append(cur_select_token // self.topk + self.tokens_ep_start)
                    cur_select_token += self.experts_per_rank

            self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

            self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

            self.real_allocated_tokens = self.allocated_tokens
            self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens

            # input/output tensors
            self.input_tensor_info = {
                "scatter_tokens": OpTensorInfo(
                    shape=[self.real_scatter_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                ),
                "scatter_tokens_offset": OpTensorInfo(
                    shape=[self.real_scatter_tokens],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.token_offset_list, dtype=dtype, device=device)
                ),
            }
            self.output_tensor_info = {
                # init zero
                "convergent_tokens": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.zeros
                ),
            }

            # calculator
            self.input_tensor_size = sum([
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            ])
            self.output_tensor_size = sum([
                calc_tensor_size(info) for info in self.output_tensor_info.values()
            ])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = \
                2 * calc_tensor_size(self.input_tensor_info["scatter_tokens"]) + \
                calc_tensor_size(self.input_tensor_info["scatter_tokens_offset"])
            self.write_bytes = calc_tensor_size(self.input_tensor_info["scatter_tokens"])
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True
            )

            # run func
            self._run_func = self.moe_gather_run

        def moe_gather_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
            convergent_tokens = tensor_mapping["convergent_tokens"]

            torch.ops.torch_ipex.moe_gather_function(scatter_tokens, scatter_tokens_offset,
                                                     convergent_tokens, self.topk)
            return convergent_tokens



except Exception:
    pass

OP_MAPPING = {"torch": _MoeGatherBaseOp}
