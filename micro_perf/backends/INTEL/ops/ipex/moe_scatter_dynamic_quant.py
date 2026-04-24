import sys
import pathlib
import torch
import random
from functools import partial
from itertools import combinations

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import MoeScatterDynamicQuantOp

try:
    torch.ops.torch_ipex.moe_scatter_dynamic_quant

    @ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "ipex")
    class MoeScatterDynamicQuantIpexOp(BasicOp):
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

            # dst_dtype
            self.dst_dtype = self.args_dict["dst_dtype"]
            if not self.dst_dtype in ["int8"]:
                raise NotImplementedError
            self.dst_torch_dtype = getattr(torch, self.dst_dtype)

            # pre-defined attrs
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
            and selected tokens are also distributed to corresponding experts
            if rank == 0, experts_per_rank == 4, and topk == 5, and num_tokens == 32, so num_tokens_per_ep == 8
            token 0: 0, 1, 2, 3, 0
            token 1: 1, 2, 3, 0, 1
            token 2: 2, 3, 0, 1, 2
            token 3: 3, 0, 1, 2, 3
            ...

            other tokens will select other tokens randomly
            """
            self.tokens_per_ep = self.num_tokens // self.ep_size
            self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
            self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep
        
            # [tokens_per_ep, topk]
            self.actual_output_tokens = self.tokens_per_ep * self.topk
            self.experts_repeat_time = 1
            if self.actual_output_tokens > self.experts_per_rank:
                self.experts_repeat_time = (self.actual_output_tokens + self.experts_per_rank - 1) // self.experts_per_rank
            self.refer_expert_seq = torch.arange(
                start=self.expert_idx_start, 
                end=self.expert_idx_end, 
                dtype=torch.int32
            ).repeat(self.experts_repeat_time)[:self.actual_output_tokens].view(
                self.tokens_per_ep, self.topk)

            # all tokens topk
            # dummy_experts = list(next(combinations(self.other_experts_set, self.topk)))

            # self.refer_selected_experts = torch.tensor(dummy_experts, dtype=torch.int32).unsqueeze(0).repeat(self.num_tokens, 1)
            # self.refer_selected_experts[self.tokens_ep_start:self.tokens_ep_end] = self.refer_expert_seq


            self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

            # reserve tokens memory for shared_tokens_per_sp/allocated_tokens
            self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

            # for extreme case
            self.max_allocated_tokens = self.num_tokens * self.topk
            self.max_scatter_tokens = self.total_shared_tokens + self.max_allocated_tokens

            # for designed real case
            self.real_allocated_tokens = self.actual_output_tokens
            self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens


            # input/output tensors
            self.input_tensor_info = {
                # complete tokens
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                # complete selected_experts
                "selected_experts": OpTensorInfo(
                    shape=[self.num_tokens, self.topk], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.randint(
                        0, self.experts_per_rank,
                        [self.num_tokens, self.topk], device=self.backend.get_torch_device_name(), dtype=torch.int32
                    )
                ), 
                "token_to_scatter_offset": OpTensorInfo(
                    shape=[self.num_tokens, self.topk], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.zeros(
                        [self.num_tokens, self.topk], device=self.backend.get_torch_device_name(), dtype=torch.int32
                    )
                ), 
                # complete moe_weights
                "moe_weights": OpTensorInfo(
                    shape=[self.num_tokens, self.topk], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ), 
                # partial (shared_experts + experts) smooth_scale
                # use 1 as smooth scale
                "smooth_scale": OpTensorInfo(
                    shape=[self.total_experts_num, self.hidden_size], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ), 
            }
            self.output_tensor_info = {
                # partial, reserved for max
                "scatter_tokens": OpTensorInfo(
                    shape=[self.max_scatter_tokens, self.hidden_size], 
                    dtype=self.dst_torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.zeros
                ), 
                # partial, reserved for max
                "scatter_per_token_scale": OpTensorInfo(
                    shape=[self.max_scatter_tokens], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ), 
                # partial, reserved for max
                "scatter_tokens_offset": OpTensorInfo(
                    shape=[self.max_scatter_tokens], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * -1
                ), 
                # partial (shared_experts + experts) token count
                "experts_token_count": OpTensorInfo(
                    shape=[self.total_experts_num], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.zeros
                ), 
                # partial (shared_experts + experts) token start
                "experts_token_start": OpTensorInfo(
                    shape=[self.total_experts_num], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.zeros
                )
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
                calc_tensor_size(self.input_tensor_info["hidden_states"]) / self.num_tokens * self.tokens_per_ep + \
                calc_tensor_size(self.input_tensor_info["selected_experts"]) + \
                calc_tensor_size(self.input_tensor_info["moe_weights"]) + \
                calc_tensor_size(self.input_tensor_info["smooth_scale"]) * (self.max_scatter_tokens / self.experts_per_rank / 4)
            self.write_bytes = \
                calc_tensor_size(self.output_tensor_info["scatter_tokens"]) / self.max_scatter_tokens * self.real_scatter_tokens + \
                calc_tensor_size(self.output_tensor_info["scatter_per_token_scale"]) / self.max_scatter_tokens * self.real_scatter_tokens + \
                calc_tensor_size(self.output_tensor_info["scatter_tokens_offset"]) / self.max_scatter_tokens * self.real_scatter_tokens + \
                calc_tensor_size(self.output_tensor_info["experts_token_count"]) + \
                calc_tensor_size(self.output_tensor_info["experts_token_start"])
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
            self._run_func = self.moe_scatter_dynamic_quant_run




        def moe_scatter_dynamic_quant_run(self, tensor_mapping):
            # get pre-allocated input tensors
            hidden_states = tensor_mapping["hidden_states"]
            selected_experts = tensor_mapping["selected_experts"]
            moe_weights = tensor_mapping["moe_weights"]
            token_to_scatter_offset = tensor_mapping["token_to_scatter_offset"]
            smooth_scale = tensor_mapping["smooth_scale"]

            # get pre-allocated output tensors
            scatter_tokens = tensor_mapping["scatter_tokens"]
            scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]
            scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_start = tensor_mapping["experts_token_start"]

            token_to_scatter_offset[...] = 0
            experts_token_count[...] = 0
            experts_token_start[...] = 0

            torch.ops.torch_ipex.moe_scatter_dynamic_quant(
                selected_experts,
                moe_weights,
                token_to_scatter_offset,
                experts_token_count, experts_token_start,
                hidden_states,
                smooth_scale,
                scatter_tokens,
                scatter_per_token_scale,
                scatter_tokens_offset,
                self.num_shared_experts)



except Exception:
    pass

OP_MAPPING = {"torch": MoeScatterDynamicQuantOp}
