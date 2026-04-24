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
from core.ops.llm_ops import MoeSwigluDynamicQuantOp

try:
    torch.ops.torch_ipex.moe_swiglu_dynamic_quant

    @ProviderRegistry.register_vendor_impl("moe_swiglu_dynamic_quant", "ipex")
    class MoeSwigluDynamicQuantIpexOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

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
            no remainder on **num_experts**
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

            self.token_list = []
            self.token_start_list = []
            temp_token_start = 0
            for i in range(self.shared_experts_per_rank):
                self.token_start_list.append(temp_token_start)
                self.token_list.append(self.shared_tokens_per_sp)
                temp_token_start += self.token_list[-1]
            for i in range(self.experts_per_rank):
                self.token_start_list.append(temp_token_start)
                if i < self.allocated_tokens_per_expert_remainder:
                    self.token_list.append(self.allocated_tokens_per_expert + 1)
                else:
                    self.token_list.append(self.allocated_tokens_per_expert)
                temp_token_start += self.token_list[-1]


            self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank

            self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank

            self.real_allocated_tokens = self.allocated_tokens
            self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens
            self.max_token_num = max(self.token_list)
        
            # input/output tensors
            self.input_tensor_info = {
                "scatter_tokens": OpTensorInfo(
                    shape=[self.real_scatter_tokens, self.hidden_size * 2], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                # use 1 as smooth scale
                "smooth_scale": OpTensorInfo(
                    shape=[self.total_experts_num, self.hidden_size], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ), 
                "experts_token_count": OpTensorInfo(
                    shape=[self.total_experts_num], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(), 
                    creator=lambda size, dtype, device: torch.tensor(
                        self.token_list, 
                        dtype=dtype, device=device
                    )
                ), 
                "experts_token_start": OpTensorInfo(
                    shape=[self.total_experts_num], 
                    dtype=torch.int32, 
                    device=self.backend.get_torch_device_name(), 
                    creator=lambda size, dtype, device: torch.tensor(
                        self.token_start_list, 
                        dtype=dtype, device=device
                    )
                ), 
            }
            self.output_tensor_info = {
                "quant_tokens": OpTensorInfo(
                    shape=[self.real_scatter_tokens, self.hidden_size], 
                    dtype=self.torch_dtype, 
                    device=self.backend.get_torch_device_name(),
                ), 
                "per_token_scale": OpTensorInfo(
                    shape=[self.real_scatter_tokens], 
                    dtype=torch.float32, 
                    device=self.backend.get_torch_device_name(),
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

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors, 
                create_inputs=True, 
                create_outputs=True
            )

            # run func
            self._run_func = self.moe_swiglu_dynamic_quant_run


        def moe_swiglu_dynamic_quant_run(self, tensor_mapping): 
            # get pre-allocated input tensors
            scatter_tokens = tensor_mapping["scatter_tokens"]
            smooth_scale = tensor_mapping["smooth_scale"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_start = tensor_mapping["experts_token_start"]

            # get per-allocated output tensors
            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            torch.ops.torch_ipex.moe_swiglu_dynamic_quant(
                scatter_tokens,
                smooth_scale,
                experts_token_count,
                experts_token_start,
                quant_tokens,
                per_token_scale,
                self.experts_per_rank,
                self.max_token_num
            )

            return quant_tokens, per_token_scale



except Exception:
    pass

OP_MAPPING = {"torch": MoeSwigluDynamicQuantOp}
