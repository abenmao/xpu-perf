import torch
from functools import partial

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import MoeScatterDynamicQuantOp, MoeSwigluDynamicQuantOp

try:
    import vllm_xpu_kernels._C  # noqa: F401
    import vllm_xpu_kernels._moe_C

    # -------------------------------------------------------------
    # 1. MOE Scatter Dynamic Quant
    # -------------------------------------------------------------
    @ProviderRegistry.register_vendor_impl("moe_scatter_dynamic_quant", "vllm_xpu_kernels")
    class VLLMXPUKernelsMoeScatterDynamicQuantOp(MoeScatterDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if self.arg_type not in ["llm"]:
                raise NotImplementedError(f"Unsupported arg_type: {self.arg_type}")

            self.dtype = self.args_dict["dtype"]
            if self.dtype not in ["float16", "bfloat16"]:
                raise NotImplementedError(f"Unsupported dtype: {self.dtype}")
            self.torch_dtype = getattr(torch, self.dtype)

            self.dst_dtype = self.args_dict["dst_dtype"]
            if self.dst_dtype not in ["int8"]:
                raise NotImplementedError(f"Unsupported dst_dtype: {self.dst_dtype}")
            self.dst_torch_dtype = getattr(torch, self.dst_dtype)

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

            self.dp_rank = self.rank // self.sp_size
            self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

            self.sp_rank = self.rank % self.sp_size
            self.shared_tokens_per_sp = self.num_tokens // self.sp_size
            self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
            self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

            self.experts_per_rank = self.num_experts // self.ep_size
            self.ep_rank = self.rank
            self.expert_idx_start = self.ep_rank * self.experts_per_rank
            self.expert_idx_end = self.expert_idx_start + self.experts_per_rank
            self.other_experts_set = set(range(self.num_experts)) - set(range(self.expert_idx_start, self.expert_idx_end))

            self.tokens_per_ep = self.num_tokens // self.ep_size
            self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
            self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

            self.actual_output_tokens = self.tokens_per_ep * self.topk
            self.experts_repeat_time = 1
            if self.actual_output_tokens > self.experts_per_rank:
                self.experts_repeat_time = (self.actual_output_tokens + self.experts_per_rank - 1) // self.experts_per_rank

            self.refer_expert_seq = torch.arange(
                start=self.expert_idx_start,
                end=self.expert_idx_end,
                dtype=torch.int32
            ).repeat(self.experts_repeat_time)[:self.actual_output_tokens].view(self.tokens_per_ep, self.topk)

            self.total_experts_num = self.shared_experts_per_rank + self.experts_per_rank
            self.total_shared_tokens = self.shared_tokens_per_sp * self.shared_experts_per_rank
            self.max_allocated_tokens = self.num_tokens * self.topk
            self.max_scatter_tokens = self.total_shared_tokens + self.max_allocated_tokens

            self.real_allocated_tokens = self.actual_output_tokens
            self.real_scatter_tokens = self.total_shared_tokens + self.real_allocated_tokens

            dev = self.backend.get_torch_device_name()

            self.input_tensor_info = {
                "hidden_states": OpTensorInfo(
                    shape=[self.num_tokens, self.hidden_size],
                    dtype=self.torch_dtype,
                    device=dev,
                ),
                "selected_experts": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.int32,
                    device=dev,
                    creator=lambda size, dtype, device: torch.randint(
                        0, self.experts_per_rank, [self.num_tokens, self.topk], device=device, dtype=dtype
                    )
                ),
                "token_to_scatter_offset": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.int32,
                    device=dev,
                    creator=lambda size, dtype, device: torch.zeros(
                        [self.num_tokens, self.topk], device=dev, dtype=dtype
                    )
                ),
                "moe_weights": OpTensorInfo(
                    shape=[self.num_tokens, self.topk],
                    dtype=torch.float32,
                    device=dev,
                    creator=torch.ones
                ),
                "smooth_scale": OpTensorInfo(
                    shape=[self.total_experts_num, self.hidden_size],
                    dtype=torch.float32,
                    device=dev,
                    creator=torch.ones
                ),
            }

            self.output_tensor_info = {
                "scatter_tokens": OpTensorInfo(
                    shape=[self.max_scatter_tokens, self.hidden_size],
                    dtype=self.dst_torch_dtype,
                    device=dev,
                    creator=torch.zeros
                ),
                "scatter_per_token_scale": OpTensorInfo(
                    shape=[self.max_scatter_tokens],
                    dtype=torch.float32,
                    device=dev,
                    creator=torch.ones
                ),
                "scatter_tokens_offset": OpTensorInfo(
                    shape=[self.max_scatter_tokens],
                    dtype=torch.int32,
                    device=dev,
                    creator=lambda size, dtype, device: torch.ones(size, dtype=dtype, device=device) * -1
                ),
                "experts_token_count": OpTensorInfo(
                    shape=[self.total_experts_num],
                    dtype=torch.int32,
                    device=dev,
                    creator=torch.zeros
                ),
                "experts_token_start": OpTensorInfo(
                    shape=[self.total_experts_num],
                    dtype=torch.int32,
                    device=dev,
                    creator=torch.zeros
                )
            }

            self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = (
                calc_tensor_size(self.input_tensor_info["hidden_states"]) / self.num_tokens * self.tokens_per_ep +
                calc_tensor_size(self.input_tensor_info["selected_experts"]) +
                calc_tensor_size(self.input_tensor_info["moe_weights"]) +
                calc_tensor_size(self.input_tensor_info["smooth_scale"]) * (self.max_scatter_tokens / self.experts_per_rank / 4)
            )
            self.write_bytes = (
                calc_tensor_size(self.output_tensor_info["scatter_tokens"]) / self.max_scatter_tokens * self.real_scatter_tokens +
                calc_tensor_size(self.output_tensor_info["scatter_per_token_scale"]) / self.max_scatter_tokens * self.real_scatter_tokens +
                calc_tensor_size(self.output_tensor_info["scatter_tokens_offset"]) / self.max_scatter_tokens * self.real_scatter_tokens +
                calc_tensor_size(self.output_tensor_info["experts_token_count"]) +
                calc_tensor_size(self.output_tensor_info["experts_token_start"])
            )
            self.io_bytes = self.read_bytes + self.write_bytes

            self.algo_size = 0
            self.bus_size = 0

            self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)
            self._run_func = self.moe_scatter_dynamic_quant_run

        def moe_scatter_dynamic_quant_run(self, tensor_mapping):
            hidden_states = tensor_mapping["hidden_states"]
            selected_experts = tensor_mapping["selected_experts"]
            moe_weights = tensor_mapping["moe_weights"]
            token_to_scatter_offset = tensor_mapping["token_to_scatter_offset"]
            smooth_scale = tensor_mapping["smooth_scale"]

            scatter_tokens = tensor_mapping["scatter_tokens"]
            scatter_per_token_scale = tensor_mapping["scatter_per_token_scale"]
            scatter_tokens_offset = tensor_mapping["scatter_tokens_offset"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_start = tensor_mapping["experts_token_start"]

            token_to_scatter_offset.zero_()
            experts_token_count.zero_()
            experts_token_start.zero_()

            torch.ops._moe_C.moe_scatter_dynamic_quant(
                selected_experts,
                moe_weights,
                token_to_scatter_offset,
                experts_token_count,
                experts_token_start,
                hidden_states,
                smooth_scale,
                scatter_tokens,
                scatter_per_token_scale,
                scatter_tokens_offset,
                self.num_shared_experts
            )

            return scatter_tokens

    # -------------------------------------------------------------
    # 2. MOE SwiGLU Dynamic Quant
    # -------------------------------------------------------------
    @ProviderRegistry.register_vendor_impl("moe_swiglu_dynamic_quant", "vllm_xpu_kernels")
    class VLLMXPUKernelsMoeSwigluDynamicQuantOp(MoeSwigluDynamicQuantOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["vllm_xpu_kernels"]

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if self.arg_type not in ["llm"]:
                raise NotImplementedError(f"Unsupported arg_type: {self.arg_type}")

            self.dtype = self.args_dict["dtype"]
            if self.dtype not in ["float16", "bfloat16"]:
                raise NotImplementedError(f"Unsupported dtype: {self.dtype}")
            self.torch_dtype = getattr(torch, self.dtype)

            self.dst_dtype = self.args_dict["dst_dtype"]
            if self.dst_dtype not in ["int8"]:
                raise NotImplementedError(f"Unsupported dst_dtype: {self.dst_dtype}")
            self.dst_torch_dtype = getattr(torch, self.dst_dtype)

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

            self.dp_rank = self.rank // self.sp_size
            self.shared_experts_per_rank = self.num_shared_experts // self.dp_size

            self.sp_rank = self.rank % self.sp_size
            self.shared_tokens_per_sp = self.num_tokens // self.sp_size
            self.shared_token_sp_start = self.sp_rank * self.shared_tokens_per_sp
            self.shared_token_sp_end = self.shared_token_sp_start + self.shared_tokens_per_sp

            self.experts_per_rank = self.num_experts // self.ep_size
            self.ep_rank = self.rank
            self.expert_idx_start = self.ep_rank * self.experts_per_rank
            self.expert_idx_end = self.expert_idx_start + self.experts_per_rank

            self.tokens_per_ep = self.num_tokens // self.ep_size
            self.tokens_ep_start = self.ep_rank * self.tokens_per_ep
            self.tokens_ep_end = self.tokens_ep_start + self.tokens_per_ep

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

            dev = self.backend.get_torch_device_name()

            self.input_tensor_info = {
                "scatter_tokens": OpTensorInfo(
                    shape=[self.real_scatter_tokens, self.hidden_size * 2],
                    dtype=self.torch_dtype,
                    device=dev,
                ),
                "smooth_scale": OpTensorInfo(
                    shape=[self.total_experts_num, self.hidden_size],
                    dtype=torch.float32,
                    device=dev,
                    creator=torch.ones
                ),
                "experts_token_count": OpTensorInfo(
                    shape=[self.total_experts_num],
                    dtype=torch.int32,
                    device=dev,
                    creator=lambda size, dtype, device: torch.tensor(self.token_list, dtype=dtype, device=device)
                ),
                "experts_token_start": OpTensorInfo(
                    shape=[self.total_experts_num],
                    dtype=torch.int32,
                    device=dev,
                    creator=lambda size, dtype, device: torch.tensor(self.token_start_list, dtype=dtype, device=device)
                ),
            }

            self.output_tensor_info = {
                "quant_tokens": OpTensorInfo(
                    shape=[self.real_scatter_tokens, self.hidden_size],
                    dtype=self.dst_torch_dtype,
                    device=dev,
                ),
                "per_token_scale": OpTensorInfo(
                    shape=[self.real_scatter_tokens],
                    dtype=torch.float32,
                    device=dev,
                ),
            }

            self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            self._create_tensors_func = partial(self._create_in_out_tensors, create_inputs=True, create_outputs=True)
            self._run_func = self.moe_swiglu_dynamic_quant_run

        def moe_swiglu_dynamic_quant_run(self, tensor_mapping):
            scatter_tokens = tensor_mapping["scatter_tokens"]
            smooth_scale = tensor_mapping["smooth_scale"]
            experts_token_count = tensor_mapping["experts_token_count"]
            experts_token_start = tensor_mapping["experts_token_start"]

            quant_tokens = tensor_mapping["quant_tokens"]
            per_token_scale = tensor_mapping["per_token_scale"]

            torch.ops._moe_C.moe_swiglu_dynamic_quant(
                scatter_tokens,
                smooth_scale,
                experts_token_count,
                experts_token_start,
                quant_tokens,
                per_token_scale,
                self.total_experts_num,
                self.max_token_num
            )

            return quant_tokens, per_token_scale

except ImportError:
    pass

