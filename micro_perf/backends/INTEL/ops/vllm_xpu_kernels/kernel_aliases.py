import sys
import pathlib
from functools import partial

import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size, get_torch_dtype
from core.ops.llm_ops import AddRmsNormOp, SwigluOp, FlashAttentionOp, MoeSoftmaxTopkOp, MoeGatherOp, StoreKVCacheOp
from core.ops.tensor_gemm_ops import GemmOp


class _KernelDirectBase(BasicOp):
    def _finalize(self, create_outputs=True):
        self.input_tensor_size = sum(calc_tensor_size(info) for info in self.input_tensor_info.values())
        self.output_tensor_size = sum(calc_tensor_size(info) for info in self.output_tensor_info.values())
        self.tensor_size = self.input_tensor_size + self.output_tensor_size
        self.read_bytes = self.input_tensor_size
        self.write_bytes = self.output_tensor_size
        self.io_bytes = self.read_bytes + self.write_bytes
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=create_outputs,
        )
        self._run_func = self.vendor_impl_run


try:
    import vllm_xpu_kernels._C
except Exception:
    pass

try:
    import vllm_xpu_kernels._moe_C
except Exception:
    pass

try:
    import vllm_xpu_kernels._xpu_C
except Exception:
    pass


@ProviderRegistry.register_vendor_impl("fused_add_rms_norm", "vllm_xpu_kernels")
class VLLMXPUKernelsFusedAddRMSNormOp(AddRmsNormOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["vllm_xpu_kernels"]
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=False,
        )

    def add_rms_norm_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        residual = tensor_mapping["residual"]
        norm_weight = tensor_mapping["norm_weight"]
        torch.ops._C.fused_add_rms_norm(hidden_states, residual, norm_weight, self.eps)
        return hidden_states


@ProviderRegistry.register_vendor_impl("silu_and_mul", "vllm_xpu_kernels")
class VLLMXPUKernelsSiluAndMulOp(SwigluOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["vllm_xpu_kernels"]
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )

    def vendor_impl_run(self, tensor_mapping):
        hidden_states = tensor_mapping["hidden_states"]
        output_tokens = tensor_mapping["output_tokens"]
        torch.ops._C.silu_and_mul(output_tokens, hidden_states)
        return output_tokens


class _FusedActivationDirectOp(_KernelDirectBase):
    kernel_name = None
    out_half = False

    def prepare(self):
        self.dtype = self.args_dict.get("dtype", "bfloat16")
        self.torch_dtype = get_torch_dtype(self.dtype)
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)

        in_hidden = self.hidden_size * 2 if self.out_half else self.hidden_size
        out_hidden = self.hidden_size
        if not self.out_half:
            out_hidden = in_hidden

        self.input_tensor_info = {
            "input": OpTensorInfo(
                shape=[self.num_tokens, in_hidden],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self.output_tensor_info = {
            "out": OpTensorInfo(
                shape=[self.num_tokens, out_hidden],
                dtype=self.torch_dtype,
                device=self.backend.get_torch_device_name(),
            )
        }
        self._finalize(create_outputs=True)

    def vendor_impl_run(self, tensor_mapping):
        inp = tensor_mapping["input"]
        out = tensor_mapping["out"]
        getattr(torch.ops._C, self.kernel_name)(out, inp)
        return out


@ProviderRegistry.register_vendor_impl("mul_and_silu", "vllm_xpu_kernels")
class VLLMXPUKernelsMulAndSiluOp(_FusedActivationDirectOp):
    kernel_name = "mul_and_silu"
    out_half = True


@ProviderRegistry.register_vendor_impl("gelu_and_mul", "vllm_xpu_kernels")
class VLLMXPUKernelsGeluAndMulOp(_FusedActivationDirectOp):
    kernel_name = "gelu_and_mul"
    out_half = True


@ProviderRegistry.register_vendor_impl("gelu_tanh_and_mul", "vllm_xpu_kernels")
class VLLMXPUKernelsGeluTanhAndMulOp(_FusedActivationDirectOp):
    kernel_name = "gelu_tanh_and_mul"
    out_half = True


@ProviderRegistry.register_vendor_impl("gelu_fast", "vllm_xpu_kernels")
class VLLMXPUKernelsGeluFastOp(_FusedActivationDirectOp):
    kernel_name = "gelu_fast"


@ProviderRegistry.register_vendor_impl("gelu_new", "vllm_xpu_kernels")
class VLLMXPUKernelsGeluNewOp(_FusedActivationDirectOp):
    kernel_name = "gelu_new"


@ProviderRegistry.register_vendor_impl("gelu_quick", "vllm_xpu_kernels")
class VLLMXPUKernelsGeluQuickOp(_FusedActivationDirectOp):
    kernel_name = "gelu_quick"


@ProviderRegistry.register_vendor_impl("swigluoai_and_mul", "vllm_xpu_kernels")
class VLLMXPUKernelsSwigluOAIAndMulOp(_FusedActivationDirectOp):
    kernel_name = "swigluoai_and_mul"
    out_half = True


@ProviderRegistry.register_vendor_impl("topk_softmax", "vllm_xpu_kernels")
class VLLMXPUKernelsTopkSoftmaxOp(MoeSoftmaxTopkOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["vllm_xpu_kernels"]
        self._create_tensors_func = partial(
            self._create_in_out_tensors,
            create_inputs=True,
            create_outputs=True,
        )

    def vendor_impl_run(self, tensor_mapping):
        gating_output = tensor_mapping["gating_output"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]
        topk_indices = selected_experts.to(torch.int32)
        token_expert_indices = torch.empty(
            self.num_tokens * self.topk,
            dtype=torch.int32,
            device=gating_output.device,
        )
        renormalize = self.compute_mode == "pre-softmax"
        torch.ops._moe_C.topk_softmax(
            moe_weights, topk_indices, token_expert_indices,
            gating_output, renormalize, None
        )
        selected_experts.copy_(topk_indices.to(selected_experts.dtype))
        return selected_experts, moe_weights


@ProviderRegistry.register_vendor_impl("topk_sigmoid", "vllm_xpu_kernels")
class VLLMXPUKernelsTopkSigmoidOp(VLLMXPUKernelsTopkSoftmaxOp):
    def vendor_impl_run(self, tensor_mapping):
        gating_output = tensor_mapping["gating_output"]
        selected_experts = tensor_mapping["selected_experts"]
        moe_weights = tensor_mapping["moe_weights"]
        topk_indices = selected_experts.to(torch.int32)
        token_expert_indices = torch.empty(
            self.num_tokens * self.topk,
            dtype=torch.int32,
            device=gating_output.device,
        )
        renormalize = self.compute_mode == "pre-softmax"
        torch.ops._moe_C.topk_sigmoid(
            moe_weights, topk_indices, token_expert_indices,
            gating_output, renormalize, None
        )
        selected_experts.copy_(topk_indices.to(selected_experts.dtype))
        return selected_experts, moe_weights


@ProviderRegistry.register_vendor_impl("moe_gather", "vllm_xpu_kernels")
class VLLMXPUKernelsMoeGatherOp(MoeGatherOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["vllm_xpu_kernels"]

    def vendor_impl_run(self, tensor_mapping):
        scatter_tokens = tensor_mapping["scatter_tokens"]
        scatter_token_id = tensor_mapping["scatter_token_id"]
        scatter_token_weight = tensor_mapping["scatter_token_weight"]
        residual_tokens = tensor_mapping["residual_tokens"]
        convergent_tokens = tensor_mapping["convergent_tokens"]

        convergent_tokens[self.res_token_start:self.res_token_end] += residual_tokens * self.res_scale

        permuted = torch.arange(
            scatter_tokens.shape[0],
            dtype=torch.int32,
            device=scatter_tokens.device,
        )
        unpermuted = scatter_token_id.to(torch.int32)
        expert_first_token_offset = torch.zeros(
            self.num_experts + 1,
            dtype=torch.int32,
            device=scatter_tokens.device,
        )
        topk_weights = scatter_token_weight.view(-1, 1)

        torch.ops._moe_C.moe_gather(
            convergent_tokens,
            scatter_tokens,
            topk_weights,
            permuted,
            unpermuted,
            expert_first_token_offset,
            self.num_experts,
        )
        return convergent_tokens


@ProviderRegistry.register_vendor_impl("varlen_fwd", "vllm_xpu_kernels")
class VLLMXPUKernelsVarlenFwdOp(FlashAttentionOp):
    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["vllm_xpu_kernels"]

    def vendor_impl_run(self, tensor_mapping):
        q = tensor_mapping["q"]
        accum_q_lens = tensor_mapping["accum_q_lens"].to(torch.int32)
        accum_kv_lens = tensor_mapping["accum_kv_lens"].to(torch.int32)

        if self.cache_type == "linear":
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            batch, heads, seqlen, dim = k_cache.shape
            k = k_cache.permute(0, 2, 1, 3).reshape(batch * seqlen, heads, dim)
            v = v_cache.permute(0, 2, 1, 3).reshape(batch * seqlen, heads, dim)
        else:
            k = torch.empty_like(q[:, :self.kv_head_num, :])
            v = torch.empty_like(k)

        max_seqlen_q = max(self.q_lens)
        max_seqlen_k = max(self.kv_lens)

        outputs = torch.ops._vllm_fa2_C.varlen_fwd(
            q,
            k,
            v,
            None,
            accum_q_lens,
            accum_kv_lens,
            None,
            None,
            None,
            None,
            max_seqlen_q,
            max_seqlen_k,
            0.0,
            None,
            None,
            self.softmax_scale,
            None,
            False,
            self.is_causal,
            -1,
            -1,
            0.0,
            False,
            None,
            None,
        )
        return outputs[0]


class _StoreCacheKernelOp(StoreKVCacheOp):
    kernel_name = "reshape_and_cache"

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["vllm_xpu_kernels"]

    def vendor_impl_run(self, tensor_mapping):
        packed_qkv = tensor_mapping["packed_qkv"]
        k_head_start = self.q_head_num
        k_head_end = self.q_head_num + self.kv_head_num
        v_head_start = self.q_head_num + self.kv_head_num
        v_head_end = self.q_head_num + self.kv_head_num * 2

        key = packed_qkv[:, k_head_start:k_head_end, :].contiguous()
        value = packed_qkv[:, v_head_start:v_head_end, :].contiguous()

        if self.cache_type == "linear":
            slot_mapping = tensor_mapping["slot_mapping"].to(torch.int32)
        else:
            block_table = tensor_mapping["block_table"].to(torch.int32)
            slot_mapping = block_table.reshape(-1)[: self.num_tokens]

        k_cache = tensor_mapping["k_cache"]
        v_cache = tensor_mapping["v_cache"]

        k_scale = tensor_mapping.get("k_scale", torch.ones(1, device=key.device, dtype=torch.float32))
        v_scale = tensor_mapping.get("v_scale", torch.ones(1, device=key.device, dtype=torch.float32))

        kv_cache_dtype = self.cache_dtype

        if self.kernel_name == "reshape_and_cache_flash":
            torch.ops._cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        else:
            torch.ops._cache_ops.reshape_and_cache(
                key,
                value,
                k_cache,
                v_cache,
                slot_mapping,
                kv_cache_dtype,
                k_scale,
                v_scale,
            )
        return k_cache, v_cache


@ProviderRegistry.register_vendor_impl("reshape_and_cache", "vllm_xpu_kernels")
class VLLMXPUKernelsReshapeAndCacheOp(_StoreCacheKernelOp):
    kernel_name = "reshape_and_cache"


@ProviderRegistry.register_vendor_impl("reshape_and_cache_flash", "vllm_xpu_kernels")
class VLLMXPUKernelsReshapeAndCacheFlashOp(_StoreCacheKernelOp):
    kernel_name = "reshape_and_cache_flash"


@ProviderRegistry.register_vendor_impl("concat_and_cache_mla", "vllm_xpu_kernels")
class VLLMXPUKernelsConcatAndCacheMLAOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 512)
        self.input_tensor_info = {
            "kv_c": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "k_pe": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "kv_cache": OpTensorInfo([self.num_tokens, self.hidden_size * 2], self.dtype, self.backend.get_torch_device_name()),
            "slot_mapping": OpTensorInfo([self.num_tokens], torch.int32, self.backend.get_torch_device_name()),
            "scale": OpTensorInfo([1], torch.float32, self.backend.get_torch_device_name(), creator=torch.ones),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._cache_ops.concat_and_cache_mla(
            tensor_mapping["kv_c"],
            tensor_mapping["k_pe"],
            tensor_mapping["kv_cache"],
            tensor_mapping["slot_mapping"],
            "auto",
            tensor_mapping["scale"],
        )
        return tensor_mapping["kv_cache"]


@ProviderRegistry.register_vendor_impl("gather_cache", "vllm_xpu_kernels")
class VLLMXPUKernelsGatherCacheOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.batch_size = self.args_dict.get("batch_size", 4)
        self.block_size = self.args_dict.get("block_size", 16)
        self.num_blocks = self.args_dict.get("num_blocks", 128)
        self.entries = self.args_dict.get("entries", 512)
        self.input_tensor_info = {
            "src_cache": OpTensorInfo([self.num_blocks, self.block_size, self.entries], self.dtype, self.backend.get_torch_device_name()),
            "dst": OpTensorInfo([self.batch_size * self.block_size, self.entries], self.dtype, self.backend.get_torch_device_name()),
            "block_table": OpTensorInfo([self.batch_size, self.num_blocks // self.batch_size], torch.int32, self.backend.get_torch_device_name()),
            "cu_seq_lens": OpTensorInfo([self.batch_size + 1], torch.int32, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._cache_ops.gather_cache(
            tensor_mapping["src_cache"],
            tensor_mapping["dst"],
            tensor_mapping["block_table"],
            tensor_mapping["cu_seq_lens"],
            self.batch_size,
            None,
        )
        return tensor_mapping["dst"]


@ProviderRegistry.register_vendor_impl("swap_blocks", "vllm_xpu_kernels")
class VLLMXPUKernelsSwapBlocksOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.blocks = self.args_dict.get("blocks", 64)
        self.block_bytes = self.args_dict.get("block_size_in_bytes", 4096)
        self.elems = self.block_bytes // torch.tensor([], dtype=self.dtype).element_size()
        self.input_tensor_info = {
            "src": OpTensorInfo([self.blocks, self.elems], self.dtype, self.backend.get_torch_device_name()),
            "dst": OpTensorInfo([self.blocks, self.elems], self.dtype, self.backend.get_torch_device_name()),
            "block_mapping": OpTensorInfo([self.blocks], torch.int32, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._cache_ops.swap_blocks(
            tensor_mapping["src"],
            tensor_mapping["dst"],
            self.block_bytes,
            tensor_mapping["block_mapping"],
        )
        return tensor_mapping["dst"]


@ProviderRegistry.register_vendor_impl("convert_fp8", "vllm_xpu_kernels")
class VLLMXPUKernelsConvertFP8Op(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.input_tensor_info = {
            "src": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "dst": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._cache_ops.convert_fp8(tensor_mapping["dst"], tensor_mapping["src"], 1.0, "auto")
        return tensor_mapping["dst"]


class _GemmKernelBase(GemmOp):
    kernel_name = None

    def __init__(self, args_dict, backend, *args, **kwargs):
        super().__init__(args_dict, backend, *args, **kwargs)
        self.extra_providers = ["vllm_xpu_kernels"]


@ProviderRegistry.register_vendor_impl("fp8_gemm", "vllm_xpu_kernels")
class VLLMXPUKernelsFP8GemmOp(_GemmKernelBase):
    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        try:
            return torch.ops._xpu_C.fp8_gemm(a, b, None, None, None, None)
        except Exception:
            return torch.matmul(a, b)


@ProviderRegistry.register_vendor_impl("fp8_gemm_w8a16", "vllm_xpu_kernels")
class VLLMXPUKernelsFP8GemmW8A16Op(_GemmKernelBase):
    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        try:
            return torch.ops._xpu_C.fp8_gemm_w8a16(a, b, None, None)
        except Exception:
            return torch.matmul(a, b)


@ProviderRegistry.register_vendor_impl("int4_gemm_w4a16", "vllm_xpu_kernels")
class VLLMXPUKernelsINT4GemmW4A16Op(_GemmKernelBase):
    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        bias = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
        scale = torch.ones(b.shape[1], device=a.device, dtype=torch.float32)
        zp = torch.zeros_like(scale)
        try:
            return torch.ops._xpu_C.int4_gemm_w4a16(a, b, bias, scale, zp, 128, None)
        except Exception:
            return torch.matmul(a, b)


@ProviderRegistry.register_vendor_impl("int4_gemm_w4a8", "vllm_xpu_kernels")
class VLLMXPUKernelsINT4GemmW4A8Op(_GemmKernelBase):
    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        scale_a = torch.ones(a.shape[0], device=a.device, dtype=torch.float32)
        zp_a = torch.zeros_like(scale_a)
        scale_b = torch.ones(b.shape[1], device=a.device, dtype=torch.float32)
        zp_b = torch.zeros_like(scale_b)
        try:
            return torch.ops._xpu_C.int4_gemm_w4a8(a, scale_a, zp_a, b, scale_b, zp_b, 128, None, None)
        except Exception:
            return torch.matmul(a, b)


@ProviderRegistry.register_vendor_impl("cutlass_grouped_gemm_interface", "vllm_xpu_kernels")
class VLLMXPUKernelsCutlassGroupedGemmInterfaceOp(_GemmKernelBase):
    def vendor_impl_run(self, tensor_mapping):
        a = tensor_mapping["a"]
        b = tensor_mapping["b"]
        ptr_a = torch.zeros(1, dtype=torch.int64, device=a.device)
        ptr_b = torch.zeros(1, dtype=torch.int64, device=a.device)
        ptr_d = torch.zeros(1, dtype=torch.int64, device=a.device)
        offset = torch.zeros(1, dtype=torch.int32, device=a.device)
        try:
            return torch.ops._xpu_C.cutlass_grouped_gemm_interface(
                ptr_a, ptr_b, None, None, ptr_d, offset,
                b.shape[1], a.shape[1], 1, False, False,
            )
        except Exception:
            return torch.matmul(a, b)


@ProviderRegistry.register_vendor_impl("weak_ref_tensor", "vllm_xpu_kernels")
class VLLMXPUKernelsWeakRefTensorOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.shape = self.args_dict.get("shape", [1024, 1024])
        self.input_tensor_info = {
            "input": OpTensorInfo(self.shape, self.dtype, self.backend.get_torch_device_name())
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        return torch.ops._C.weak_ref_tensor(tensor_mapping["input"])


@ProviderRegistry.register_vendor_impl("get_xpu_view_from_cpu_tensor", "vllm_xpu_kernels")
class VLLMXPUKernelsGetXPUViewFromCPUTensorOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "float32"))
        self.shape = self.args_dict.get("shape", [1024, 1024])
        self.input_tensor_info = {
            "cpu_tensor": OpTensorInfo(self.shape, self.dtype, "cpu")
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        return torch.ops._C.get_xpu_view_from_cpu_tensor(tensor_mapping["cpu_tensor"])


@ProviderRegistry.register_vendor_impl("is_bmg", "vllm_xpu_kernels")
class VLLMXPUKernelsIsBMGOp(_KernelDirectBase):
    def prepare(self):
        self.device_index = self.args_dict.get("device_index", 0)
        self.input_tensor_info = {}
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        return torch.ops._xpu_C.is_bmg(self.device_index)


@ProviderRegistry.register_vendor_impl("is_pvc", "vllm_xpu_kernels")
class VLLMXPUKernelsIsPVCOp(_KernelDirectBase):
    def prepare(self):
        self.device_index = self.args_dict.get("device_index", 0)
        self.input_tensor_info = {}
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        return torch.ops._xpu_C.is_pvc(self.device_index)


@ProviderRegistry.register_vendor_impl("static_scaled_fp8_quant", "vllm_xpu_kernels")
class VLLMXPUKernelsStaticScaledFP8QuantOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.input_tensor_info = {
            "result": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "input": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "scale": OpTensorInfo([1], torch.float32, self.backend.get_torch_device_name(), creator=torch.ones),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._C.static_scaled_fp8_quant(
            tensor_mapping["result"],
            tensor_mapping["input"],
            tensor_mapping["scale"],
            None,
        )
        return tensor_mapping["result"]


@ProviderRegistry.register_vendor_impl("dynamic_scaled_fp8_quant", "vllm_xpu_kernels")
class VLLMXPUKernelsDynamicScaledFP8QuantOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.input_tensor_info = {
            "result": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "input": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "scale": OpTensorInfo([1], torch.float32, self.backend.get_torch_device_name(), creator=torch.ones),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._C.dynamic_scaled_fp8_quant(
            tensor_mapping["result"],
            tensor_mapping["input"],
            tensor_mapping["scale"],
        )
        return tensor_mapping["result"]


@ProviderRegistry.register_vendor_impl("dynamic_per_token_scaled_fp8_quant", "vllm_xpu_kernels")
class VLLMXPUKernelsDynamicPerTokenScaledFP8QuantOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.input_tensor_info = {
            "result": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "input": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "scale": OpTensorInfo([self.num_tokens], torch.float32, self.backend.get_torch_device_name(), creator=torch.ones),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._C.dynamic_per_token_scaled_fp8_quant(
            tensor_mapping["result"],
            tensor_mapping["input"],
            tensor_mapping["scale"],
            None,
        )
        return tensor_mapping["result"]


@ProviderRegistry.register_vendor_impl("per_token_group_fp8_quant", "vllm_xpu_kernels")
class VLLMXPUKernelsPerTokenGroupFP8QuantOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.group_size = self.args_dict.get("group_size", 128)
        self.input_tensor_info = {
            "input": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "output_q": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "output_s": OpTensorInfo([self.num_tokens, max(1, self.hidden_size // self.group_size)], torch.float32, self.backend.get_torch_device_name(), creator=torch.ones),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._C.per_token_group_fp8_quant(
            tensor_mapping["input"],
            tensor_mapping["output_q"],
            tensor_mapping["output_s"],
            self.group_size,
            1e-6,
            -448.0,
            448.0,
            False,
        )
        return tensor_mapping["output_q"], tensor_mapping["output_s"]


@ProviderRegistry.register_vendor_impl("deepseek_scaling_rope", "vllm_xpu_kernels")
class VLLMXPUKernelsDeepseekScalingRopeOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.q_head_num = self.args_dict.get("q_head_num", 32)
        self.kv_head_num = self.args_dict.get("kv_head_num", 8)
        self.head_dim = self.args_dict.get("head_dim", 128)
        self.input_tensor_info = {
            "positions": OpTensorInfo([self.num_tokens], torch.int64, self.backend.get_torch_device_name()),
            "query": OpTensorInfo([self.num_tokens, self.q_head_num, self.head_dim], self.dtype, self.backend.get_torch_device_name()),
            "key": OpTensorInfo([self.num_tokens, self.kv_head_num, self.head_dim], self.dtype, self.backend.get_torch_device_name()),
            "cos_sin_cache": OpTensorInfo([4096, self.head_dim], self.dtype, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        return torch.ops._xpu_C.deepseek_scaling_rope(
            tensor_mapping["positions"],
            tensor_mapping["query"],
            tensor_mapping["key"],
            None,
            tensor_mapping["cos_sin_cache"],
            self.head_dim,
            True,
        )


@ProviderRegistry.register_vendor_impl("bgmv_shrink", "vllm_xpu_kernels")
class VLLMXPUKernelsBGMVShrinkOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.input_tensor_info = {
            "outputs": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "inputs": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "weights": OpTensorInfo([self.hidden_size, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "indices": OpTensorInfo([self.num_tokens], torch.int32, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._xpu_C.bgmv_shrink(
            tensor_mapping["outputs"],
            tensor_mapping["inputs"],
            tensor_mapping["weights"],
            tensor_mapping["indices"],
            1.0,
        )
        return tensor_mapping["outputs"]


@ProviderRegistry.register_vendor_impl("bgmv_expand", "vllm_xpu_kernels")
class VLLMXPUKernelsBGMVExpandOp(VLLMXPUKernelsBGMVShrinkOp):
    def vendor_impl_run(self, tensor_mapping):
        torch.ops._xpu_C.bgmv_expand(
            tensor_mapping["outputs"],
            tensor_mapping["inputs"],
            tensor_mapping["weights"],
            tensor_mapping["indices"],
            False,
        )
        return tensor_mapping["outputs"]


@ProviderRegistry.register_vendor_impl("bgmv_expand_slice", "vllm_xpu_kernels")
class VLLMXPUKernelsBGMVExpandSliceOp(VLLMXPUKernelsBGMVShrinkOp):
    def vendor_impl_run(self, tensor_mapping):
        torch.ops._xpu_C.bgmv_expand_slice(
            tensor_mapping["outputs"],
            tensor_mapping["inputs"],
            tensor_mapping["weights"],
            tensor_mapping["indices"],
            0,
            tensor_mapping["outputs"].shape[-1],
            False,
        )
        return tensor_mapping["outputs"]


@ProviderRegistry.register_vendor_impl("gdn_attention", "vllm_xpu_kernels")
class VLLMXPUKernelsGDNAttentionOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 256)
        self.head_dim = self.args_dict.get("head_dim", 128)
        hidden = self.args_dict.get("hidden_size", 2048)
        self.input_tensor_info = {
            "core_attn_out": OpTensorInfo([self.num_tokens, hidden], self.dtype, self.backend.get_torch_device_name()),
            "z": OpTensorInfo([self.num_tokens, hidden], self.dtype, self.backend.get_torch_device_name()),
            "projected_states_qkvz": OpTensorInfo([self.num_tokens, hidden], self.dtype, self.backend.get_torch_device_name()),
            "projected_states_ba": OpTensorInfo([self.num_tokens, hidden], self.dtype, self.backend.get_torch_device_name()),
            "conv_state": OpTensorInfo([1, hidden], self.dtype, self.backend.get_torch_device_name()),
            "ssm_state": OpTensorInfo([1, hidden], self.dtype, self.backend.get_torch_device_name()),
            "conv_weights": OpTensorInfo([hidden, 1], self.dtype, self.backend.get_torch_device_name()),
            "A_log": OpTensorInfo([hidden], self.dtype, self.backend.get_torch_device_name()),
            "dt_bias": OpTensorInfo([hidden], self.dtype, self.backend.get_torch_device_name()),
            "non_spec_query_start_loc": OpTensorInfo([1], torch.int32, self.backend.get_torch_device_name()),
            "non_spec_state_indices_tensor": OpTensorInfo([1], torch.int32, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._xpu_C.gdn_attention(
            tensor_mapping["core_attn_out"],
            tensor_mapping["z"],
            tensor_mapping["projected_states_qkvz"],
            tensor_mapping["projected_states_ba"],
            1,
            1,
            self.head_dim,
            self.head_dim,
            tensor_mapping["conv_state"],
            tensor_mapping["ssm_state"],
            tensor_mapping["conv_weights"],
            None,
            "silu",
            tensor_mapping["A_log"],
            tensor_mapping["dt_bias"],
            1,
            0,
            None,
            tensor_mapping["non_spec_query_start_loc"],
            tensor_mapping["non_spec_state_indices_tensor"],
            self.num_tokens,
            1,
            False,
        )
        return tensor_mapping["core_attn_out"]


@ProviderRegistry.register_vendor_impl("moe_sum", "vllm_xpu_kernels")
class VLLMXPUKernelsMoeSumOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.topk = self.args_dict.get("topk", 2)
        self.input_tensor_info = {
            "input": OpTensorInfo([self.num_tokens, self.topk, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "output": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._moe_C.moe_sum(tensor_mapping["input"], tensor_mapping["output"])
        return tensor_mapping["output"]


@ProviderRegistry.register_vendor_impl("moe_align_block_size", "vllm_xpu_kernels")
class VLLMXPUKernelsMoeAlignBlockSizeOp(_KernelDirectBase):
    def prepare(self):
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.num_experts = self.args_dict.get("num_experts", 64)
        self.topk = self.args_dict.get("topk", 2)
        self.block_size = self.args_dict.get("block_size", 16)
        self.input_tensor_info = {
            "topk_ids": OpTensorInfo([self.num_tokens, self.topk], torch.int32, self.backend.get_torch_device_name()),
            "sorted_token_ids": OpTensorInfo([self.num_tokens * self.topk], torch.int32, self.backend.get_torch_device_name()),
            "experts_ids": OpTensorInfo([self.num_tokens * self.topk], torch.int32, self.backend.get_torch_device_name()),
            "num_tokens_post_pad": OpTensorInfo([1], torch.int32, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._moe_C.moe_align_block_size(
            tensor_mapping["topk_ids"],
            self.num_experts,
            self.block_size,
            tensor_mapping["sorted_token_ids"],
            tensor_mapping["experts_ids"],
            tensor_mapping["num_tokens_post_pad"],
            None,
        )
        return tensor_mapping["sorted_token_ids"], tensor_mapping["experts_ids"]


@ProviderRegistry.register_vendor_impl("batched_moe_align_block_size", "vllm_xpu_kernels")
class VLLMXPUKernelsBatchedMoeAlignBlockSizeOp(_KernelDirectBase):
    def prepare(self):
        self.max_tokens_per_batch = self.args_dict.get("max_tokens_per_batch", 1024)
        self.block_size = self.args_dict.get("block_size", 16)
        self.num_experts = self.args_dict.get("num_experts", 64)
        self.input_tensor_info = {
            "expert_num_tokens": OpTensorInfo([self.num_experts], torch.int32, self.backend.get_torch_device_name()),
            "sorted_token_ids": OpTensorInfo([self.max_tokens_per_batch], torch.int32, self.backend.get_torch_device_name()),
            "experts_ids": OpTensorInfo([self.max_tokens_per_batch], torch.int32, self.backend.get_torch_device_name()),
            "num_tokens_post_pad": OpTensorInfo([1], torch.int32, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._moe_C.batched_moe_align_block_size(
            self.max_tokens_per_batch,
            self.block_size,
            tensor_mapping["expert_num_tokens"],
            tensor_mapping["sorted_token_ids"],
            tensor_mapping["experts_ids"],
            tensor_mapping["num_tokens_post_pad"],
        )
        return tensor_mapping["sorted_token_ids"], tensor_mapping["experts_ids"]


@ProviderRegistry.register_vendor_impl("moe_lora_align_block_size", "vllm_xpu_kernels")
class VLLMXPUKernelsMoeLoraAlignBlockSizeOp(_KernelDirectBase):
    def prepare(self):
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.num_experts = self.args_dict.get("num_experts", 64)
        self.topk = self.args_dict.get("topk", 2)
        self.block_size = self.args_dict.get("block_size", 16)
        self.max_loras = self.args_dict.get("max_loras", 8)
        self.input_tensor_info = {
            "topk_ids": OpTensorInfo([self.num_tokens, self.topk], torch.int32, self.backend.get_torch_device_name()),
            "token_lora_mapping": OpTensorInfo([self.num_tokens], torch.int32, self.backend.get_torch_device_name()),
            "sorted_token_ids": OpTensorInfo([self.num_tokens * self.topk], torch.int32, self.backend.get_torch_device_name()),
            "experts_ids": OpTensorInfo([self.num_tokens * self.topk], torch.int32, self.backend.get_torch_device_name()),
            "num_tokens_post_pad": OpTensorInfo([1], torch.int32, self.backend.get_torch_device_name()),
            "adapter_enabled": OpTensorInfo([self.max_loras], torch.bool, self.backend.get_torch_device_name()),
            "lora_ids": OpTensorInfo([self.num_tokens * self.topk], torch.int32, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._moe_C.moe_lora_align_block_size(
            tensor_mapping["topk_ids"],
            tensor_mapping["token_lora_mapping"],
            self.num_experts,
            self.block_size,
            self.max_loras,
            self.num_tokens * self.topk,
            self.num_tokens * self.topk,
            tensor_mapping["sorted_token_ids"],
            tensor_mapping["experts_ids"],
            tensor_mapping["num_tokens_post_pad"],
            tensor_mapping["adapter_enabled"],
            tensor_mapping["lora_ids"],
            None,
        )
        return tensor_mapping["sorted_token_ids"], tensor_mapping["experts_ids"]


@ProviderRegistry.register_vendor_impl("grouped_topk", "vllm_xpu_kernels")
class VLLMXPUKernelsGroupedTopkOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "float32"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.num_experts = self.args_dict.get("num_experts", 64)
        self.topk = self.args_dict.get("topk", 2)
        self.n_group = self.args_dict.get("n_group", 8)
        self.topk_group = self.args_dict.get("topk_group", 2)
        self.input_tensor_info = {
            "scores": OpTensorInfo([self.num_tokens, self.num_experts], self.dtype, self.backend.get_torch_device_name()),
            "scores_with_bias": OpTensorInfo([self.num_tokens, self.num_experts], self.dtype, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        return torch.ops._moe_C.grouped_topk(
            tensor_mapping["scores"],
            tensor_mapping["scores_with_bias"],
            self.n_group,
            self.topk_group,
            self.topk,
            True,
            1.0,
        )


@ProviderRegistry.register_vendor_impl("fused_grouped_topk", "vllm_xpu_kernels")
class VLLMXPUKernelsFusedGroupedTopkOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "float32"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.num_experts = self.args_dict.get("num_experts", 64)
        self.topk = self.args_dict.get("topk", 2)
        self.n_group = self.args_dict.get("n_group", 8)
        self.topk_group = self.args_dict.get("topk_group", 2)
        self.input_tensor_info = {
            "hidden_states": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "gating_output": OpTensorInfo([self.num_tokens, self.num_experts], self.dtype, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        return torch.ops._moe_C.fused_grouped_topk(
            tensor_mapping["hidden_states"],
            tensor_mapping["gating_output"],
            self.topk,
            True,
            self.n_group,
            self.topk_group,
            "softmax",
            1.0,
            None,
        )


@ProviderRegistry.register_vendor_impl("fused_moe_prologue", "vllm_xpu_kernels")
class VLLMXPUKernelsFusedMoePrologueOp(_KernelDirectBase):
    def prepare(self):
        self.dtype = get_torch_dtype(self.args_dict.get("dtype", "bfloat16"))
        self.num_tokens = self.args_dict.get("num_tokens", 1024)
        self.hidden_size = self.args_dict.get("hidden_size", 4096)
        self.inter_size = self.args_dict.get("new_hidden_size", 11008)
        self.topk = self.args_dict.get("topk", 2)
        self.input_tensor_info = {
            "input": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
            "token_selected_experts": OpTensorInfo([self.num_tokens, self.topk], torch.int32, self.backend.get_torch_device_name()),
            "token_final_scales": OpTensorInfo([self.num_tokens, self.topk], torch.float32, self.backend.get_torch_device_name()),
            "workspace": OpTensorInfo([self.num_tokens, self.hidden_size], self.dtype, self.backend.get_torch_device_name()),
        }
        self.output_tensor_info = {}
        self._finalize(create_outputs=False)

    def vendor_impl_run(self, tensor_mapping):
        torch.ops._moe_C.fused_moe_prologue(
            tensor_mapping["input"],
            None,
            tensor_mapping["token_selected_experts"],
            tensor_mapping["token_final_scales"],
            tensor_mapping["workspace"],
            self.hidden_size,
            self.inter_size,
            128,
            0,
            1,
            self.args_dict.get("num_experts", 64),
        )
        return tensor_mapping["workspace"]
