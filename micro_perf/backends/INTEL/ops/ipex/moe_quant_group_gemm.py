import sys
import pathlib
import torch
import os

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import MoeQuantGroupGemmOp

try:
    import intel_extension_for_pytorch as ipex
    from intel_extension_for_pytorch.llm.quantization.utils import XPUWoqActQuantMode
except Exception:
    XPUWoqActQuantMode = None

import torch.distributed as dist

#################################### ggemm_w4a8 interface #####################################################
# n_experts   number of experts  
# experts_token_count       xpu buffer, int32  shape (n_experts)   indicate each expert token count
# prepared by ggemm_preprocess from experts_token_count. And will be used to set oneDNN primitives.
#
# input_int8                 xpu buffer, int8, shape(m_scattered, k)
# input_scales       xpu buffer, fp16, shape(m_scattered, 1)
#
# weight_ba_full             xpu buffer, int32 (uint4 packed to int32, on k direction)   shape(k//8, n)
# Note: weight_ba_full stride is (1, k//8).  it is contiguous on k.
# weight_scales              xpu buffer, fp16, shape(k // 128, n)     128 quant group size.
# weight_zero_points         xpu buffer, must always be 8 since its SYM quant as u4
# output                     xpu buffer, fp16, shape(m_scattered, n)
def ggemm_w4a8(
    n_experts,
    experts_token_count,
    input_int8,
    input_scales,
    weight_ba_full,
    weight_scales,
    weight_zero_points,
    output):

    acc_aligned = 0
    acc = 0

    weight_ba_full = weight_ba_full.transpose(1, 2)

    for i in range(n_experts):
        currnt_t_exp = experts_token_count[i].item()
        if currnt_t_exp == 0:
            continue
        cur = input_int8[acc:acc+currnt_t_exp]
        cur_scale = input_scales[acc_aligned:acc_aligned+currnt_t_exp]
        
        torch.ops.torch_ipex.mm_w4a8(
            cur,
            cur_scale,
            None,
            weight_ba_full[i],
            weight_scales[i],
            weight_zero_points,
            XPUWoqActQuantMode.QUANT_A_PER_M_SYM,
            128,
            None,
            output[acc:acc+currnt_t_exp]
        )
        
        acc += currnt_t_exp
        currnt_t_exp_aligned = (currnt_t_exp + 255) // 256 * 256
        acc_aligned += currnt_t_exp_aligned

#################################### ggemm_w8a8 interface #####################################################
# n_experts                  number of experts  
# experts_token_count       xpu buffer, int32  shape (n_experts)   indicate each expert token count
# experts_token_count_host  cpu buffer (pin_memory)  host buffer of experts_token_count, 
# prepared by ggemm_preprocess from experts_token_count. And will be used to set oneDNN primitives.
#
# input_int8                 xpu buffer, int8, shape(m_scattered, k)
# input_scales               xpu buffer, fp16, shape(m_scattered, 1)
#
# weight_ba_full             xpu buffer, int8 shape(k, n)
# Note: weight_ba_full stride is (1, k).  it is contiguous on k.
# weight_scales              xpu buffer, fp16, shape(1, n)     per channel quant
# weight_zero_points         xpu buffer, must always be 0 since its SYM quant as int8
# output                     xpu buffer, fp16, shape(m_scattered, n)
def ggemm_w8a8(
    n_experts,
    experts_token_count,
    input_int8,
    input_scales,
    weight_ba_full,
    weight_scales,
    weight_zero_points,
    output):

    acc_aligned = 0
    acc = 0

    weight_ba_full = weight_ba_full.transpose(1, 2)

    for i in range(n_experts):
        currnt_t_exp = experts_token_count[i].item()
        if currnt_t_exp == 0:
            continue
        cur = input_int8[acc:acc+currnt_t_exp]
        cur_scale = input_scales[acc_aligned:acc_aligned+currnt_t_exp]
        
        torch.ops.torch_ipex.mm_w8a8(
            cur,
            cur_scale,
            None,
            weight_ba_full[i],
            weight_scales[i],
            weight_zero_points,
            output[acc:acc+currnt_t_exp]
        )

        acc += currnt_t_exp
        currnt_t_exp_aligned = (currnt_t_exp + 255) // 256 * 256
        acc_aligned += currnt_t_exp_aligned

class init_experts_tensor_count:
    def __init__(self, total_m, equal_div=False) -> None:
        self.total_m = total_m
        self.equal_div = equal_div

    def __call__(self, size, dtype, device):
        n_experts = size[0]
        experts_token_count = torch.zeros(
            n_experts, device=device, dtype=dtype
        )
        if not self.equal_div:
            lower_bound = 0
            upper_bound = n_experts
            for i in range(self.total_m):
                eachselection = torch.randint(lower_bound, upper_bound, (1,)).item()
                experts_token_count[eachselection] += 1
        elif self.total_m >= n_experts:
            token_per_exp = self.total_m // n_experts
            rest = self.total_m % n_experts
            for i in range(n_experts):
                experts_token_count[i] = token_per_exp
            
            if rest != 0:
                experts_token_count[-1] = rest
        else:
            for i in range(self.total_m):
                experts_token_count[i] = 1
        
        print("experts_token_count are: ", experts_token_count)
        
        return experts_token_count


try:
    torch.ops.torch_ipex.mm_w8a8

    @ProviderRegistry.register_vendor_impl("moe_quant_group_gemm", "ipex")
    class GGemmOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self.device = None
            self.local_rank = None
            self.global_rank = None

        def prepare(self):        

            self.arg_type = self.args_dict["arg_type"]
            if self.arg_type not in ["default", "llm"]:
                raise NotImplementedError(f"Unsupported arg_type: {self.arg_type}")

            self.dtype = self.args_dict["dtype"]
            quant_group_size = self.args_dict.get("quant_group_size", 128)
            self.n_experts = self.args_dict.get("n_experts", self.args_dict.get("num_experts", 1))
            ep_size = self.args_dict.get("ep_size", 1)
            self.n_experts = self.n_experts // ep_size
            self.topk = self.args_dict.get("topk", 1)

            if self.dtype == "w4a8":
                self.act_dtype = torch.int8
                self.weight_dtype = torch.int32
                self.scale_dtype = torch.float32
                self.out_dtype = torch.float16

            elif self.dtype == "w8a8" or (self.dtype == "int8" and self.n_experts > 1):
                self.dtype = "w8a8"  # normalize
                self.act_dtype = torch.int8
                self.weight_dtype = torch.int8
                self.scale_dtype = torch.float32
                dst_dtype_str = self.args_dict.get("dst_dtype", "float16")
                dst_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
                self.out_dtype = dst_map.get(dst_dtype_str, torch.float16)

            elif self.dtype in ["bfloat16", "float16", "float32", "tfloat32", "int8"]:
                if self.dtype in ["bfloat16", "float16"]:
                    self.torch_dtype = getattr(torch, self.dtype)
                    self.out_dtype = self.torch_dtype
                elif self.dtype == "float32":
                    self.torch_dtype = torch.float32
                    self.out_dtype = torch.float32
                elif self.dtype == "tfloat32":
                    self.torch_dtype = torch.float32
                    self.out_dtype = torch.float32
                elif self.dtype == "int8":
                    self.torch_dtype = torch.int8
                    self.out_dtype = torch.bfloat16
            else:
                raise NotImplementedError(f"Unsupported dtype: {self.dtype}")


            if self.arg_type == "llm":
                # Support both formats: m/n/k or num_tokens/hidden_size/new_hidden_size
                self.m = self.args_dict.get("m", self.args_dict.get("num_tokens", 0) // self.args_dict.get("sp_size", 1))
                self.k = self.args_dict.get("k", self.args_dict.get("hidden_size", 0))
                self.n = self.args_dict.get("n", self.args_dict.get("new_hidden_size", 0))
                self.m_scattered = self.m * self.topk  
                print("m_scattered",self.m_scattered)
                self.group_num = self.k // quant_group_size  if self.dtype=="w4a8" else 1
            elif self.arg_type == "default":
                self.M = self.args_dict["M"]
                self.K = self.args_dict["K"]
                self.N = self.args_dict["N"]


            if self.dtype == "w4a8":
                init_experts_tensor_count_func = init_experts_tensor_count(self.m_scattered, True)

                self.input_tensor_info = {
                    "a": OpTensorInfo(
                        shape=[self.m_scattered, self.k],  
                        dtype=self.act_dtype,
                        device=self.backend.get_torch_device_name()),
                    "b": OpTensorInfo(
                        shape=[self.n_experts, self.n, self.k // 8],  
                        dtype=self.weight_dtype,
                        device=self.backend.get_torch_device_name()),
                    "input_scales": OpTensorInfo(
                        shape=[self.m_scattered + 255*self.n_experts, 1], 
                        dtype=self.scale_dtype,
                        device=self.backend.get_torch_device_name()),
                    "scales": OpTensorInfo(
                        shape=[self.n_experts, self.group_num, self.n], 
                        dtype=self.scale_dtype,
                        device=self.backend.get_torch_device_name()),
                    "zero_points": OpTensorInfo(
                        shape=[1],  
                        dtype=torch.int8,
                        device=self.backend.get_torch_device_name()),
                    "experts_token_count": OpTensorInfo(
                        shape=[self.n_experts], 
                        dtype=torch.int32,
                        device="cpu",
                        creator=init_experts_tensor_count_func)
                }
                self.output_tensor_info = {
                    "c": OpTensorInfo(
                        shape=[self.m_scattered, self.n], 
                        dtype=self.out_dtype,
                        device=self.backend.get_torch_device_name())
                }
            elif self.dtype == "w8a8":
                init_experts_tensor_count_func = init_experts_tensor_count(self.m_scattered, True)
                trans_w = self.args_dict.get("trans_w", False)
                w_shape = [self.n_experts, self.k, self.n] if trans_w else [self.n_experts, self.n, self.k]

                self.input_tensor_info = {
                    "a": OpTensorInfo(
                        shape=[self.m_scattered, self.k],
                        dtype=self.act_dtype,
                        device=self.backend.get_torch_device_name()),
                    "b": OpTensorInfo(
                        shape=w_shape, 
                        dtype=self.weight_dtype, 
                        device=self.backend.get_torch_device_name()),  
                    "input_scales": OpTensorInfo(
                        shape=[self.m_scattered + 255*self.n_experts, 1], 
                        dtype=self.scale_dtype, 
                        device=self.backend.get_torch_device_name()),
                    "scales": OpTensorInfo(
                        shape=[self.n_experts, 1, self.n], 
                        dtype=self.scale_dtype, 
                        device=self.backend.get_torch_device_name()), 
                    "zero_points": OpTensorInfo(
                        shape=[1],
                        dtype=torch.int8, 
                        device=self.backend.get_torch_device_name()),
                    "experts_token_count": OpTensorInfo(
                        shape=[self.n_experts], 
                        dtype=torch.int32, 
                        device="cpu",
                        creator=init_experts_tensor_count_func)
                }
                self.output_tensor_info = {
                "c": OpTensorInfo(
                    shape=[self.m_scattered, self.n], 
                    dtype=self.out_dtype, 
                    device=self.backend.get_torch_device_name())
            }


            self.input_tensor_size = sum([calc_tensor_size(info) for info in self.input_tensor_info.values()])
            self.output_tensor_size = sum([calc_tensor_size(info) for info in self.output_tensor_info.values()])
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = self.input_tensor_size
            self.write_bytes = self.output_tensor_size
            self.io_bytes = self.read_bytes + self.write_bytes

            if self.dtype in ["w4a8","w8a8"]:
                total_tokens = sum(self.args_dict.get("experts_token_count", [self.m_scattered//self.n_experts]*self.n_experts))
                self.calc_flops = total_tokens * self.n * self.k * 2 
                print("total_tokens",total_tokens)
            else:

                self.calc_flops = self.M * self.N * self.K * 2


            self._run_func = self.ggemm_run


        def ggemm_run(self, tensor_mapping):

            if self.dtype == "w4a8":
                with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
                    ggemm_w4a8(
                        n_experts=self.n_experts,
                        experts_token_count=tensor_mapping["experts_token_count"],
                        input_int8=tensor_mapping["a"],
                        input_scales=tensor_mapping["input_scales"],
                        weight_ba_full=tensor_mapping["b"],
                        weight_scales=tensor_mapping["scales"],
                        weight_zero_points=tensor_mapping["zero_points"],
                        output=tensor_mapping["c"]
                    )

                    return tensor_mapping["c"]

            elif self.dtype == "w8a8":
                with torch.xpu.compute_eng(torch.xpu.XPUComputeEng.ONEDNN):
                    ggemm_w8a8(
                        self.n_experts,
                        tensor_mapping["experts_token_count"],
                        tensor_mapping["a"],
                        tensor_mapping["input_scales"],
                        tensor_mapping["b"],
                        tensor_mapping["scales"],
                        tensor_mapping["zero_points"],
                        tensor_mapping["c"]
                    )
                return tensor_mapping["c"]

            elif self.dtype in ["float32", "tfloat32", "float16", "bfloat16"]:
                a = tensor_mapping["a"]
                b = tensor_mapping["b"]
                c = tensor_mapping["c"]
                torch.matmul(a, b, out=c)
                return c

            elif self.dtype == "int8":
                a = tensor_mapping["a"]
                b = tensor_mapping["b"]
                a_scale = tensor_mapping["a_scale"]
                b_scale = tensor_mapping["b_scale"]
                c = tensor_mapping["c"]

                a_dequant = a.to(torch.float32) * a_scale.view(-1, 1)
                b_dequant = b.to(torch.float32) * b_scale.view(1, -1)
                c_out = torch.matmul(a_dequant, b_dequant).to(self.out_dtype)
                c.copy_(c_out)
                return c

            else:
                raise NotImplementedError(f"Unsupported dtype in run: {self.dtype}")



except Exception:
    pass

OP_MAPPING = {"torch": MoeQuantGroupGemmOp}
