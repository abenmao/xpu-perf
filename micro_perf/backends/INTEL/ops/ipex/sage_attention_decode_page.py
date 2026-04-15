import sys
import pathlib
from functools import partial
import torch
sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)
import math

from core.op import ProviderRegistry
from core.ops.llm_ops import FlashAttentionOp
from core.utils import OpTensorInfo, calc_tensor_size


try:
    torch.ops.torch_ipex.sage_attn_decode_paged

    @ProviderRegistry.register_vendor_impl("sage_attention_decode_page", "ipex")
    class SADecodeOp(FlashAttentionOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)
            self.extra_providers = ["sage_attn_decode_paged_intel"]

            self.mode = args_dict.get("attn_mode", args_dict.get("mode", "decode"))
            self.batch_size = args_dict.get("batch_size", 20)
            self.q_seq_len = args_dict.get("q_len", args_dict.get("q_seq_len", 4))
            self.kv_seq_len = self.args_dict.get("k_seq_len", 8192)
            self.head_dim = args_dict.get("head_dim", 128)
            self.q_head_num = args_dict.get("q_head_num", 96)
            self.num_head_kv = args_dict.get("kv_head_num", 8)

            assert self.q_head_num % self.num_head_kv == 0, \
                f"num_head_q ({self.q_head_num}) must be divisible by num_head_kv ({self.num_head_kv})"


            self.dtype = args_dict.get("dtype", torch.bfloat16)

            self.scale = float(1.0 / math.sqrt(self.head_dim))

            self.max_seqlen_q = self.q_seq_len if isinstance(self.q_seq_len, int) else max(self.q_seq_len)
            self.max_seqlen_k = self.kv_seq_len if isinstance(self.kv_seq_len, int) else max(self.kv_seq_len)

            self.cu_seqlen_q = self._compute_cu_seqlen(self.batch_size, self.q_seq_len)
            self.cu_seqlen_k = self._compute_cu_seqlen(self.batch_size, self.kv_seq_len)

            self.block_size = args_dict.get("block_size", 512)

            self.block_tables = self._generate_block_tables(self.batch_size, self.kv_seq_len, self.block_size)

            self.cu_seqlen_q = self.cu_seqlen_q.to(self.backend.get_torch_device_name())
            self.cu_seqlen_k = self.cu_seqlen_k.to(self.backend.get_torch_device_name())
            self.block_tables = self.block_tables.to(self.backend.get_torch_device_name())

            # Per-batch KV sequence lengths for SageAttention
            if isinstance(self.kv_seq_len, int):
                self.seq_lens = torch.tensor([self.kv_seq_len] * self.batch_size, dtype=torch.int32,
                                             device=self.backend.get_torch_device_name())
            else:
                self.seq_lens = torch.tensor(self.kv_seq_len, dtype=torch.int32,
                                             device=self.backend.get_torch_device_name())

            if self.head_dim == 128:
                self.chunk_size = 2048
                if self.q_head_num == 96 or self.q_head_num == 128 or self.q_head_num == 80 or self.q_head_num == 64:
                    self.chunk_size = 4096
            if self.head_dim == 64:
                self.chunk_size = 4096
                if self.kv_seq_len < 16384 and self.batch_size <= 10 and self.q_head_num <=64:
                    self.chunk_size = 2048
            self.chunk_num = (self.kv_seq_len + self.chunk_size - 1) // self.chunk_size

            self.alibi_slopes = None

            self.is_causal = args_dict.get("is_causal", self.mode == "decode")

        def prepare(self):

            self.arg_type = self.args_dict["arg_type"]
            if self.arg_type not in ["llm"]:
                raise NotImplementedError

            self.dtype = self.args_dict.get("dtype", "float16")
            if self.dtype not in ["float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            self.is_causal = self.args_dict.get("is_causal", True)
            if not self.is_causal:
                raise NotImplementedError

            self.q_head_num = self.args_dict["q_head_num"]
            self.kv_head_num = self.args_dict["kv_head_num"]
            self.head_dim = self.args_dict["head_dim"]

            self.batch_size = self.args_dict["batch_size"]
            self.q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len"))

            self.kv_seq_len = self.args_dict.get("k_seq_len", self.q_seq_len)

            self.softmax_scale = self.head_dim ** (-0.5)

            self.block_size = self.args_dict["block_size"]
            self.num_blocks = (self.kv_seq_len + self.block_size - 1) // self.block_size * self.batch_size

            if self.head_dim == 128:
                self.chunk_size = 2048
                if self.q_head_num == 96 or self.q_head_num == 128 or self.q_head_num == 80 or self.q_head_num == 64:
                    self.chunk_size = 4096
            if self.head_dim == 64:
                self.chunk_size = 4096
                if self.kv_seq_len < 16384 and self.batch_size <= 10 and self.q_head_num <=64:
                    self.chunk_size = 2048
            self.chunk_num = (self.kv_seq_len + self.chunk_size - 1) // self.chunk_size

            if self.block_size != 512:
                print("only support block_size 512")
                raise NotImplementedError

            self.input_tensor_info = {
                "q": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=torch.int8,
                    device=self.backend.get_torch_device_name()
                ),
                "k": OpTensorInfo(
                    shape=[self.num_blocks, self.block_size, self.kv_head_num, self.head_dim],
                    dtype=torch.int8,
                    device=self.backend.get_torch_device_name()
                ),
                "v": OpTensorInfo(
                    shape=[self.num_blocks, self.block_size, self.kv_head_num, self.head_dim],
                    dtype=torch.int8,
                    device=self.backend.get_torch_device_name()
                ),
                "qs": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, 1],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                ),
                "ks": OpTensorInfo(
                    shape=[self.batch_size, self.kv_seq_len, self.kv_head_num, 1],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                ),
                "vs": OpTensorInfo(
                    shape=[1, self.kv_head_num, self.head_dim],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                ),
                "out_reduce": OpTensorInfo(
                    shape=[self.batch_size, self.chunk_num, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                ),
                "out_lse": OpTensorInfo(
                    shape=[self.batch_size, self.chunk_num, self.q_seq_len, self.q_head_num],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                ),
                "output_max": OpTensorInfo(
                    shape=[self.batch_size, self.chunk_num, self.q_seq_len, self.q_head_num],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                )
            }

            self.output_tensor_info = {
                "out_reduce": OpTensorInfo(
                    shape=[self.batch_size, self.chunk_num, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                ),
                "out_lse": OpTensorInfo(
                    shape=[self.batch_size, self.chunk_num, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name()
                ),
                "out": OpTensorInfo(
                    shape=[self.batch_size, self.q_seq_len, self.q_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                )
            }

            # ESIMD kernels are not accurately measured by the XPU profiler
            # (PTI event subscription conflict), so use event-based timing.
            self.skip_profiling = True

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


            self.calc_flops = 0
            # here calc_flops are counted as int8 Tops

            for idx in range(self.batch_size):
                q_len = self.q_seq_len
                kv_len = self.kv_seq_len
                cache_len = kv_len - q_len

                # 1st gemm in precision int8
                gemm_flops = self.q_head_num * q_len * self.head_dim * kv_len * 2

                non_gemm_flops = 0

                # cast(int32->fp32)
                non_gemm_flops += self.q_head_num * kv_len * q_len

                # cast(fp16->fp32)
                non_gemm_flops += self.q_head_num * kv_len * q_len * 2

                # kq result dequant
                non_gemm_flops += self.q_head_num * kv_len * q_len * 2

                # softmax(fp32)
                non_gemm_flops += self.q_head_num * ((7+5/64) * kv_len * q_len + q_len * self.head_dim)

                # quant
                non_gemm_flops += self.q_head_num * kv_len * q_len * 4

                # 2nd gemm in precision int8
                gemm_flops += gemm_flops

                # cast(int32->fp32)
                non_gemm_flops += self.q_head_num * kv_len / 64 * q_len * self.head_dim

                # cast(v scale fp16->fp32)
                non_gemm_flops += self.q_head_num * q_len * self.head_dim

                # dequant
                non_gemm_flops += self.q_head_num * kv_len / 64 * q_len * self.head_dim

                # dequant
                non_gemm_flops += self.q_head_num * q_len * self.head_dim

                # acc
                non_gemm_flops += self.q_head_num * kv_len / 64 * q_len * self.head_dim

                # cast(fp32->fp16)
                non_gemm_flops += self.q_head_num * q_len * self.head_dim

                # non-gemm compute are in precision fp32 (4x int8)
                non_gemm_flops *= 4

                if self.is_causal:
                    flops_ratio = (q_len * kv_len - q_len * q_len / 2 )/ (q_len * kv_len)
                else:
                    flops_ratio = 1.

                self.calc_flops += gemm_flops * flops_ratio

                # not including non_gemm_flops
                # self.calc_flops += non_gemm_flops * flops_ratio

            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True,
            )

            self._run_func = self.flash_attention_run

        def _compute_cu_seqlen(self, batch_size, seq_lens):
            if isinstance(seq_lens, int):
                seq_lens = [seq_lens] * batch_size
            else:
                raise NotImplementedError
            cu_seqlen = [0]
            current = 0
            for seq_len in seq_lens:
                current += seq_len
                cu_seqlen.append(current)
            return torch.tensor(cu_seqlen, dtype=torch.int32)

        def _generate_block_tables(self, batch_size, cache_lens, block_size):
            if isinstance(cache_lens, int):
                cache_lens = [cache_lens] * batch_size
            else:
                raise NotImplementedError

            max_blocks = (cache_lens[0] + block_size - 1) // block_size
            block_tables = torch.zeros([batch_size, max_blocks], dtype=torch.int32)
            for i in range(batch_size):
                block_table = []
                for ii in range(max_blocks):
                    block_table.append(ii)
                tt = torch.tensor(block_table, dtype=torch.int32)
                tt_shuffle = torch.randperm(tt.size(0))
                tt_shuffle += tt.size(0) * i
                block_tables[i] = tt_shuffle

            # print("random block_tables are: ", block_tables)
            return block_tables

        def flash_attention_run(self, tensor_mapping):
            q = tensor_mapping["q"]
            k = tensor_mapping["k"]
            v = tensor_mapping["v"]
            qs = tensor_mapping["qs"]
            ks = tensor_mapping["ks"]
            vs = tensor_mapping["vs"]
            out = tensor_mapping["out"]
            out_reduce = tensor_mapping["out_reduce"]
            out_lse = tensor_mapping["out_lse"]
            output_max = tensor_mapping["output_max"]

            try:
                torch.ops.torch_ipex.sage_attn_decode_paged(
                    q,
                    k,
                    v,
                    qs,
                    ks,
                    vs,
                    self.cu_seqlen_q,  # [batch + 1]
                    self.cu_seqlen_k,  # [batch + 1]
                    self.block_tables, # [batch, num_max_seq_block]
                    out_reduce,
                    out_lse,
                    output_max,
                    out,
                    q.shape[0],   # max q_len in batches
                    self.chunk_size
                )
                torch.xpu.synchronize(q.device)
            except Exception:
                from sageattention import sageattn_decode_paged_esimd
                # Reshape IPEX 4D tensors to SageAttention 3D format
                # q: [B, q_seq_len, H_Q, D] -> [B, H_Q, D] (take first token)
                q_3d = q[:, 0, :, :]
                # qs: [B, q_seq_len, H_Q, 1] -> [B, H_Q, 1]
                qs_3d = qs[:, 0, :, :]
                # out: [B, q_seq_len, H_Q, D] -> [B, H_Q, D]
                out_3d = torch.zeros(
                    q.shape[0], q.shape[2], q.shape[3],
                    dtype=out.dtype, device=out.device
                )
                sageattn_decode_paged_esimd(
                    q_3d,
                    k,
                    v,
                    qs_3d,
                    ks,
                    vs,
                    self.block_tables,
                    self.seq_lens,
                    self.block_size,
                    out_3d,
                    self.chunk_size,
                )
                # Copy back: [B, H_Q, D] -> out[:, 0, :, :]
                out[:, 0, :, :] = out_3d

            return out

except Exception:
    pass
