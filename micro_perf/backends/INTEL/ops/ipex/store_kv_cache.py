import sys
import pathlib
from functools import partial
import random

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry, BasicOp
from core.utils import OpTensorInfo, calc_tensor_size
from core.ops.llm_ops import StoreKVCacheOp
from backends.INTEL.ops.utils import generate_decode_data, generate_prefill_data, generate_prefill_session_cache_data

try:
    import torch
    torch.ops.torch_ipex.store_kv_cache

    @ProviderRegistry.register_vendor_impl("store_kv_cache", "ipex")
    class StoreKVCacheIpexOp(BasicOp):
        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

        def prepare(self):
            self.arg_type = self.args_dict["arg_type"]
            if not self.arg_type in ["llm"]:
                raise NotImplementedError

            # src_dtype
            self.dtype = self.args_dict.get("dtype", "bfloat16")
            if not self.dtype in ["float16", "bfloat16"]:
                raise NotImplementedError
            self.torch_dtype = getattr(torch, self.dtype)

            # dst_dtype
            self.dst_dtype = self.args_dict.get("cache_dtype", self.args_dict.get("dst_dtype", "int8"))
            if not self.dst_dtype in ["int8"]:
                raise NotImplementedError
            self.torch_dst_dtype = getattr(torch, self.dst_dtype)

            # pre-defined attrs
            self.q_head_num = self.args_dict["q_head_num"]
            self.kv_head_num = self.args_dict["kv_head_num"]
            self.head_dim = self.args_dict["head_dim"]
            self.total_head_num = self.q_head_num + 2 * self.kv_head_num

            self.mode = self.args_dict.get("attn_mode", self.args_dict.get("mode", "prefill"))
            if self.mode == "prefill":
                # [q_seq_len, total_head_num, head_dim]
                self.batch_size = 1
                self.q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len"))
                self.cache_len = self.args_dict["cache_len"]

                self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                    generate_prefill_data(
                        self.q_seq_len,
                        self.cache_len
                    )

            elif self.mode == "prefill_session_cache":
                # [accumed_num_tokens, total_head_num, head_dim]
                self.batch_size = self.args_dict.get("batch_size", 1)
                self.q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len"))
                self.cache_len = self.args_dict["cache_len"]

                self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                    generate_prefill_session_cache_data(
                        self.batch_size,
                        self.q_seq_len,
                        self.cache_len
                    )

            elif self.mode == "decode":
                # [batch_size * q_seq_len, total_head_num, head_dim]
                self.batch_size = self.args_dict.get("batch_size", 1)
                self.q_seq_len = self.args_dict.get("q_len", self.args_dict.get("q_seq_len"))
                self.cache_len = self.args_dict["cache_len"]

                self.q_lens, self.accum_q_lens, self.cache_lens, self.cache_slot_ids, self.kv_lens = \
                    generate_decode_data(
                        self.batch_size,
                        self.q_seq_len,
                        self.cache_len
                    )

            else:
                raise NotImplementedError

            # accum q_lens
            self.num_tokens = sum(self.q_lens)
            # accum cache_lens
            self.num_cache_tokens = sum(self.cache_lens)
            # max q_len + cache_len
            self.max_kv_len = max(self.kv_lens)
            self.max_q_len = max(self.q_lens)

            self.input_tensor_info = {
                "packed_qkv": OpTensorInfo(
                    shape=[self.num_tokens, self.total_head_num, self.head_dim],
                    dtype=self.torch_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "q_lens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.q_lens, dtype=dtype, device=device)
                ),
                "accum_q_lens": OpTensorInfo(
                    shape=[self.batch_size + 1],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.accum_q_lens, dtype=dtype, device=device)
                ),
                "cache_lens": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.cache_lens, dtype=dtype, device=device)
                ),
                "cache_slot_ids": OpTensorInfo(
                    shape=[self.batch_size],
                    dtype=torch.int32,
                    device=self.backend.get_torch_device_name(),
                    creator=lambda size, dtype, device: torch.tensor(self.cache_slot_ids, dtype=dtype, device=device)
                ),
                "k_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                    dtype=self.torch_dst_dtype,
                    device=self.backend.get_torch_device_name()
                ),
                "v_cache": OpTensorInfo(
                    shape=[self.batch_size, self.kv_head_num, self.max_kv_len, self.head_dim],
                    dtype=self.torch_dst_dtype,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.empty
                ),
                "k_scale": OpTensorInfo(
                    shape=[self.kv_head_num, self.head_dim],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                ),
                "v_scale": OpTensorInfo(
                    shape=[self.kv_head_num, self.head_dim],
                    dtype=torch.float32,
                    device=self.backend.get_torch_device_name(),
                    creator=torch.ones
                )
            }

            self.output_tensor_info = {

            }

            # calculator
            self.input_tensor_size = sum([
                calc_tensor_size(info) for info in self.input_tensor_info.values()
            ])
            self.output_tensor_size = 0
            self.tensor_size = self.input_tensor_size + self.output_tensor_size

            self.read_bytes = \
                calc_tensor_size(self.input_tensor_info["packed_qkv"]) / self.total_head_num * 2 * self.kv_head_num + \
                calc_tensor_size(self.input_tensor_info["q_lens"]) + \
                calc_tensor_size(self.input_tensor_info["accum_q_lens"]) + \
                calc_tensor_size(self.input_tensor_info["cache_lens"]) + \
                calc_tensor_size(self.input_tensor_info["cache_slot_ids"]) + \
                calc_tensor_size(self.input_tensor_info["k_scale"]) + \
                calc_tensor_size(self.input_tensor_info["v_scale"])

            self.write_bytes = \
                calc_tensor_size(self.input_tensor_info["k_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens + \
                calc_tensor_size(self.input_tensor_info["v_cache"]) / self.batch_size / self.max_kv_len * self.num_tokens

            self.io_bytes = self.read_bytes + self.write_bytes

            # creator func
            self._create_tensors_func = partial(
                self._create_in_out_tensors,
                create_inputs=True,
                create_outputs=True
            )

            # run func
            self._run_func = self.store_kv_cache_run

        def store_kv_cache_run(self, tensor_mapping):
            packed_qkv = tensor_mapping["packed_qkv"]
            q_lens = tensor_mapping["q_lens"]
            accum_q_lens = tensor_mapping["accum_q_lens"]
            cache_lens = tensor_mapping["cache_lens"]
            cache_slot_ids = tensor_mapping["cache_slot_ids"]
            k_cache = tensor_mapping["k_cache"]
            v_cache = tensor_mapping["v_cache"]
            k_scale = tensor_mapping["k_scale"]
            v_scale = tensor_mapping["v_scale"]

            torch.ops.torch_ipex.store_kv_cache(
                packed_qkv, q_lens, accum_q_lens, cache_lens, cache_slot_ids, k_cache, v_cache, k_scale,
                v_scale, self.max_q_len)
            return k_cache, v_cache



except Exception:
    pass

OP_MAPPING = {"torch": StoreKVCacheOp}
