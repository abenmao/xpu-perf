import os
import sys
import pathlib
import importlib.util
import random
import torch

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[4])
)

from core.op import ProviderRegistry
from core.ops.vector_index_ops import ScatterOp

try:
    _OP_DIR = pathlib.Path(__file__).resolve().parent
    _SYCL_SO = _OP_DIR / "scatter_sycl.so"

    @ProviderRegistry.register_vendor_impl("scatter", "sycl_ext")
    class SyclExtScatterOp(ScatterOp):
        _SO_MODULE = None

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            self._verify = int(self.args_dict.get(
                "sycl_ext_verify",
                os.getenv("SYCL_EXT_SCATTER_VERIFY", "0"),
            ))

            self._sycl_so = self._load_sycl_so()

            if self._verify:
                self._verify_once()

            self._run_func = self._run_sycl_scatter

        @classmethod
        def _load_sycl_so(cls):
            if cls._SO_MODULE is not None:
                return cls._SO_MODULE

            if not _SYCL_SO.is_file():
                raise FileNotFoundError(
                    f"scatter sycl extension not found: {_SYCL_SO}. "
                    "Please run backends/INTEL/ops/sycl_ext/build.sh first."
                )

            spec = importlib.util.spec_from_file_location(
                "scatter_sycl", str(_SYCL_SO)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls._SO_MODULE = module
            return cls._SO_MODULE

        def create_tensors(self, instance_num):
            cur_device = self.backend.get_torch_device_name()

            all_tensor_list = []
            for _ in range(instance_num):
                tensor_mapping = {}

                src = torch.randn(
                    size=(self.src_batch_size, self.dim_size),
                    dtype=self.torch_dtype,
                    device=cur_device,
                )

                random_index = []
                for value in range(self.src_batch_size):
                    random_index.append(value % self.dst_batch_size)
                random.shuffle(random_index)
                # 1D index for the SYCL kernel (not expanded)
                index = torch.tensor(
                    random_index,
                    dtype=torch.int64,
                    device=cur_device,
                )

                dst = torch.randn(
                    size=(self.dst_batch_size, self.dim_size),
                    dtype=self.torch_dtype,
                    device=cur_device,
                )

                tensor_mapping["src"] = src
                tensor_mapping["index"] = index
                tensor_mapping["dst"] = dst
                all_tensor_list.append(tensor_mapping)

            return all_tensor_list

        def _run_sycl_scatter(self, tensor_mapping):
            self._sycl_so.scatter_compute(
                tensor_mapping["dst"],
                tensor_mapping["src"],
                tensor_mapping["index"],
            )
            return None

        def _verify_once(self):
            tensors = self.create_tensors(1)[0]
            src = tensors["src"]
            idx_1d = tensors["index"]
            dst_sycl = tensors["dst"].clone()
            dst_ref = tensors["dst"].clone()

            # SYCL kernel
            self._sycl_so.scatter_compute(dst_sycl, src, idx_1d)

            # PyTorch reference
            idx_2d = idx_1d.view(-1, 1).expand_as(src)
            dst_ref.scatter_(0, idx_2d, src)

            if hasattr(torch, "xpu") and hasattr(torch.xpu, "synchronize"):
                torch.xpu.synchronize()

            if not torch.equal(dst_sycl, dst_ref):
                max_diff = (dst_sycl.float() - dst_ref.float()).abs().max().item()
                raise RuntimeError(
                    f"scatter sycl verify failed (mode={self._scatter_mode}, "
                    f"max_diff={max_diff})"
                )

        def summary(self, latency_us, kernel_mapping={}):
            kernel_name = "sycl_ext_scatter_vectorized"
            target_dict = super().summary(latency_us, [kernel_name])
            if target_dict:
                target_dict["sycl_ext_verify"] = self._verify
            return target_dict

except Exception as e:
    print(f"[SyclExtScatterOp] Failed to register: {e}")
    pass
