import os
import sys
import csv
import json
import pathlib
import random
import shutil
import subprocess
from datetime import timedelta

import torch
try:
    import intel_extension_for_pytorch
except:
    pass
import torch.distributed as dist
import time

FILE_DIR = pathlib.Path(__file__).parent.absolute()
BACKEND_DIR = FILE_DIR.parent
MICRO_PERF_DIR = BACKEND_DIR.parent

sys.path.insert(0, str(MICRO_PERF_DIR))

from core.backend import Backend
from core.utils import suppress_stdout_stderr
try:
    from backends.INTEL.provider_intel import INTEL_PROVIDER
except:
    INTEL_PROVIDER = {}


class BackendINTEL(Backend):
    def __init__(self):
        super().__init__()

    def get_backend_info(self):
        info_dict = {}

        # device info
        info_dict["device_name"] = torch.xpu.get_device_name(0)
        info_dict["device_count"] = torch.xpu.device_count()

        device_properties = torch.xpu.get_device_properties(0)
        info_dict["device_memory_mb"] = device_properties.total_memory / (1024 ** 2)

        info_dict["torch_version"] = torch.__version__

        return info_dict

    def get_provider_info(self):
        return INTEL_PROVIDER

    def clean_extra_files(self):
        PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling")
        if PROFILER_DIR.exists():
            shutil.rmtree(PROFILER_DIR)


    """
    device management related
    """
    def get_torch_device_name(self):
        return "xpu"
    
    def get_device_name(self, index = 0):
        return torch.xpu.get_device_name(index)
    
    def get_device_properties(self, index = 0):
        return torch.xpu.get_device_properties(index)
    
    def get_mem_info(self, index = 0):
        total_memory = torch.xpu.get_device_properties(index).total_memory
        allocated_memory = torch.xpu.memory_allocated(index)
        cached_memory = torch.xpu.memory_reserved(index)
        free_memory = (total_memory - allocated_memory)
        return (free_memory, total_memory)
    
    def get_device_count(self):
        device_count = torch.xpu.device_count()
        return device_count, list(range(device_count))
    
    def set_device(self, device_index : int):
        torch.xpu.set_device(device_index)

    def get_device(self):
        return torch.xpu.current_device()

    def device_synchronize(self):
        torch.xpu.synchronize()

    def empty_cache(self):
        torch.xpu.empty_cache()

    """
    ccl related
    """
    def get_dist_module(self):
        return dist
    
    """
    For Pytorch 2.9.1 and above, XCCL is added as distributed communication backend for Intel GPUs.
    XCCL is a distributed backend that enables various distributed training paradigms
    such as DDP (DistributedDataParallel), FSDP (FullyShardedDataParallel),
    PP (pipeline parallelism), and TP (tensor parallelism) on XPU devices.
    XCCL provides all PyTorch communication operations (allreduce, allgather, reducescatter),
    and can be transparently applied on XPU or explicitly specified as "xccl" backend.
    Add a check here to use XCCL if available, otherwise fallback to CCL.
    """
    def get_dist_backend(self):
        if dist.distributed_c10d.is_xccl_available():
            return "xccl"
        else:
            try:
                import oneccl_bindings_for_pytorch
            except ImportError:
                raise RuntimeError(
                    "Neither XCCL backend (PyTorch >= 2.9.1) nor oneccl_bindings_for_pytorch is available. "
                    "Please upgrade PyTorch or install oneccl_bindings_for_pytorch for CCL backend support."
                )
            return "ccl"
    

    def core_perf(
        self, op_instance, 
        warmup_iterations, prefer_iterations, 
        tensor_list, 
        profiling=True
    ):
        op_group = op_instance.op_group
        group_size = op_instance.group_size

        if not op_instance.is_concurrent and profiling and not getattr(op_instance, 'skip_profiling', False):
            process_id = os.getpid()
            PROFILER_DIR = pathlib.Path.cwd().joinpath("profiling", f"{process_id}")
            PROFILER_DIR.mkdir(parents=True, exist_ok=True)
            TRACE_FILE = PROFILER_DIR.joinpath("trace.json")

            # profiling
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.XPU], 
                schedule=torch.profiler.schedule(
                    wait=0, 
                    warmup=warmup_iterations, 
                    active=prefer_iterations, 
                    repeat=1
                ), 
                on_trace_ready=lambda prof: prof.export_chrome_trace(f"{TRACE_FILE}")
            ) as prof:
                for i in range(prefer_iterations + warmup_iterations):
                    op_instance.core_run(tensor_list[i % len(tensor_list)])
                    self.device_synchronize()
                    prof.step()

            # parse and delete profiling json file
            average_latency = 0.
            flash_attn_latency = 0.
            kernel_latency_list = {}
            if TRACE_FILE.exists():
                profiling_data = json.loads(TRACE_FILE.read_text())
                for event in profiling_data["traceEvents"]:
                    if event.get("cat", None) in ["kernel", "gpu_memcpy"]:
                        kernel_name = event["name"]
                        kernel_latency = event["dur"]
                        if kernel_name not in kernel_latency_list:
                            kernel_latency_list[kernel_name] = []
                        kernel_latency_list[kernel_name].append(kernel_latency)

                for kernel_name, latency_list in kernel_latency_list.items():
                    average_latency += sum(latency_list)
                    if "micro_sdpa" in kernel_name or "cute::MhaName" in kernel_name:
                        flash_attn_latency += sum(latency_list)

                average_latency /= prefer_iterations
                flash_attn_latency /= prefer_iterations

                shutil.rmtree(PROFILER_DIR)

            return flash_attn_latency if flash_attn_latency > 0 else average_latency, list(kernel_latency_list.keys())
        
        else:
            for i in range(warmup_iterations):
                op_instance.core_run(tensor_list[i % len(tensor_list)])
            
            self.device_synchronize()

            # ESIMD kernels bypass PyTorch's dispatcher and submit directly to
            # the raw SYCL queue, so torch.xpu.Event cannot track them.
            # Use wall-clock timing instead (sync=True in binding ensures
            # queue->wait() blocks until kernel completes).
            if getattr(op_instance, 'skip_profiling', False):
                import time
                self.device_synchronize()
                self.op_group_barrier(op_group=op_group, group_size=group_size)
                start_time = time.perf_counter()
                for i in range(prefer_iterations):
                    op_instance.core_run(tensor_list[i % len(tensor_list)])
                end_time = time.perf_counter()
                self.device_synchronize()
                self.op_group_barrier(op_group=op_group, group_size=group_size)
                latency_us = (end_time - start_time) * 1e6 / prefer_iterations
                return latency_us, []
            
            start_event = torch.xpu.Event(enable_timing=True)
            end_event = torch.xpu.Event(enable_timing=True)

            self.device_synchronize()
            self.op_group_barrier(op_group=op_group, group_size=group_size)
            start_event.record()
            for i in range(prefer_iterations):
                op_instance.core_run(tensor_list[i % len(tensor_list)])
            end_event.record()
            end_event.synchronize()
            
            self.device_synchronize()
            self.op_group_barrier(op_group=op_group, group_size=group_size)

            latency_us = start_event.elapsed_time(end_event) * 1e3 / prefer_iterations
            return latency_us, []
