"""
XCCL infer loop for Intel XPU backend.

Sorts each dispatch batch by world_size descending before execution to
avoid the CCL communicator deadlock on Intel XE GPUs (PCIe, no XeLink):
once a narrower subgroup has been used, broader subgroups on overlapping
ranks deadlock in the Level-Zero KVS handshake.
"""

import json
import os
import queue
import signal
import time
import traceback

import psutil
import prettytable
import torch
import torch.distributed as dist

# Timeout (seconds) for draining remaining items after the first get().
# dispatch() puts items in a tight CPU loop, so they arrive in <1 ms.
_DRAIN_TIMEOUT_S = 1.0


def _prepare_group_switch(backend, cpu_group):
    """Clean up XPU state on ALL ranks before ``dist.new_group()``."""
    backend.device_synchronize()
    backend.empty_cache()
    torch.xpu.set_device(backend.get_device())
    time.sleep(0.5)
    dist.barrier(group=cpu_group)


def _drain_batch(input_queue):
    """Read one dispatch batch.  Returns (batch_list, session_done)."""
    first = input_queue.get()
    if first is None:
        return [], True
    batch = [first]
    while True:
        try:
            item = input_queue.get(timeout=_DRAIN_TIMEOUT_S)
        except queue.Empty:
            break
        if item is None:
            return batch, True
        batch.append(item)
    return batch, False


def _gather_from_rank0(rank, world_size, cpu_group, payload):
    """all_gather_object wrapper; returns rank-0's payload on every rank."""
    if world_size <= 1:
        return payload
    area = [None] * world_size
    dist.all_gather_object(area, {"rank": rank, "d": payload}, group=cpu_group)
    return sorted(area, key=lambda x: x["rank"])[0]["d"]


def _execute_case(data, rank, world_size, local_device_id,
                  backend, cpu_group, groups, output_queue):
    """Run one benchmark case and put the result into output_queue."""
    case_idx, (op_name, op_provider, op_cls), task_case = data
    task_ws = task_case.get("world_size", 1)
    result_dict = {}

    if task_ws > world_size:
        if rank == 0:
            output_queue.put((case_idx, result_dict))
        return

    # Create subgroup on first use (always large-first due to sorting).
    if task_ws > 1 and task_ws not in groups:
        _prepare_group_switch(backend, cpu_group)
        groups[task_ws] = dist.new_group(ranks=list(range(task_ws)))

    # Create op instance on participating ranks.
    op_instance = None
    if rank < task_ws:
        try:
            op_instance = op_cls(
                task_case, backend,
                op_group=groups.get(task_ws), group_size=task_ws,
            )
            op_instance.is_concurrent = True
        except Exception:
            traceback.print_exc()

    # Verify all participating ranks created the op.
    area = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(
            area, {"rank": rank, "ok": op_instance is not None}, group=cpu_group)
    else:
        area[0] = {"rank": rank, "ok": op_instance is not None}
    if not all(x["ok"] for x in sorted(area, key=lambda x: x["rank"])[:task_ws]):
        if rank == 0:
            output_queue.put((case_idx, result_dict))
        return

    # Benchmark.
    target_dict = {}
    if rank < task_ws:
        try:
            target_dict = backend.perf(op_instance)
        except Exception:
            traceback.print_exc()

    # Gather results.
    area = [None] * world_size
    payload = {"rank": rank, "device_id": local_device_id,
               "target_dict": target_dict}
    if world_size > 1:
        dist.all_gather_object(area, payload, group=cpu_group)
    else:
        area[0] = payload
    ranked = sorted(area, key=lambda x: x["rank"])[:task_ws]

    target_list = [x["target_dict"] for x in ranked]
    if not all(target_list):
        if rank == 0:
            output_queue.put((case_idx, result_dict))
        return

    if rank == 0:
        result_dict = op_instance.merge_summary(target_list)
        pt = prettytable.PrettyTable(["key", "value"])
        pt.align = "l"
        for k, v in [("op_name", op_name), ("op_provider", op_provider),
                      ("rank", [x["rank"] for x in ranked]),
                      ("device_id", [x["device_id"] for x in ranked]),
                      ("idx", case_idx)]:
            pt.add_row([k, str(v)])
        print(f"{pt}\n{json.dumps(task_case)}\n"
              f"{json.dumps(result_dict, indent=4)}\n")
        output_queue.put((case_idx, result_dict))


def run_infer_loop(backend, local_process_rank, process_mapping,
                   input_queue, output_queue,
                   master_addr, xccl_port, node_world_size, node_rank):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    info = process_mapping[local_process_rank]

    psutil.Process(os.getpid()).cpu_affinity(info["numa_cores"])

    local_device_id = info["device_id"]
    local_ws = len(process_mapping)
    local_rank = local_process_rank
    world_size = local_ws * node_world_size
    rank = local_rank + node_rank * local_ws

    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": master_addr, "MASTER_PORT": str(xccl_port),
    })

    print(f"device_id: {local_device_id}, "
          f"local: {local_rank}/{local_ws}, world: {rank}/{world_size}")

    if world_size > 1:
        backend.initialize_ccl(rank, world_size)
        print(f"xccl init done, rank: {rank}, world_size: {world_size}")
        backend.set_device(local_device_id)
        t = torch.ones(1, dtype=torch.float32, device=backend.get_torch_device_name())
        t *= 1 if rank < world_size // 2 else -1
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        print(t)
        cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    else:
        backend.set_device(local_device_id)
        cpu_group = None

    if local_process_rank == 0:
        output_queue.put("success")

    groups = {world_size: None}  # ws -> ProcessGroup (None = default)

    # Main loop: buffer one dispatch batch, sort ws descending, execute.
    while True:
        if rank == 0:
            batch, done = _drain_batch(input_queue)
            batch.sort(key=lambda x: x[2].get("world_size", 1), reverse=True)
            meta = {"count": len(batch), "done": done}
        else:
            batch, meta = None, None

        meta = _gather_from_rank0(rank, world_size, cpu_group, meta)

        for i in range(meta["count"]):
            case_data = batch[i] if rank == 0 else None
            case_data = _gather_from_rank0(rank, world_size, cpu_group, case_data)
            _execute_case(case_data, rank, world_size, local_device_id,
                          backend, cpu_group, groups, output_queue)

        if meta["done"]:
            break
