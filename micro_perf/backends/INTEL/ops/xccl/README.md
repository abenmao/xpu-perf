## ccl_ops

### Table of Contents
- [ccl\_ops](#ccl_ops)
  - [Table of Contents](#table-of-contents)
  - [all\_reduce](#all_reduce)
  - [reduce\_scatter](#reduce_scatter)
  - [all\_gather](#all_gather)
  - [all\_to\_all](#all_to_all)
  - [p2p](#p2p)
  - [host2device](#host2device)
  - [device2host](#device2host)
  - [device2device](#device2device)
- [llm ccl\_ops](#llm-ccl_ops)
  - [adjust\_batch\_size.sh](#adjust_batch_sizesh)

**Note:** Since the Intel B60 GPU is PCIe Gen5 x8, there are hardware issues when running all_reduce/all_gather/reduce_scatter/all_to_all tests with datasets of 4 GB and larger. It is recommended to use the [adjust_batch_size.sh](#adjust_batch_sizesh) script to adjust the test dataset upper limit to ensure stable testing.

### [all_reduce](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce)
```python
torch.distributed.all_reduce(tensor, op=<RedOpType.SUM: 0>, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

Reference test command for Intel B60
```shell
# By default, the test use all the devices on the server. Use ZE_AFFINITY_MASK=0,1 can specify only use device 0,1
python3 launch.py --workload workloads/xccl_ops/all_reduce.json --backend INTEL --report_dir xccl_ops_report
```

### [reduce_scatter](https://pytorch.org/docs/stable/distributed.html#torch.distributed.reduce_scatter_tensor)
```python
torch.distributed.reduce_scatter_tensor(output, input, op=<RedOpType.SUM: 0>, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size // world_size, dim_size] |

Reference test command for Intel B60
```shell
# By default, the test use all the devices on the server. Use ZE_AFFINITY_MASK=0,1 can specify only use device 0,1
python3 launch.py --workload workloads/xccl_ops/reduce_scatter.json --backend INTEL --report_dir xccl_ops_report
```

### [all_gather](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_into_tensor)
```python
torch.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size // world_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

Reference test command for Intel B60
```shell
# By default, the test use all the devices on the server. Use ZE_AFFINITY_MASK=0,1 can specify only use device 0,1
python3 launch.py --workload workloads/xccl_ops/all_gather.json --backend INTEL --report_dir xccl_ops_report
```

### [all_to_all](https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single)
```python
torch.distributed.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)[source]
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

Reference test command for Intel B60
```shell
# By default, the test use all the devices on the server. Use ZE_AFFINITY_MASK=0,1 can specify only use device 0,1
CCL_SYCL_CCL_BARRIER=1 python3 launch.py --workload workloads/xccl_ops/all_to_all.json --backend INTEL --report_dir xccl_ops_report
```


### [p2p](https://pytorch.org/docs/stable/distributed.html#torch.distributed.isend)
```python
torch.distributed.isend(tensor, dst, tag=0, group=None, async_op=False)
torch.distributed.irecv(tensor, src, tag=0, group=None, async_op=False)
```
| tensor_name | tensor_shape |
| ----------- | ------------ |
| input_tensor | [batch_size, dim_size] |
| output_tensor | [batch_size, dim_size] |

Given a set of devices, we perform pairwise bandwidth tests between every two devices to evaluate the link bandwidth performance of the entire system topology. The test configuration is fixed with a tensor shape of [1024, 2097152], a data type of int8, and a total data volume of 2 GiB.


### host2device

Measures host (CPU) to device (GPU) memory copy bandwidth using `tensor.copy_()`.

| tensor_name | tensor_shape | location |
| ----------- | ------------ | -------- |
| src | [batch_size * dim_size] | CPU |
| dst | [batch_size * dim_size] | GPU |

Reference test command for Intel B60
```shell
# By default, the test use all the devices on the server. Use ZE_AFFINITY_MASK=0,1 can specify only use device 0,1
python3 launch.py --workload workloads/xccl_ops/host2device.json --backend INTEL --report_dir xccl_ops_report
```

### device2host

Measures device (GPU) to host (CPU) memory copy bandwidth using `tensor.copy_()`.

| tensor_name | tensor_shape | location |
| ----------- | ------------ | -------- |
| src | [batch_size * dim_size] | GPU |
| dst | [batch_size * dim_size] | CPU |

Reference test command for Intel B60
```shell
# By default, the test use all the devices on the server. Use ZE_AFFINITY_MASK=0,1 can specify only use device 0,1
python3 launch.py --workload workloads/xccl_ops/device2host.json --backend INTEL --report_dir xccl_ops_report
```

### device2device

Measures device (GPU) to device (GPU) memory copy bandwidth using `tensor.copy_()`. Both src and dst tensors reside on the GPU.

| tensor_name | tensor_shape | location |
| ----------- | ------------ | -------- |
| src | [batch_size * dim_size] | GPU |
| dst | [batch_size * dim_size] | GPU |

Reference test command for Intel B60
```shell
# By default, Battlemage has the memory compression feature enabled, which results in
# higher memory bandwidth than the theoretical value. It is necessary to add the environment
# variables "RenderCompressedBuffersEnabled=0” and “NEOReadDebugKeys=1" to disable
# this feature when testing memory bandwidth."
NEOReadDebugKeys=1 RenderCompressedBuffersEnabled=0 python3 launch.py --workload workloads/xccl_ops/device2device.json --backend INTEL --report_dir xccl_ops_report
```

## llm ccl_ops
Quickly verify the performance of four communication primitives (`all_reduce`, `reduce_scatter`, `all_gather`, `all_to_all`) under two scenarios: launch latency bottleneck and bandwidth bottleneck.

Reference test command for Intel B60
```shell
CCL_SYCL_ALLTOALL_ARC_LL=1 CCL_SYCL_CCL_BARRIER=1 python3 launch.py --workload workloads/llm/single_test_ops/ccl_ops.json --backend INTEL --report_dir llm_ccl_ops_report
```

### adjust_batch_size.sh

A utility script to adjust or restore the `batch_size` upper limit in xccl_ops test files for Intel B60 GPU performance tuning.

**Usage:**
```bash
./adjust_batch_size.sh b60      # Apply B60 limits
./adjust_batch_size.sh restore  # Restore original limits (2097152)
```

**B60 batch_size limit configuration:**

| File | B60 Limit | Original Limit |
| ---- | -------- | -------- |
| all_reduce.json | 524288 | 2097152 |
| all_gather.json | 1048576 | 2097152 |
| reduce_scatter.json | 1048576 | 2097152 |
| all_to_all.json | 1048576 | 2097152 |


