## ccl_ops
快速验证 `all_reduce`，`reduce_scatter`、`all_gather`、`all_to_all` 四个通信原语在启动瓶颈、带宽瓶颈两个场景下的性能。

Reference test command for Intel B60
```shell
CCL_SYCL_ALLTOALL_ARC_LL=1 CCL_SYCL_CCL_BARRIER=1 python3 launch.py --workload workloads/llm/single_test_ops/ccl_ops.json --backend INTEL --report_dir llm_ccl_ops_report
```