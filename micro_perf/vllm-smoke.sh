#!/bin/bash
set -ex

python launch.py --workload /root/code/xpu-perf/micro_perf/workloads/llm/single_test_ops/vllm_xpu_kernels_smoke.json --backend INTEL
