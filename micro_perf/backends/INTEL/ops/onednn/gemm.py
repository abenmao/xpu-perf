import sys
import os
import re
import pathlib
import subprocess

sys.path.insert(
    0,
    str(pathlib.Path(__file__).absolute().parents[5])
)

from core.op import ProviderRegistry
from core.ops.tensor_gemm_ops import GemmOp


# benchdnn / locally-built oneDNN paths.
# We deliberately use the locally-built libdnnl.so under <oneDNN>/build/src
# instead of the one shipped with system oneAPI.
# Resolve oneDNN path relative to xpu-perf root (xpu-perf/../oneDNN), same
# convention as the sycl_tla provider.
# gemm.py -> onednn -> ops -> INTEL -> backends -> micro_perf -> xpu-perf
_XPU_PERF_ROOT = pathlib.Path(__file__).resolve().parents[5]
ONEDNN_DIR = str(_XPU_PERF_ROOT.parent / "oneDNN")
ONEDNN_BUILD_DIR = os.path.join(ONEDNN_DIR, "build")
ONEDNN_LIB_DIR = os.path.join(ONEDNN_BUILD_DIR, "src")
BENCHDNN_BIN = os.path.join(ONEDNN_BUILD_DIR, "tests", "benchdnn", "benchdnn")


# Map xpu-perf dtype names -> benchdnn --dt tokens
_DTYPE_MAP = {
    "float32": "f32",
    "tfloat32": "f32",
    "float16": "f16",
    "bfloat16": "bf16",
    "float8_e4m3": "f8_e4m3",
    "float8_e5m2": "f8_e5m2",
    "int8": "s8",
    "uint8": "u8",
    "int32": "s32",
}


try:
    if not os.path.isfile(BENCHDNN_BIN):
        print(
            f"[WARNING] benchdnn binary not found at {BENCHDNN_BIN}. "
            f"onednn gemm provider will NOT be available. "
            f"Please build oneDNN at {ONEDNN_BUILD_DIR}."
        )
        raise FileNotFoundError(BENCHDNN_BIN)

    if not os.path.isdir(ONEDNN_LIB_DIR) or not any(
        f.startswith("libdnnl.so") for f in os.listdir(ONEDNN_LIB_DIR)
    ):
        print(
            f"[WARNING] locally-built libdnnl.so not found under {ONEDNN_LIB_DIR}. "
            f"onednn gemm provider will NOT be available."
        )
        raise FileNotFoundError(os.path.join(ONEDNN_LIB_DIR, "libdnnl.so"))


    @ProviderRegistry.register_vendor_impl("gemm", "onednn")
    class OneDNNGemmOp(GemmOp):
        # benchdnn matmul supports common float and integer dtypes.
        SUPPORTED_DTYPES = list(_DTYPE_MAP.keys())

        def __init__(self, args_dict, backend, *args, **kwargs):
            super().__init__(args_dict, backend, *args, **kwargs)

            if self.dtype not in self.SUPPORTED_DTYPES:
                raise ValueError(
                    f"OneDNNGemmOp only supports dtype in {self.SUPPORTED_DTYPES}, "
                    f"got {self.dtype}"
                )

            # Run benchdnn and store parsed perf results
            self._onednn_result = self._run_benchdnn()

            # Override run/create-tensors to no-op; benchdnn already measured perf.
            self._run_func = lambda tensor_mapping: None
            self._create_tensors_func = lambda instance_num: [{}] * max(instance_num, 1)

        def _select_device_index(self):
            try:
                return int(self.backend.get_device())
            except Exception:
                return 0

        def _build_dt_arg(self):
            src_dt = _DTYPE_MAP[self.dtype]
            dst_dt = _DTYPE_MAP.get(self.dst_dtype, src_dt)
            return f"{src_dt}:{src_dt}:{dst_dt}"

        def _build_env(self):
            """Build env that forces benchdnn to load the locally-built
            libdnnl.so and pins the GPU device."""
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = (
                ONEDNN_LIB_DIR + ":" + env.get("LD_LIBRARY_PATH", "")
            )
            device_id = self._select_device_index()
            env["ZE_AFFINITY_MASK"] = str(device_id)
            return env

        def _verify_libdnnl_resolution(self, env):
            """Run `ldd` on the benchdnn binary with the prepared env and make
            sure libdnnl.so resolves to ONEDNN_LIB_DIR (i.e. our locally-built
            libdnnl), NOT the one shipped by system oneAPI. This is done OUTSIDE
            the perf-timing path so it does not pollute results."""
            try:
                ldd = subprocess.run(
                    ["ldd", BENCHDNN_BIN],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to run `ldd {BENCHDNN_BIN}`: {e}"
                )

            libdnnl_lines = [
                ln.strip()
                for ln in (ldd.stdout or "").splitlines()
                if "libdnnl.so" in ln
            ]
            if not libdnnl_lines:
                raise RuntimeError(
                    f"`ldd {BENCHDNN_BIN}` did not list libdnnl.so. "
                    f"ldd output:\n{ldd.stdout}\n{ldd.stderr}"
                )

            # Each line looks like: 'libdnnl.so.3 => /path/to/libdnnl.so.3 (0x...)'
            resolved_paths = []
            for ln in libdnnl_lines:
                if "=>" in ln:
                    rhs = ln.split("=>", 1)[1].strip()
                    # strip trailing '(0x...)' address
                    path = rhs.split(" ", 1)[0].strip()
                    resolved_paths.append(path)

            if not resolved_paths:
                raise RuntimeError(
                    f"Could not parse libdnnl.so resolution from ldd output:\n"
                    f"{ldd.stdout}"
                )

            expected_prefix = os.path.realpath(ONEDNN_LIB_DIR)
            for p in resolved_paths:
                real = os.path.realpath(p)
                if not real.startswith(expected_prefix):
                    raise RuntimeError(
                        f"benchdnn resolved libdnnl from '{p}' (real: '{real}'), "
                        f"which is NOT under expected local build dir "
                        f"'{expected_prefix}'. Refusing to run to avoid "
                        f"measuring system oneDNN. Full ldd lines:\n"
                        + "\n".join(libdnnl_lines)
                    )

            print(
                f"[OneDNNGemmOp] libdnnl resolution OK: "
                + ", ".join(resolved_paths)
            )

        def _run_benchdnn(self):
            """Run benchdnn --matmul and parse min/avg time, gflops, gbw."""
            dt_arg = self._build_dt_arg()
            prb = f"{self.M}x{self.K}:{self.K}x{self.N}"

            perf_tpl = (
                "%prb%"
                ",ms_min=%-time%,ms_avg=%0time%,ms_max=%+time%"
                ",Gflops_max=%-Gflops%,Gflops_avg=%0Gflops%,Gflops_min=%+Gflops%"
                ",Gbw_max=%-Gbw%,Gbw_avg=%0Gbw%,Gbw_min=%+Gbw%"
            )

            env = self._build_env()

            # Sanity check (NOT timed): make sure benchdnn will load our
            # locally-built libdnnl.so, not the system oneAPI one.
            self._verify_libdnnl_resolution(env)

            cmd = [
                BENCHDNN_BIN,
                "--matmul",
                "--engine=gpu",
                "--mode=P",
                "--cold-cache=all",
                f"--dt={dt_arg}",
                f"--perf-template={perf_tpl}",
                prb,
            ]

            print(
                f"[OneDNNGemmOp] LD_LIBRARY_PATH={env['LD_LIBRARY_PATH'].split(':')[0]} "
                f"Running: {' '.join(cmd)}"
            )

            result = subprocess.run(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=600,
            )

            output = (result.stdout or "") + (
                "\n" + result.stderr if result.stderr else ""
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"benchdnn failed (rc={result.returncode}):\n{output}"
                )

            print(f"[OneDNNGemmOp] Output:\n{output}")

            # Pick the perf-template line (it starts with the problem descriptor
            # that contains 'ms_min=').
            perf_lines = [
                ln for ln in output.splitlines() if "ms_min=" in ln
            ]
            if not perf_lines:
                raise RuntimeError(
                    f"Failed to find benchdnn perf line in output:\n{output}"
                )
            perf_line = perf_lines[-1]

            def _pick(name):
                m = re.search(rf"{name}=([0-9.eE+\-]+)", perf_line)
                return float(m.group(1)) if m else None

            ms_min = _pick("ms_min")
            ms_avg = _pick("ms_avg")
            ms_max = _pick("ms_max")
            gflops_max = _pick("Gflops_max")
            gflops_avg = _pick("Gflops_avg")
            gbw_max = _pick("Gbw_max")
            gbw_avg = _pick("Gbw_avg")

            if ms_min is None:
                raise RuntimeError(
                    f"Failed to parse ms_min from benchdnn perf line:\n{perf_line}"
                )

            return {
                "latency_ms_min": ms_min,
                "latency_ms_avg": ms_avg,
                "latency_ms_max": ms_max,
                "tflops_max": gflops_max / 1000.0 if gflops_max else None,
                "tflops_avg": gflops_avg / 1000.0 if gflops_avg else None,
                "gbw_max": gbw_max,
                "gbw_avg": gbw_avg,
            }

        def summary(self, latency_us, kernel_mapping={}):
            # Prefer min-time (= max-perf) reported by benchdnn for headline number.
            if self._onednn_result:
                ms_min = self._onednn_result.get("latency_ms_min")
                if ms_min is not None and ms_min > 0:
                    latency_us = ms_min * 1000.0
            return super().summary(latency_us, kernel_mapping)


except Exception as e:
    print(f"[OneDNNGemmOp] Failed to register: {e}")
    pass
