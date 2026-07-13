from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import modal

REMOTE_REPO = Path("/root/klein4B")
if REMOTE_REPO.is_dir() and str(REMOTE_REPO) not in sys.path:
    sys.path.insert(0, str(REMOTE_REPO))

from modal_bench_inference_nvfp4 import MODEL_VOLUME, VOLUME_MOUNT, image  # noqa: E402


APP = modal.App("klein4b-nsight-profile")
NSIGHT_IMAGE = image.apt_install(
    "cuda-nsight-systems-13-0",
    "cuda-nsight-compute-13-0",
).run_commands(
    "command -v nsys && nsys --version",
    "command -v ncu && ncu --version",
    "nsys profile --help | grep -q -- '--capture-range-end'",
    "nsys profile --help | grep -q -- '--force-overwrite'",
    "nsys export --help | grep -q -- '--force-overwrite'",
    "nsys stats --help | grep -q -- '--report'",
    "ncu --list-sets | grep -q 'full'",
    "ncu --help | grep -q -- '--graph-profiling'",
    "ncu --help | grep -q -- '--nvtx-include'",
    "ncu --help | grep -q -- '--kernel-name-base'",
    "ncu --help | grep -q -- '--replay-mode'",
    "ncu --help | grep -q -- '--target-processes'",
)
RUNNER = "/root/klein4B/nsight_profile_runner.py"


def _run(command: list[str], *, check: bool = True, output: Path | None = None) -> subprocess.CompletedProcess:
    print("NSIGHT_COMMAND " + json.dumps(command))
    lines: deque[str] = deque(maxlen=4096)
    output_file = output.open("w") if output is not None else None
    try:
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            lines.append(line)
            if output_file is not None:
                output_file.write(line)
                output_file.flush()
        return_code = process.wait()
    finally:
        if output_file is not None:
            output_file.close()
    completed = subprocess.CompletedProcess(command, return_code, "".join(lines), None)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")
    return completed


def _probe_systems_trace(run_dir: Path) -> str:
    probe_code = (
        "import torch; "
        "torch.ones(1, device='cuda'); "
        "torch.cuda.synchronize(); "
        "torch.cuda.cudart().cudaProfilerStart(); "
        "x=torch.ones(1024, device='cuda')+1; "
        "torch.cuda.synchronize(); "
        "torch.cuda.cudart().cudaProfilerStop(); "
        "assert x[0].item() == 2"
    )
    attempts = []
    for trace in ("cuda,nvtx", "cuda-sw,nvtx"):
        report_base = run_dir / f"nsys_probe_{trace.split(',', maxsplit=1)[0]}"
        completed = _run(
            [
                "nsys",
                "profile",
                f"--trace={trace}",
                "--sample=none",
                "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi",
                "--capture-range-end=stop",
                "--force-overwrite=true",
                f"--output={report_base}",
                "python",
                "-c",
                probe_code,
            ],
            check=False,
            output=run_dir / f"nsys_probe_{trace.split(',', maxsplit=1)[0]}.log",
        )
        report_exists = report_base.with_suffix(".nsys-rep").is_file()
        attempts.append(
            {"trace": trace, "exit_code": completed.returncode, "report_exists": report_exists}
        )
        if completed.returncode == 0 and report_exists:
            (run_dir / "nsys_probe.json").write_text(json.dumps(attempts, indent=2))
            return trace
    (run_dir / "nsys_probe.json").write_text(json.dumps(attempts, indent=2))
    raise RuntimeError(
        "Modal's gVisor runtime rejected both Nsight Systems CUDA collectors; "
        f"attempts={attempts}. Use a Colab or ordinary privileged Linux VM for Nsight Systems."
    )


def _collect_systems(run_dir: Path) -> dict:
    selected_trace = _probe_systems_trace(run_dir)
    report_base = run_dir / "klein4b_exact_e2e"
    command = [
        "nsys",
        "profile",
        f"--trace={selected_trace}",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        f"--output={report_base}",
        "python",
        RUNNER,
    ]
    _run(command, output=run_dir / "nsys_profile.log")

    report = report_base.with_suffix(".nsys-rep")
    if not report.is_file():
        raise RuntimeError(f"Nsight Systems did not create {report}")

    sqlite = report_base.with_suffix(".sqlite")
    _run(
        ["nsys", "export", "--type=sqlite", "--force-overwrite=true", f"--output={sqlite}", str(report)],
        output=run_dir / "nsys_export.log",
    )
    stats = _run(
        [
            "nsys",
            "stats",
            "--report=cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum",
            "--format=csv",
            str(report),
        ],
        check=False,
        output=run_dir / "nsys_stats.csv",
    )
    return {
        "report": str(report),
        "sqlite": str(sqlite),
        "stats_exit_code": stats.returncode,
        "trace": selected_trace,
    }


NCU_TARGETS = {
    "gemm": r"regex:.*(cutlass3x_sm120_bstensorop|MainloopSm120TmaWarpSpecializedBlockScaled).*",
    "attention": r"regex:.*flash_fwd_kernel.*",
    "activation_quantization": r"regex:.*triton_quantize_nvfp4_kernel.*",
}


def _collect_compute(run_dir: Path) -> dict:
    probe = _run(
        [
            "ncu",
            "--set=basic",
            "--launch-count=1",
            "python",
            "-c",
            (
                "import torch; "
                "x=torch.ones(1024, device='cuda'); "
                "torch.cuda.synchronize(); "
                "y=x+1; "
                "torch.cuda.synchronize(); "
                "assert y[0].item() == 2"
            ),
        ],
        check=False,
        output=run_dir / "ncu_permission_probe.log",
    )
    if probe.returncode != 0:
        print(f"NCU_COUNTERS_UNAVAILABLE exit_code={probe.returncode}")
        return {"permission_probe_exit_code": probe.returncode, "captures": {}}

    results = {}
    for name, kernel_filter in NCU_TARGETS.items():
        report = run_dir / f"klein4b_{name}"
        command = [
            "ncu",
            "--target-processes=all",
            "--set=full",
            "--nvtx",
            "--nvtx-include=klein_e2e_exact/",
            "--kernel-name-base=demangled",
            f"--kernel-name={kernel_filter}",
            "--launch-count=1",
            "--graph-profiling=node",
            "--replay-mode=kernel",
            "--force-overwrite",
            f"--export={report}",
            "python",
            RUNNER,
        ]
        completed = _run(command, check=False, output=run_dir / f"ncu_{name}.log")
        results[name] = {
            "exit_code": completed.returncode,
            "report": str(report.with_suffix(".ncu-rep")),
            "report_exists": report.with_suffix(".ncu-rep").is_file(),
        }
        if completed.returncode != 0:
            print(f"NCU_CAPTURE_UNAVAILABLE target={name} exit_code={completed.returncode}")
    return {"permission_probe_exit_code": 0, "captures": results}


@APP.function(
    image=NSIGHT_IMAGE,
    gpu="RTX-PRO-6000",
    timeout=60 * 60 * 4,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
)
def profile(mode: str = "all") -> dict:
    if mode not in {"systems", "compute", "all"}:
        raise ValueError(f"mode must be systems, compute, or all; got {mode!r}")

    os.chdir("/root/klein4B")
    run_dir = Path(VOLUME_MOUNT) / "nsight_profiles" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    result = {"run_dir": str(run_dir), "status": "running"}
    try:
        result["environment"] = {
            "nsys_version": _run(
                ["nsys", "--version"], output=run_dir / "nsys_version.txt"
            ).stdout.strip(),
            "ncu_version": _run(
                ["ncu", "--version"], output=run_dir / "ncu_version.txt"
            ).stdout.strip(),
            "nsys_status_exit_code": _run(
                ["nsys", "status", "-e"], check=False, output=run_dir / "nsys_status.txt"
            ).returncode,
        }
        if mode in {"systems", "all"}:
            result["systems"] = _collect_systems(run_dir)
        if mode in {"compute", "all"}:
            result["compute"] = _collect_compute(run_dir)
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        (run_dir / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        (run_dir / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True))
        MODEL_VOLUME.commit()

    print("NSIGHT_PROFILE_COMPLETE " + json.dumps(result, sort_keys=True))
    return result


@APP.local_entrypoint()
def main(mode: str = "all") -> None:
    profile.remote(mode=mode)
