"""
launch_generate_coinrun_data_multi.py

Spawn multiple worker processes for generate_coinrun_data.py on a single machine.

This launcher is intentionally lightweight:
- It passes all unknown args through to generate_coinrun_data.py
- It sets CUDA_VISIBLE_DEVICES to pin workers to a single GPU (default: "0")
- It prefixes each worker's stdout/stderr lines with the worker id

Example:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python launch_generate_coinrun_data_multi.py --num-workers 64 --cuda-visible-devices 7  --  --wandb-run-path sgoodfriend/rl-algo-impls-benchmarks/vmjd3amn   --output-dir /ephemeral/datasets/coinrun_hard_agent_episodes --n-envs=32
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from typing import List
import time

def _stream_lines(prefix: str, pipe) -> None:
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            sys.stdout.write(f"{prefix}{line}")
            sys.stdout.flush()
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        help="Do not prefix worker output lines with [wXX].",
    )
    parser.add_argument(
        "passthrough",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to generate_coinrun_data.py (use `--` before them).",
    )
    args = parser.parse_args()

    num_workers = int(args.num_workers)
    if num_workers <= 0:
        raise ValueError(f"--num-workers must be > 0, got {num_workers}")

    # Strip a leading '--' if present (common convention with REMAINDER).
    passthrough: List[str] = list(args.passthrough)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    script_path = os.path.join(os.path.dirname(__file__), "generate_coinrun_data.py")

    procs: List[subprocess.Popen] = []
    threads: List[threading.Thread] = []
    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    base_env.setdefault("PYTHONUNBUFFERED", "1")

    for worker_id in range(num_workers):
        print(f"Starting worker {worker_id}")
        cmd = [
            sys.executable,
            script_path,
            "--num-workers",
            str(num_workers),
            "--worker-id",
            str(worker_id),
            *passthrough,
        ]
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=base_env,
        )
        procs.append(p)
        if p.stdout is not None:
            prefix = "" if args.no_prefix else f"[w{worker_id:02d}] "
            t = threading.Thread(target=_stream_lines, args=(prefix, p.stdout), daemon=True)
            t.start()
            threads.append(t)
        # give each worker time to warm up
        time.sleep(5)

    exit_codes: List[int] = []
    try:
        for p in procs:
            exit_codes.append(p.wait())
    except KeyboardInterrupt:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        raise

    worst = max(exit_codes) if exit_codes else 0
    if worst != 0:
        raise SystemExit(worst)


if __name__ == "__main__":
    main()
