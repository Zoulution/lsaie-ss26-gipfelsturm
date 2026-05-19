#!/usr/bin/env python3
"""
training_guard.py

Standalone external monitor for Slurm/Megatron training jobs.

It does NOT patch Megatron-LM.
It watches a Slurm/Megatron log while the job is running and can make simple
dynamic decisions such as:
  - stop the job if NaN / Inf / OOM / checkpoint errors appear
  - stop the job if throughput stays below a threshold after warmup
  - stop the job if no new log output appears for too long
  - optionally submit a follow-up command after stopping

Default mode is DRY RUN: it prints what it would do, but does not call scancel
or run the follow-up command unless --act is passed.

Examples:
  # Watch only, never cancel:
  python scripts/training_guard.py --log logs/job.log --job-id 123456

  # Actually cancel if NaN/OOM/checkpoint error appears:
  python scripts/training_guard.py --log logs/job.log --job-id 123456 --act

  # Cancel if recent throughput is below 30000 tok/s/GPU after 20 iterations:
  python scripts/training_guard.py --log logs/job.log --job-id 123456 \
      --min-tokens-per-gpu 30000 --warmup-iters 20 --act

  # Cancel if log is stale for 20 minutes:
  python scripts/training_guard.py --log logs/job.log --job-id 123456 \
      --max-stale-minutes 20 --act

  # Submit another command after cancellation:
  python scripts/training_guard.py --log logs/job.log --job-id 123456 \
      --act --on-cancel-cmd "AUTO_RESUBMIT=true bash launch_resumable.sh train 1.5b 10000 4 1 1 1 4096 auto"

Exit codes:
  0 = monitor exited normally
  1 = monitor took an action or detected a fatal condition
  2 = usage / file error
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ITER_PATTERNS = [
    re.compile(r"\biteration\s+([0-9]+)\b", re.IGNORECASE),
    re.compile(r"\biter(?:ation)?[:=\s]+([0-9]+)\b", re.IGNORECASE),
    re.compile(r"\bglobal\s+step[:=\s]+([0-9]+)\b", re.IGNORECASE),
]

# Try to catch common throughput formats:
#   tokens/sec/gpu: 34567
#   tokens per gpu: 34567
#   throughput ... 34567
#   tok/s/GPU = 34567
TOKENS_PER_GPU_PATTERNS = [
    re.compile(r"(?:tokens|tok)[/_\s-]*(?:per|/)?[_\s-]*(?:sec|s)[/_\s-]*(?:per|/)?[_\s-]*gpu[:=\s]+([0-9.]+)", re.IGNORECASE),
    re.compile(r"(?:tokens|tok)[/_\s-]*s[/_\s-]*gpu[:=\s]+([0-9.]+)", re.IGNORECASE),
    re.compile(r"throughput.*?([0-9.]+)\s*(?:tokens|tok).*(?:gpu)", re.IGNORECASE),
]

LOSS_PATTERNS = [
    re.compile(r"\blm loss[:=\s]+([0-9.eE+-]+)\b", re.IGNORECASE),
    re.compile(r"\bloss[:=\s]+([0-9.eE+-]+)\b", re.IGNORECASE),
]

FATAL_PATTERNS = {
    "OOM": [
        re.compile(r"out of memory", re.IGNORECASE),
        re.compile(r"\bOOM\b", re.IGNORECASE),
        re.compile(r"CUDA error:.*memory", re.IGNORECASE),
    ],
    "BAD_ARGS": [
        re.compile(r"unrecognized arguments", re.IGNORECASE),
        re.compile(r"error: argument", re.IGNORECASE),
    ],
    "CHECKPOINT_ERROR": [
        re.compile(r"checkpoint.*(error|failed|exception|corrupt|missing)", re.IGNORECASE),
        re.compile(r"(error|failed|exception|corrupt|missing).*checkpoint", re.IGNORECASE),
        re.compile(r"could not.*load", re.IGNORECASE),
    ],
    "NCCL_ERROR": [
        re.compile(r"\bNCCL\b.*(error|failed|timeout|unhandled)", re.IGNORECASE),
        re.compile(r"(error|failed|timeout|unhandled).*\bNCCL\b", re.IGNORECASE),
    ],
    "CUDA_GRAPH_ERROR": [
        re.compile(r"cuda graph.*(error|failed|capture|illegal)", re.IGNORECASE),
        re.compile(r"(error|failed|capture|illegal).*cuda graph", re.IGNORECASE),
    ],
    "NAN_OR_INF": [
        re.compile(r"\bNaN\b", re.IGNORECASE),
        re.compile(r"non[-\s]?finite", re.IGNORECASE),
        re.compile(r"loss.*\binf\b", re.IGNORECASE),
    ],
}


@dataclass
class MonitorState:
    latest_iter: Optional[int] = None
    latest_loss: Optional[float] = None
    latest_tokens_per_gpu: Optional[float] = None
    last_log_update_time: float = 0.0
    recent_throughputs: collections.deque = None

    def __post_init__(self) -> None:
        if self.recent_throughputs is None:
            self.recent_throughputs = collections.deque(maxlen=5)


def match_int(patterns: list[re.Pattern[str]], line: str) -> Optional[int]:
    for pat in patterns:
        m = pat.search(line)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None


def match_float(patterns: list[re.Pattern[str]], line: str) -> Optional[float]:
    for pat in patterns:
        m = pat.search(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def match_fatal(line: str) -> Optional[str]:
    for name, patterns in FATAL_PATTERNS.items():
        for pat in patterns:
            if pat.search(line):
                return name
    return None


def run_cmd(cmd: str, act: bool) -> int:
    print(f"[training_guard] command: {cmd}")
    if not act:
        print("[training_guard] dry-run mode: command not executed. Pass --act to execute.")
        return 0
    return subprocess.call(cmd, shell=True)


def cancel_job(job_id: str, reason: str, act: bool) -> None:
    print(f"[training_guard] DECISION: cancel job {job_id}. Reason: {reason}")
    run_cmd(f"scancel {shlex.quote(job_id)}", act=act)


def follow_up(cmd: Optional[str], act: bool) -> None:
    if not cmd:
        return
    print("[training_guard] follow-up command requested.")
    run_cmd(cmd, act=act)


def seek_to_end(path: Path) -> int:
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        return f.tell()


def read_new_lines(path: Path, offset: int) -> tuple[int, list[str]]:
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
        new_offset = f.tell()
    text = data.decode("utf-8", errors="replace")
    return new_offset, text.splitlines()


def process_lines(lines: list[str], state: MonitorState) -> Optional[str]:
    """Return fatal reason if any."""
    for line in lines:
        fatal = match_fatal(line)
        if fatal:
            return fatal

        it = match_int(ITER_PATTERNS, line)
        if it is not None:
            state.latest_iter = it

        loss = match_float(LOSS_PATTERNS, line)
        if loss is not None:
            state.latest_loss = loss

        tpg = match_float(TOKENS_PER_GPU_PATTERNS, line)
        if tpg is not None:
            state.latest_tokens_per_gpu = tpg
            state.recent_throughputs.append(tpg)

    return None


def should_stop_for_low_throughput(
    state: MonitorState,
    min_tokens_per_gpu: Optional[float],
    warmup_iters: int,
    window: int,
) -> Optional[str]:
    if min_tokens_per_gpu is None:
        return None
    if state.latest_iter is None or state.latest_iter < warmup_iters:
        return None
    if len(state.recent_throughputs) < min(window, state.recent_throughputs.maxlen):
        return None

    recent = list(state.recent_throughputs)[-window:]
    avg = sum(recent) / len(recent)
    if avg < min_tokens_per_gpu:
        return (
            f"recent average throughput {avg:.2f} tok/s/GPU "
            f"< threshold {min_tokens_per_gpu:.2f}"
        )
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path, help="Path to Slurm/Megatron log file.")
    parser.add_argument("--job-id", required=True, help="Slurm job id to cancel if needed.")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval.")
    parser.add_argument("--act", action="store_true", help="Actually execute scancel/follow-up commands.")
    parser.add_argument("--from-start", action="store_true", help="Read existing log content from beginning.")
    parser.add_argument("--max-stale-minutes", type=float, default=None, help="Cancel if log has no new output for this many minutes.")
    parser.add_argument("--min-tokens-per-gpu", type=float, default=None, help="Cancel if recent tok/s/GPU stays below this threshold.")
    parser.add_argument("--warmup-iters", type=int, default=20, help="Do not enforce throughput threshold before this iteration.")
    parser.add_argument("--throughput-window", type=int, default=3, help="Number of recent throughput entries to average.")
    parser.add_argument("--max-runtime-minutes", type=float, default=None, help="Stop monitoring after this many minutes.")
    parser.add_argument("--on-cancel-cmd", default=None, help="Optional shell command to run after cancellation.")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"[training_guard] log file does not exist: {args.log}", file=sys.stderr)
        return 2

    if args.poll_seconds <= 0:
        print("[training_guard] --poll-seconds must be positive.", file=sys.stderr)
        return 2

    state = MonitorState(last_log_update_time=time.time())
    offset = 0 if args.from_start else seek_to_end(args.log)
    start_time = time.time()

    print("[training_guard] started")
    print(f"[training_guard] log: {args.log}")
    print(f"[training_guard] job id: {args.job_id}")
    print(f"[training_guard] mode: {'ACT' if args.act else 'DRY RUN'}")
    print("[training_guard] waiting for new log lines...")

    while True:
        now = time.time()

        if args.max_runtime_minutes is not None:
            if (now - start_time) > args.max_runtime_minutes * 60:
                print("[training_guard] max monitor runtime reached; exiting.")
                return 0

        try:
            new_offset, lines = read_new_lines(args.log, offset)
        except OSError as e:
            print(f"[training_guard] could not read log: {e}", file=sys.stderr)
            return 2

        if lines:
            offset = new_offset
            state.last_log_update_time = now
            fatal_reason = process_lines(lines, state)

            if state.latest_iter is not None:
                msg = f"[training_guard] iter={state.latest_iter}"
                if state.latest_loss is not None:
                    msg += f" loss={state.latest_loss}"
                if state.latest_tokens_per_gpu is not None:
                    msg += f" tok/s/GPU={state.latest_tokens_per_gpu}"
                print(msg)

            if fatal_reason is not None:
                cancel_job(args.job_id, fatal_reason, act=args.act)
                follow_up(args.on_cancel_cmd, act=args.act)
                return 1

            low_tp_reason = should_stop_for_low_throughput(
                state,
                min_tokens_per_gpu=args.min_tokens_per_gpu,
                warmup_iters=args.warmup_iters,
                window=args.throughput_window,
            )
            if low_tp_reason is not None:
                cancel_job(args.job_id, low_tp_reason, act=args.act)
                follow_up(args.on_cancel_cmd, act=args.act)
                return 1

        if args.max_stale_minutes is not None:
            stale_seconds = now - state.last_log_update_time
            if stale_seconds > args.max_stale_minutes * 60:
                reason = f"log stale for {stale_seconds / 60:.1f} minutes"
                cancel_job(args.job_id, reason, act=args.act)
                follow_up(args.on_cancel_cmd, act=args.act)
                return 1

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())


# Watch a running job without taking action:
# python scripts/training_guard.py \
#   --log logs/gipfel-train-1.5b-123456.log \
#   --job-id 123456

# cancel if fatal errors appear:
# python scripts/training_guard.py \
#   --log logs/gipfel-train-1.5b-123456.log \
#   --job-id 123456 \
#   --act

# Cancel if the log has no new output for 20 minutes:
# python scripts/training_guard.py \
#   --log logs/gipfel-train-1.5b-123456.log \
#   --job-id 123456 \
#   --max-stale-minutes 20 \
#   --act

# Cancel if throughput is bad after warmup:
# python scripts/training_guard.py \
#   --log logs/gipfel-train-1.5b-123456.log \
#   --job-id 123456 \
#   --min-tokens-per-gpu 30000 \
#   --warmup-iters 20 \
#   --act




