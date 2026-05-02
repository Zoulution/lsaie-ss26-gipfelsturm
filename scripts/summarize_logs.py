#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--out", type=Path, default=Path("experiments/results/runs.csv"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--parser", type=Path, default=Path("experiments/scripts/parse_megatron_log.py"))
    args = parser.parse_args()

    if args.out.exists():
        args.out.unlink()

    for log in sorted(args.log_dir.glob("*.log")):
        try:
            subprocess.run(
                ["python", str(args.parser), str(log), "--warmup", str(args.warmup), "--out", str(args.out)],
                check=True,
            )
            print(f"parsed {log}")
        except subprocess.CalledProcessError:
            print(f"failed {log}")

if __name__ == "__main__":
    main()

# python experiments/scripts/summarize_logs.py --log-dir logs --warmup 10