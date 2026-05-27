#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev

ITER_RE = re.compile(
    r"iteration\s+(?P<iter>\d+)/\s*(?P<total>\d+).*?"
    r"elapsed time per iteration \(ms\):\s*(?P<iter_ms>[0-9.]+).*?"
    r"throughput per GPU \(TFLOP/s/GPU\):\s*(?P<tflops>[0-9.]+).*?"
    r"tokens/sec/GPU:\s*(?P<tok_s_gpu>[0-9.]+).*?"
    r"global batch size:\s*(?P<gbs>\d+).*?"
    r"lm loss:\s*(?P<loss>[0-9.Ee+-]+).*?"
    r"number of skipped iterations:\s*(?P<skipped>\d+).*?"
    r"number of nan iterations:\s*(?P<nan>\d+)"
)

CMD_RE = re.compile(r"CMD:\s*(?P<cmd>torchrun .*)")

def get_flag(cmd: str, flag: str, default=""):
    # Handles: --flag value
    m = re.search(rf"{re.escape(flag)}\s+([^\s]+)", cmd)
    return m.group(1) if m else default

def has_flag(cmd: str, flag: str) -> bool:
    return flag in cmd.split()

def parse_log(path: Path, warmup: int):
    text = path.read_text(errors="ignore")

    cmd_match = CMD_RE.search(text)
    cmd = cmd_match.group("cmd") if cmd_match else ""

    rows = []
    for m in ITER_RE.finditer(text):
        d = m.groupdict()
        row = {
            "iter": int(d["iter"]),
            "total_iters": int(d["total"]),
            "iter_ms": float(d["iter_ms"]),
            "tflops_gpu": float(d["tflops"]),
            "tok_s_gpu": float(d["tok_s_gpu"]),
            "gbs": int(d["gbs"]),
            "loss": float(d["loss"]),
            "skipped": int(d["skipped"]),
            "nan": int(d["nan"]),
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"No iteration lines found in {path}")

    measured = [r for r in rows if r["iter"] > warmup]
    if not measured:
        measured = rows

    # Config from command line
    nodes_match = re.search(r"--nnodes\s+([0-9]+)", cmd)
    nproc_match = re.search(r"--nproc-per-node\s+([0-9]+)", cmd)

    nodes = int(nodes_match.group(1)) if nodes_match else ""
    gpus_per_node = int(nproc_match.group(1)) if nproc_match else ""
    world_size = nodes * gpus_per_node if nodes and gpus_per_node else ""

    tp = int(get_flag(cmd, "--tensor-model-parallel-size", "1"))
    pp = int(get_flag(cmd, "--pipeline-model-parallel-size", "1"))
    cp = int(get_flag(cmd, "--context-parallel-size", "1"))

    dp = ""
    if world_size:
        denom = tp * pp * cp
        dp = world_size // denom if denom > 0 else ""

    out = {
        "log_file": str(path),
        "job_name": path.stem,
        "nodes": nodes,
        "gpus_per_node": gpus_per_node,
        "world_size": world_size,
        "tp": tp,
        "pp": pp,
        "cp": cp,
        "dp": dp,
        "seq_len": get_flag(cmd, "--seq-length"),
        "max_pos": get_flag(cmd, "--max-position-embeddings"),
        "micro_batch_size": get_flag(cmd, "--micro-batch-size"),
        "global_batch_size": measured[-1]["gbs"],
        "train_iters": get_flag(cmd, "--train-iters"),
        "num_layers": get_flag(cmd, "--num-layers"),
        "hidden_size": get_flag(cmd, "--hidden-size"),
        "ffn_hidden_size": get_flag(cmd, "--ffn-hidden-size"),
        "num_attention_heads": get_flag(cmd, "--num-attention-heads"),
        "num_query_groups": get_flag(cmd, "--num-query-groups"),
        "attention_backend": get_flag(cmd, "--attention-backend", "auto"),
        "transformer_impl": get_flag(cmd, "--transformer-impl"),
        "bf16": has_flag(cmd, "--bf16"),
        "use_distributed_optimizer": has_flag(cmd, "--use-distributed-optimizer"),
        "warmup_excluded": warmup,
        "num_measured_iters": len(measured),
        "mean_tok_s_gpu": round(mean(r["tok_s_gpu"] for r in measured), 3),
        "std_tok_s_gpu": round(stdev(r["tok_s_gpu"] for r in measured), 3) if len(measured) > 1 else 0.0,
        "mean_tflops_gpu": round(mean(r["tflops_gpu"] for r in measured), 3),
        "mean_iter_ms": round(mean(r["iter_ms"] for r in measured), 3),
        "min_iter_ms": round(min(r["iter_ms"] for r in measured), 3),
        "max_iter_ms": round(max(r["iter_ms"] for r in measured), 3),
        "first_loss": rows[0]["loss"],
        "final_loss": rows[-1]["loss"],
        "total_skipped": sum(r["skipped"] for r in rows),
        "total_nan": sum(r["nan"] for r in rows),
    }
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    row = parse_log(args.log, args.warmup)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.out.exists()
        with args.out.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()


python3 scripts/parse_megatron_log.py \
  logs/gipfel-throughput-1.5b-50s-4n-tp1-pp1-cp1-s4096-flash-2319603.log \
  --warmup 10 \
  --out results/runs.csv