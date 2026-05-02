#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("experiments/results/runs.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/results/plots"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Useful numeric conversions
    for col in ["tp", "pp", "cp", "nodes", "seq_len", "mean_tok_s_gpu", "mean_iter_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Plot 1: throughput vs TP for each node count
    if {"tp", "nodes", "mean_tok_s_gpu"}.issubset(df.columns):
        plt.figure()
        for nodes, g in df.groupby("nodes"):
            g = g.sort_values("tp")
            plt.plot(g["tp"], g["mean_tok_s_gpu"], marker="o", label=f"{int(nodes)} nodes")
        plt.xlabel("Tensor parallel size")
        plt.ylabel("Mean tokens/sec/GPU")
        plt.title("Throughput vs tensor parallelism")
        plt.legend()
        plt.grid(True)
        plt.savefig(args.out_dir / "throughput_vs_tp.png", bbox_inches="tight", dpi=200)

    # Plot 2: throughput vs sequence length
    if {"seq_len", "mean_tok_s_gpu"}.issubset(df.columns):
        g = df.sort_values("seq_len")
        plt.figure()
        plt.plot(g["seq_len"], g["mean_tok_s_gpu"], marker="o")
        plt.xlabel("Sequence length")
        plt.ylabel("Mean tokens/sec/GPU")
        plt.title("Throughput vs sequence length")
        plt.grid(True)
        plt.savefig(args.out_dir / "throughput_vs_seq_len.png", bbox_inches="tight", dpi=200)

if __name__ == "__main__":
    main()