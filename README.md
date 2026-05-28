# Megatron-LM Exploration: Throughput and Training Loss Analysis

This repository contains our project for the **Large-Scale AI Engineering** course at ETH Zurich. The project explores training throughput and training-loss behavior in **Megatron-LM** on a multi-node GPU cluster.

The main goal is to understand how different distributed training configurations and runtime optimizations affect language-model pre-training efficiency. Instead of focusing only on final model quality, we primarily evaluate system-level performance using:

* tokens/sec/GPU
* training loss

## Project Overview

The experiments are organized around two stages.

First, we establish baseline performance by varying distributed training and workload parameters, including:

* number of nodes
* tensor parallelism
* pipeline parallelism
* context parallelism
* model size
* sequence length

Second, we evaluate optimization techniques on top of the baseline setup, including:

* Mixture-of-Experts (MoE)
* BF16 vs FP8 mixed precision
* FlashAttention-2 and FlashAttention-3
* CUDA Graphs
* training-log monitoring and parsing utilities

## Repository Structure

```text
.
├── data/                         # Tokenizer files and small data assets
├── launch_scripts/               # Slurm launch scripts for different experiment groups
│   ├── install_fa3.sh             # Install/build FlashAttention-3 in the EDF container environment
│   ├── launch.sh                  # Main/default Megatron-LM launch script
│   ├── launch_ablation.sh         # Launch script for optimization/ablation experiments
│   ├── launch_parallelism.sh      # Launch script for parallelism sweeps
│   └── launch_fa3.sh              # Launch script using the patched FA3 Megatron-LM copy
├── logs/                         # Slurm and Megatron-LM log files
├── Megatron-LM/                  # Original Megatron-LM source tree / submodule
├── patches/                      # Patch files used to modify Megatron-LM behavior
├── results/                      # Parsed experiment outputs and result summaries
├── scripts/                      # Utility scripts for checking, parsing, plotting, and monitoring
│   ├── sanity_checks/             # Small standalone checks for environment and FA3 correctness
│   │   ├── attn_check.sh           # Check GPU, FlashAttention, and Transformer Engine availability
│   │   ├── config.sh               # Shared configuration for sanity checks
│   │   ├── fa3_backward_test.sbatch # Standalone FA3 forward/backward test
│   │   └── fa3-shape-test.sbatch   # FA3 shape compatibility test
│   ├── parse_megatron_log.py      # Parse Megatron-LM logs into structured metrics
│   ├── quick_plot.py              # Quick plotting utility for parsed results
│   ├── summarize_logs.py          # Summarize groups of Megatron-LM logs
│   └── training_guard.py          # Monitor running logs and detect failures/stalls
├── .env                          # Local environment variables, not intended for sharing
├── .gitignore                    # Git ignore rules
├── .gitmodules                   # Submodule configuration
├── alps3.toml                    # EDF/CSCS container environment configuration
├── README.md                     # Project documentation
├── test_infra.py                 # Infrastructure test helper
└── test-infra.sbatch             # Slurm script for infrastructure testing

## Launch Script

The main launch script makes repeated Megatron-LM experiments easier to configure, name, submit, and compare. Each job name encodes the key configuration:

```text
mode, model size, steps, nodes, TP, PP, CP, sequence length, attention backend
```

Example:

```bash
./launch_auto.sh throughput 1.5b 50 4 1 1 1 4096 auto
```

Arguments:

```text
./launch_auto.sh <mode> <model_size> [steps] [nodes] [tp] [pp] [cp] [seq_len] [attn_backend]
```

Example modes:

```text
throughput
train
```

Example model sizes:

```text
125m, 350m, 760m, 1.5b, 3b, 4.5b, 8b
```

The script passes the parallelism configuration to Megatron-LM using:

```bash
--tensor-model-parallel-size $TP
--pipeline-model-parallel-size $PP
--context-parallel-size $CP
```

It also keeps distributed optimizer and communication overlap enabled:

```bash
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
```

## Parallelism Experiments

We study the effect of data parallelism, tensor parallelism, model scale, and sequence length.

For a fixed number of nodes, the total world size is:

```text
world_size = nodes × 4
```

because each node provides four GPUs.

The data-parallel degree is determined by the remaining GPUs after model parallelism:

```text
DP = world_size / (TP × PP × CP)
```

In the baseline sweep, we mainly use:

```text
TP = 1
PP = 1
CP = 1
```

and vary node count, model size, and sequence length.

## Mixture-of-Experts Experiments

We evaluate MoE as a sparse-capacity scaling strategy. In a dense Transformer, every token passes through the same FFN in every layer. In an MoE Transformer, the dense FFN is replaced by multiple expert FFNs, and each token is routed to only a subset of experts.

The main MoE configuration uses four experts with top-1 routing:

```bash
--num-experts 4
--expert-model-parallel-size 4
--moe-router-topk 1
--moe-router-pre-softmax
--moe-aux-loss-coeff 1e-2
--moe-token-dispatcher-type alltoall
--moe-permute-fusion
--moe-router-fusion
--moe-grouped-gemm
```

We compare:

1. A dense model with approximately 4.5B active parameters.
2. A 4-expert MoE model with approximately 4.5B total parameters but only about 1.5B active parameters per token.

This allows us to evaluate whether MoE improves the throughput-capacity tradeoff and whether it can improve training loss under a similar active-compute budget.

## Mixed Precision

We evaluate BF16 and FP8 mixed precision using Transformer Engine.

The BF16 baseline uses:

```bash
--bf16
--transformer-impl transformer_engine
--use-precision-aware-optimizer
--main-grads-dtype bf16
```

The FP8 experiment enables Transformer Engine FP8 hybrid precision:

```bash
--fp8-format hybrid
```

In our 1.5B model experiments, FP8 improved throughput compared with the BF16 baseline while remaining numerically stable over the measured run.

## FlashAttention

Attention is implemented through Transformer Engine in Megatron-LM.

The default configuration uses:

```bash
--attention-backend auto
```

which allows Transformer Engine to choose the backend automatically.

When forcing:

```bash
--attention-backend flash
```

the EDF container uses the installed FlashAttention package:

```text
flash_attn==2.7.4.post1
```

which corresponds to FlashAttention-2.

To test FlashAttention-3, we installed the Hopper-specific FA3 package and exposed:

```python
from flash_attn_interface import flash_attn_func
```

Megatron-LM does not automatically call this standalone FA3 interface, so we patched:

```text
megatron/core/extensions/transformer_engine.py
```

specifically the `TEDotProductAttention` wrapper, to directly call `flash_attn_func`.

The direct FA3 path is enabled with:

```bash
export MEGATRON_USE_FA3_DIRECT=1
```

For debugging:

```bash
export MEGATRON_FA3_DEBUG=1
```

The patch converts Megatron/Transformer Engine's attention layout from:

```text
[S, B, H, D]
```

to FA3's expected layout:

```text
[B, S, H, D]
```

After the FA3 call, the output is converted back and flattened to:

```text
[S, B, H * D]
```

which is required by the following projection layer.

## Runtime Execution Optimizations

The original plan was to compare:

1. standard eager execution
2. `torch.compile`
3. CUDA Graphs

Each mode was intended to be benchmarked on the same workload, reporting both throughput and step-time variance.

In practice, we performed preliminary CUDA Graph experiments. The observed throughput gain was limited, suggesting that this workload was already dominated by large fused GPU kernels, communication, and optimizer overheads rather than Python dispatch overhead. Since CUDA Graphs usually provide a stronger launch-overhead reduction than `torch.compile` for static training loops, we deprioritized `torch.compile` and focused on attention backends, mixed precision, and MoE scaling.

## Training Guard

The `training_guard.py` utility monitors Megatron-LM logs while a job is running. It can detect:

* NaN or Inf losses
* CUDA out-of-memory errors
* invalid arguments
* checkpoint loading errors
* NCCL failures
* CUDA Graph errors
* stalled jobs with no log updates
* throughput below a configurable threshold

By default, the guard runs in dry-run mode and only prints what it would do. With `--act`, it can cancel the Slurm job using `scancel`.

## Log Parser

The `parse_megatron_log.py` utility converts raw Megatron-LM logs into structured CSV rows.

It extracts:

* model size
* number of nodes
* GPUs per node
* world size
* TP, PP, CP, and DP
* sequence length
* attention backend
* iteration time
* tokens/sec/GPU
* TFLOP/s/GPU
* enabled ablations

This makes it easier to compare many runs consistently.


