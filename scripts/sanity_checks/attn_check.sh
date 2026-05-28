#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:05:00
#SBATCH --job-name=check-flash-attn
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=100000
#SBATCH --no-requeue

set -euo pipefail

mkdir -p logs

srun -lu \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment=alps3 \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --wait 60 \
  bash -lc 'numactl --membind=0-3 python - <<'"'"'PY'"'"'
import sys
import torch

print("Python:", sys.executable)
print("Python version:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    print("CUDA runtime:", torch.version.cuda)

try:
    import flash_attn
    print("flash_attn module:", flash_attn)
    print("flash_attn version:", getattr(flash_attn, "__version__", "unknown"))
except Exception as e:
    print("flash_attn import failed:", repr(e))

try:
    import flash_attn_interface
    print("flash_attn_interface available:", flash_attn_interface)
except Exception as e:
    print("flash_attn_interface import failed:", repr(e))

try:
    import transformer_engine
    print("transformer_engine version:", getattr(transformer_engine, "__version__", "unknown"))
except Exception as e:
    print("transformer_engine import failed:", repr(e))
PY'