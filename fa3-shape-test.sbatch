#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --job-name=fa3-shape-test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
#SBATCH --output=fa3-shape-test-%j.log
#SBATCH --error=fa3-shape-test-%j.err
#SBATCH --no-requeue

set -euo pipefail

WORKDIR=/users/course_00269/lsaie-ss26-gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM

export FA3_PREFIX=/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3_install
export PYTHONPATH="$FA3_PREFIX/lib/python3.12/site-packages:$MEGATRON_LM_DIR:${PYTHONPATH:-}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))

mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

echo "START TIME: $(date)"
echo "FA3_PREFIX=$FA3_PREFIX"
echo "PYTHONPATH=$PYTHONPATH"

srun -lu \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment=alps3 \
  --cpus-per-task "$SLURM_CPUS_PER_TASK" \
  --wait 60 \
  bash -lc 'numactl --membind=0-3 python - <<'"'"'PY'"'"'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name())
print("cap:", torch.cuda.get_device_capability())

from flash_attn_3.flash_attn_interface import flash_attn_func
print("flash_attn_func imported OK")

q = torch.randn(4, 4096, 20, 80, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(4, 4096, 4, 80, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(4, 4096, 4, 80, device="cuda", dtype=torch.bfloat16, requires_grad=True)

out = flash_attn_func(q, k, v, causal=True)
out.float().sum().backward()
torch.cuda.synchronize()

print("FA3 exact-shape test OK:", out.shape)
PY'

echo "END TIME: $(date)"