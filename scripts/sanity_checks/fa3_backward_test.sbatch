#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:10:00
#SBATCH --job-name=fa3-bwd-test
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120000
#SBATCH --no-requeue

set -euo pipefail
mkdir -p logs

FA3_PREFIX=/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3_install

srun -lu \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment=alps3 \
  --cpus-per-task "$SLURM_CPUS_PER_TASK" \
  --wait 60 \
  bash -lc "
set -euo pipefail

export FA3_PREFIX=$FA3_PREFIX
export PYTHONPATH=\$FA3_PREFIX/lib/python3.12/site-packages:\${PYTHONPATH:-}

echo \"FA3_PREFIX=\$FA3_PREFIX\"
echo \"PYTHONPATH=\$PYTHONPATH\"
which python

python - <<'PY'
import torch

try:
    from flash_attn_interface import flash_attn_func
    print('Imported from flash_attn_interface')
except Exception as e:
    print('flash_attn_interface import failed:', repr(e))
    from flash_attn_3.flash_attn_interface import flash_attn_func
    print('Imported from flash_attn_3.flash_attn_interface')

print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu:', torch.cuda.get_device_name(0))
print('cap:', torch.cuda.get_device_capability(0))

B, S, Hq, Hkv, D = 4, 4096, 20, 4, 80

q = torch.randn(B, S, Hq, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, S, Hkv, D, device='cuda', dtype=torch.bfloat16, requires_grad=True)

out = flash_attn_func(q, k, v, causal=True)
print('out:', out.shape)

loss = out.float().square().mean()
loss.backward()
torch.cuda.synchronize()

print('FA3 forward+backward OK')
print('q.grad:', q.grad.shape)
print('k.grad:', k.grad.shape)
print('v.grad:', v.grad.shape)
PY
"