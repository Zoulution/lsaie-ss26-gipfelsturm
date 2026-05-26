#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:30:00
#SBATCH --job-name=install-fa3
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=240000
#SBATCH --no-requeue

set -euo pipefail
mkdir -p logs

INSTALL_PREFIX=/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3_install
BUILD_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3_build

mkdir -p "$INSTALL_PREFIX" "$BUILD_DIR"

srun -lu \
  --mpi=pmix \
  --network=disable_rdzv_get \
  --environment=alps3 \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --wait 60 \
  bash -lc "
set -euo pipefail

export INSTALL_PREFIX=$INSTALL_PREFIX
export BUILD_DIR=$BUILD_DIR

echo 'Python:' \$(which python)
python - <<'PY'
import sys, torch
print('Python version:', sys.version)
print('Torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
    print('CUDA runtime:', torch.version.cuda)
PY

python -m pip install --user ninja || true

cd \$BUILD_DIR
if [ ! -d flash-attention ]; then
    git clone https://github.com/Dao-AILab/flash-attention.git
fi

cd flash-attention
git fetch --all
git submodule update --init --recursive

cd hopper

export MAX_JOBS=4
export PYTHONUSERBASE=\$INSTALL_PREFIX
export PYTHONPATH=\$INSTALL_PREFIX/lib/python3.12/site-packages:\${PYTHONPATH:-}
python setup.py install --user

echo 'Preparing Megatron-style flash_attn_3 package...'
export SITE=\$INSTALL_PREFIX/lib/python3.12/site-packages
mkdir -p \$SITE/flash_attn_3

if [ -f \$BUILD_DIR/flash-attention/hopper/flash_attn_interface.py ]; then
    cp \$BUILD_DIR/flash-attention/hopper/flash_attn_interface.py \$SITE/flash_attn_3/flash_attn_interface.py
fi

touch \$SITE/flash_attn_3/__init__.py

echo 'Testing import...'
export PYTHONPATH=\$SITE:\${PYTHONPATH:-}

python - <<'PY'
import flash_attn_3.flash_attn_interface as fai
print('flash_attn_3.flash_attn_interface imported:', fai)
print('has flash_attn_func:', hasattr(fai, 'flash_attn_func'))
print('has flash_attn_varlen_func:', hasattr(fai, 'flash_attn_varlen_func'))
PY
"