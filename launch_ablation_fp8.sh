#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput  (50 steps, no logging)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Usage: ./launch_auto.sh <mode> <model_size> [steps] [nodes] [tp] [pp] [cp] [seq_len] [attn_backend]
#
# Optional environment variables for ablations without changing positional args:
#   RUNTIME_MODE=eager|cuda_graph_te|cuda_graph_local_full   default: eager
#   PROFILE_MODE=none|nsys                                  default: none
#   TE_PRECISION_CONFIG_FILE=/path/to/te_precision.json      optional per-module TE precision config
#   MBS_OVERRIDE=<int>                                       optional micro-batch override

set -euo pipefail

MODE=${1:?Usage: ./launch_auto.sh <mode> <model_size> [steps] [nodes] [tp] [pp] [cp] [seq_len] [attn_backend]}
MODEL_SIZE=${2:?Usage: ./launch_auto.sh <mode> <model_size> [steps] [nodes] [tp] [pp] [cp] [seq_len] [attn_backend]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TP=${5:-1}
        PP=${6:-1}
        CP=${7:-1}
        SEQ_LEN=${8:-4096}
        ATTN_BACKEND=${9:-auto}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=false
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch_auto.sh train <model_size> <steps> [nodes] [tp] [pp] [cp] [seq_len] [attn_backend]}
        NODES=${4:-4}
        TP=${5:-1}
        PP=${6:-1}
        CP=${7:-1}
        SEQ_LEN=${8:-4096}
        ATTN_BACKEND=${9:-auto}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

# Forward any additional arguments after attn_backend directly to Megatron-LM.
# Example: ./launch_ablation.sh throughput 1.5b 50 4 1 1 1 4096 auto --fp8-format hybrid
EXTRA_ARGS=("${@:10}")

if (( TP * PP * CP > NODES * 4 )); then
    echo "Invalid parallelism: TP*PP*CP=$((TP * PP * CP)) > total GPUs=$((NODES * 4))"
    exit 1
fi

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

# Optional ablation controls. Keep these as environment variables so the
# normal launch interface stays unchanged.
RUNTIME_MODE=${RUNTIME_MODE:-eager}
PROFILE_MODE=${PROFILE_MODE:-none}
TE_PRECISION_CONFIG_FILE=${TE_PRECISION_CONFIG_FILE:-}
if [ -n "${MBS_OVERRIDE:-}" ]; then
    MBS=${MBS_OVERRIDE}
fi

case "$RUNTIME_MODE" in
    eager|cuda_graph_te|cuda_graph_local_full) ;;
    *)
        echo "Unknown RUNTIME_MODE=$RUNTIME_MODE. Choose: eager, cuda_graph_te, cuda_graph_local_full"
        exit 1
        ;;
esac

case "$PROFILE_MODE" in
    none|nsys) ;;
    *)
        echo "Unknown PROFILE_MODE=$PROFILE_MODE. Choose: none, nsys"
        exit 1
        ;;
esac

GBS=256
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-tp${TP}-pp${PP}-cp${CP}-s${SEQ_LEN}-${ATTN_BACKEND}-${RUNTIME_MODE}"
if [ -n "$TE_PRECISION_CONFIG_FILE" ]; then
    JOB_NAME="${JOB_NAME}-teprec"
fi
if [ "$PROFILE_MODE" != "none" ]; then
    JOB_NAME="${JOB_NAME}-${PROFILE_MODE}"
fi
################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=lsaie-ss26
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY'

echo "START TIME: $(date)"

################ Configs ################
WORKDIR=/users/course_00269/lsaie-ss26-gipfelsturm
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache
BODY

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Parallelism config
TP=${TP}
PP=${PP}
CP=${CP}
ATTN_BACKEND=${ATTN_BACKEND}
RUNTIME_MODE=${RUNTIME_MODE}
PROFILE_MODE=${PROFILE_MODE}
TE_PRECISION_CONFIG_FILE=${TE_PRECISION_CONFIG_FILE}
EXTRA_ARGS=(${EXTRA_ARGS[@]@Q})

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME=${JOB_NAME}
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

ATTENTION_ARGS=()
if [ "$ATTN_BACKEND" != "auto" ]; then
    ATTENTION_ARGS+=(--attention-backend "$ATTN_BACKEND")
fi

RUNTIME_ARGS=()
case "$RUNTIME_MODE" in
    eager)
        ;;
    cuda_graph_te)
        # Transformer Engine CUDA graphs capture layer-level regions.
        RUNTIME_ARGS+=(--cuda-graph-impl transformer_engine)
        ;;
    cuda_graph_local_full)
        # Full training iteration capture. Requires --no-check-for-nan-in-loss-and-grad,
        # which is already set in TRAINING_ARGS below.
        RUNTIME_ARGS+=(--cuda-graph-impl local --cuda-graph-scope full_iteration)
        ;;
esac

TE_PRECISION_ARGS=()
if [ -n "$TE_PRECISION_CONFIG_FILE" ]; then
    TE_PRECISION_ARGS+=(--te-precision-config-file "$TE_PRECISION_CONFIG_FILE")
fi

MEGATRON_EXTRA_ARGS=("${EXTRA_ARGS[@]}")

PROFILE_PREFIX=()
if [ "$PROFILE_MODE" = "nsys" ]; then
    PROFILE_DIR="$LOG_DIR/nsys"
    mkdir -p "$PROFILE_DIR"
    PROFILE_PREFIX=(nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt,cublas,cudnn,nccl --output "$PROFILE_DIR/${EXP_NAME}-%q{SLURM_PROCID}")
fi

SETUP

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --context-parallel-size $CP
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${ATTENTION_ARGS[@]} \
    ${RUNTIME_ARGS[@]} \
    ${TE_PRECISION_ARGS[@]} \
    ${MEGATRON_EXTRA_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: ${PROFILE_PREFIX[*]} $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 ${PROFILE_PREFIX[*]} $TRAINING_CMD"

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
