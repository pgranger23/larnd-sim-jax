#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=siren_shallow
#SBATCH --output=siren_training/%x_seed%a-%j.out
#SBATCH --error=siren_training/%x_seed%a-%j.err
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=04:00:00

# Shallow SIREN Training - fast inference test
# 32 features × 1 hidden layer = ~2K params (~150× faster than default)

SEED=${SLURM_ARRAY_TASK_ID:-42}
RUN_NAME="shallow_32x1_seed${SEED}"

# Architecture
HIDDEN_FEATURES=32
HIDDEN_LAYERS=1

# Training parameters
NUM_STEPS=500000
BATCH_SIZE=65536
LR=1e-4
LR_DECAY=0.99998
LR_MIN=1e-6

echo "=========================================="
echo "SIREN Shallow Training (${HIDDEN_FEATURES}×${HIDDEN_LAYERS})"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "Architecture:"
echo "  hidden_features: ${HIDDEN_FEATURES}"
echo "  hidden_layers: ${HIDDEN_LAYERS}"
echo ""
echo "Training:"
echo "  seed: ${SEED}"
echo "  run_name: ${RUN_NAME}"
echo "  num_steps: ${NUM_STEPS}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  learning_rate: ${LR}"
echo "  lr_decay_rate: ${LR_DECAY}"
echo "  lr_min: ${LR_MIN}"
echo ""

cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax
mkdir -p siren_training

CONTAINER="/sdf/group/neutrino/pgranger/larnd-sim-jax.sif"

singularity exec --nv -B /sdf ${CONTAINER} /bin/bash -c "
    export PYTHONNOUSERSITE=1
    pip3 install --quiet --user .
    python3 -m src.siren.train_surrogate \
        --hidden_features ${HIDDEN_FEATURES} \
        --hidden_layers ${HIDDEN_LAYERS} \
        --use_cdf \
        --square_output \
        --run_name ${RUN_NAME} \
        --seed ${SEED} \
        --batch_size ${BATCH_SIZE} \
        --num_steps ${NUM_STEPS} \
        --learning_rate ${LR} \
        --lr_decay_rate ${LR_DECAY} \
        --lr_min ${LR_MIN} \
        --log_every 100 \
        --plot_every 10000 \
        --checkpoint_every 50000
"

echo ""
echo "=========================================="
echo "Training finished: $(date)"
echo "Results in: siren_training/${RUN_NAME}/"
echo "=========================================="
