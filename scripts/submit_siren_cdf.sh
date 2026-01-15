#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=siren_cdf
#SBATCH --output=siren_training/%x_seed%a-%j.out
#SBATCH --error=siren_training/%x_seed%a-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# SIREN CDF Training Script
# Usage:
#   sbatch scripts/submit_siren_cdf.sh                        # Single seed=42
#   sbatch --array=42-46 scripts/submit_siren_cdf.sh          # Multiple seeds

# Get seed from SLURM array task ID or use default
SEED=${SLURM_ARRAY_TASK_ID:-42}
RUN_NAME="seed${SEED}"

# Training parameters
NUM_STEPS=1000000
BATCH_SIZE=65536
LR=1e-4
LR_DECAY=0.99998
LR_MIN=0

echo "=========================================="
echo "SIREN CDF Training (Square Output)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "Parameters:"
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
        --plot_every 5000 \
        --checkpoint_every 50000
"

echo ""
echo "=========================================="
echo "Training finished: $(date)"
echo "Results in: siren_training/${RUN_NAME}/"
echo "=========================================="
