#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=siren_arch
#SBATCH --output=siren_training/architecture_test/%x_%a-%j.out
#SBATCH --error=siren_training/architecture_test/%x_%a-%j.err
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=06:00:00

# Architecture comparison test
# Array job: each task tests a different architecture
# Usage: sbatch --array=0-3 scripts/submit_siren_shallow_test.sh

SEED=46
OUTPUT_BASE="siren_training/architecture_test"

# Define architectures: "features:layers"
ARCHITECTURES=("64:2" "32:3" "32:2" "16:2")

# Get architecture for this array task
ARCH=${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}
HIDDEN_FEATURES=$(echo $ARCH | cut -d: -f1)
HIDDEN_LAYERS=$(echo $ARCH | cut -d: -f2)

RUN_NAME="${HIDDEN_FEATURES}x${HIDDEN_LAYERS}_seed${SEED}"

# Training parameters
NUM_STEPS=500000
BATCH_SIZE=65536
LR=1e-4
LR_DECAY=0.99998
LR_MIN=1e-6

echo "=========================================="
echo "SIREN Architecture Test: ${HIDDEN_FEATURES}×${HIDDEN_LAYERS}"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
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
echo "  output_dir: ${OUTPUT_BASE}"
echo "  num_steps: ${NUM_STEPS}"
echo ""

cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax
mkdir -p ${OUTPUT_BASE}

CONTAINER="/sdf/group/neutrino/pgranger/larnd-sim-jax.sif"

singularity exec --nv -B /sdf ${CONTAINER} /bin/bash -c "
    export PYTHONNOUSERSITE=1
    pip3 install --quiet --user .
    python3 -m src.siren.train_surrogate \
        --output_dir ${OUTPUT_BASE} \
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
echo "Results in: ${OUTPUT_BASE}/${RUN_NAME}/"
echo "=========================================="
