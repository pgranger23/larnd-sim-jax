#!/bin/bash
#SBATCH --job-name=siren_500k
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=siren_training/siren_seed46_500k-%j.out
#SBATCH --error=siren_training/siren_seed46_500k-%j.err

echo "=========================================="
echo "SIREN Training 500k Steps"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

SEED=46
RUN_NAME="seed46_500k"
NUM_STEPS=500000
BATCH_SIZE=65536
LEARNING_RATE=1e-4
LR_DECAY_RATE=0.99998
LR_MIN=1e-6

echo "Parameters:"
echo "  seed: $SEED"
echo "  run_name: $RUN_NAME"
echo "  num_steps: $NUM_STEPS"
echo "  batch_size: $BATCH_SIZE"
echo "  learning_rate: $LEARNING_RATE"
echo "  lr_decay_rate: $LR_DECAY_RATE"
echo "  lr_min: $LR_MIN"
echo ""

cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax
mkdir -p siren_training

CONTAINER="/sdf/group/neutrino/pgranger/larnd-sim-jax.sif"

singularity exec --nv -B /sdf ${CONTAINER} /bin/bash -c "
    export PYTHONNOUSERSITE=1
    pip3 install --quiet --user .
    python3 -m src.siren.train_surrogate \
        --seed ${SEED} \
        --run_name ${RUN_NAME} \
        --num_steps ${NUM_STEPS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --lr_decay_rate ${LR_DECAY_RATE} \
        --lr_min ${LR_MIN} \
        --use_cdf \
        --square_output \
        --checkpoint_every 50000 \
        --plot_every 10000
"

echo ""
echo "Finished: $(date)"
