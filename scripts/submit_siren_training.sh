#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=siren_train
#SBATCH --output=siren_training/job-%j.out
#SBATCH --error=siren_training/job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=12:00:00

# SIREN Surrogate Training Script for S3DF
# Usage: sbatch scripts/submit_siren_training.sh

echo "=========================================="
echo "SIREN Surrogate Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Change to project directory
cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax

# Create output directory
mkdir -p siren_training

# Container path
CONTAINER="/sdf/group/neutrino/pgranger/larnd-sim-jax.sif"

echo "Running training inside container..."
echo ""

singularity exec --nv -B /sdf ${CONTAINER} /bin/bash -c "
    # Install package
    pip3 install --quiet .

    # Run training
    python3 -m src.siren.train_surrogate \
        --output_dir siren_training \
        --batch_size 65536 \
        --num_steps 50000 \
        --log_every 100 \
        --val_every 500 \
        --checkpoint_every 5000
"

echo ""
echo "=========================================="
echo "Training finished: $(date)"
echo "=========================================="
