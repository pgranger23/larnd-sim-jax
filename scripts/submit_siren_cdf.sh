#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=siren_cdf
#SBATCH --output=siren_training/cdf_job-%j.out
#SBATCH --error=siren_training/cdf_job-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00

# SIREN CDF Training Script
# Usage:
#   sbatch scripts/submit_siren_cdf.sh                    # CDF-only
#   sbatch scripts/submit_siren_cdf.sh --lambda_deriv 1.0 # CDF + derivative

# Parse arguments (defaults)
RUN_NAME="cdf_only"
LAMBDA_DERIV="0.0"
NUM_STEPS="50000"
BATCH_SIZE="8192"

# Parse command line args
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_name) RUN_NAME="$2"; shift 2 ;;
        --lambda_deriv) LAMBDA_DERIV="$2"; shift 2 ;;
        --num_steps) NUM_STEPS="$2"; shift 2 ;;
        --batch_size) BATCH_SIZE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "=========================================="
echo "SIREN CDF Training"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""
echo "Parameters:"
echo "  run_name: ${RUN_NAME}"
echo "  lambda_deriv: ${LAMBDA_DERIV}"
echo "  num_steps: ${NUM_STEPS}"
echo "  batch_size: ${BATCH_SIZE}"
echo ""

cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax
mkdir -p siren_training

CONTAINER="/sdf/group/neutrino/pgranger/larnd-sim-jax.sif"

singularity exec --nv -B /sdf ${CONTAINER} /bin/bash -c "
    export PYTHONNOUSERSITE=1
    pip3 install --quiet --no-deps .

    python3 -m src.siren.train_surrogate \
        --use_cdf \
        --lambda_deriv ${LAMBDA_DERIV} \
        --run_name ${RUN_NAME} \
        --batch_size ${BATCH_SIZE} \
        --num_steps ${NUM_STEPS} \
        --log_every 100 \
        --val_every 500 \
        --checkpoint_every 5000
"

echo ""
echo "=========================================="
echo "Training finished: $(date)"
echo "Results in: siren_training/${RUN_NAME}/"
echo "=========================================="
