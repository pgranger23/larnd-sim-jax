#!/bin/bash
#SBATCH --job-name=ci_sur_fit
#SBATCH --partition=ampere
#SBATCH --account=mli:cider-ml
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=ci_tests/ci_surrogate_fit-%j.out
#SBATCH --error=ci_tests/ci_surrogate_fit-%j.err

echo "=========================================="
echo "CI Test: Surrogate Fit"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs -I {} echo "GPU: {}"
echo ""

cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax

# Create ci_tests directory if it doesn't exist
mkdir -p ci_tests

singularity exec --nv -B /sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif /bin/bash -c "
    export PYTHONNOUSERSITE=1
    export PYTHONPATH=/sdf/home/c/cjesus/LArTPC/larnd-sim-jax/src:\$PYTHONPATH
    python3 -m optimize.example_run \
    --mode surrogate \
    --surrogate_model src/siren/trained_surrogate.npz \
    --data_sz -1 \
    --max_nbatch 5 \
    --params optimize/scripts/params_test.yaml \
    --input_file_sim prepared_data/input_1.h5 \
    --input_file_tgt prepared_data/input_1.h5 \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed 1 \
    --out_label ci_surrogate_fit \
    --test_name ci_tests \
    --seed 3 \
    --optimizer_fn Adam \
    --random_ntrack \
    --iterations 100 \
    --max_batch_len 5 \
    --lr_scheduler exponential_decay \
    --lr_kw '{\"decay_rate\":0.97}' \
    --track_z_bound 28 \
    --max_clip_norm_val 1 \
    --clip_from_range \
    --electron_sampling_resolution 0.01 \
    --number_pix_neighbors 2 \
    --signal_length 150 \
    --loss_fn mse_adc \
    --fit_type chain \
    --mc_diff
"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "CI TEST PASSED: Surrogate fit completed successfully"
else
    echo "CI TEST FAILED: Surrogate fit failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
