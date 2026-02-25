#!/bin/bash

#SBATCH --partition=ampere
##SBATCH --account=neutrino:dune-ml
#SBATCH --account=mli:cider-ml
#SBATCH --job-name=scan_collapsed_mse
#SBATCH --output=logs/scan/job_collapsed_mse_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=3:00:00
#SBATCH --array=0

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
# PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=100
ITERATIONS=10
DATA_SEED=1
LOSS=mse_adc
MAX_CLIP_NORM_VAL=1

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0.h5

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
UUID=$(uuidgen)

nvidia-smi

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}
LABEL=scan_collapsed_mse_sigma10_${PARAM}_${UUID}

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install .
pip install tensorflow tensorboard xprof
pip install "nvidia-cuda-cupti-cu12==12.2.*" "nvidia-cudnn-cu12==8.9.*" "nvidia-cublas-cu12==12.2.*" "nvidia-cufft-cu12==11.0.*"
JAX_EXPLAIN_CACHE_MISSES=1 python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 1 \
    --params ${PARAM} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --no-noise-target \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name scan_collapsed_mse \
    --seed ${TARGET_SEED} \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution 0.01 \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --number_pix_neighbors 2 \
    --signal_length 150 \
    --mode 'lut' \
    --loss_fn ${LOSS} \
    --fit_type 'scan' \
    --sim_seed_strategy 'same' \
    --scan_tgt_nom \
    --mc_diff \
    --probabilistic-sim \
    --loss_fn_kw '{\"sigma\": 10, \"collapsed\": false}' \
    --profile
"
