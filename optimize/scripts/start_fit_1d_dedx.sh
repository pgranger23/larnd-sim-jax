#!/bin/bash

#SBATCH --partition=ampere

#SBATCH --account=mli:cider-ml
###BATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_dedx
#SBATCH --output=logs/fit_noise/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=10:00:00
#SBATCH --array=0-8

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=200
ITERATIONS=5000
MAX_CLIP_NORM_VAL=100
DATA_SEED=1
LOSS=llhd
SEED_STRATEGY=random
SAMPLING_STEP=0.01 # cm
N_NEIGH=4
MODE="lut"
LR_SCHEDULER=warmup_exponential_decay_schedule

# per-segment dEdx fitting settings
DEDX_PRIOR_WEIGHT=0.1
DEDX_LR=1e-2  # typically lower than global LR
#DEDX_LR=0.1  # typically lower than global LR
DEDX_START_ITER=0  # calibration-only warm-up iterations before dEdx is activated
DEDX_FREEZE_ITER=5100

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

## true through going muon
INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_main.sif
LABEL=true_throughmuons_6par_dedxfit_priw${DEDX_PRIOR_WEIGHT}_dlr${DEDX_LR}_dsi${DEDX_START_ITER}_noise_tgtsim_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}

ONAME=fit_noise_dedx_${SLURM_ARRAY_JOB_ID}
nvidia-smi

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
export PYTHONPATH=\$PWD/src:\$PWD:\$PYTHONPATH; \
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 20 \
    --params ${PARAM} \
    --lr 1e-1 \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name $ONAME \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --random_nevent \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length 150 \
    --mc_diff \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 500}' \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --fit_dedx \
    --dedx_prior_weight ${DEDX_PRIOR_WEIGHT} \
    --dedx_lr ${DEDX_LR} \
    --dedx_start_iter ${DEDX_START_ITER} \
    --dedx_freeze_iter ${DEDX_FREEZE_ITER} \
    --probabilistic_sim \
    --probabilistic_sampling_target
"
