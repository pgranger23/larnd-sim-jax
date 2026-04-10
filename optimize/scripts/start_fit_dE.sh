#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
#SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_dE
#SBATCH --output=logs/fit_noise/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1:00:00
#SBATCH --array=1

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=200
MAX_NBATCH=500
ITERATIONS=10000
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
LOSS=mse_adc #nll #llhd #mse_adc
SEED_STRATEGY=different #random #different_epoch
SAMPLING_STEP=0.01 # cm
N_NEIGH=4
MODE="lut"  #"parametrized"
LR_SCHEDULER=warmup_exponential_decay_schedule
SIGNAL_LENGTH=200
NORM=sigmoid #divide
DE_LR=1e-4
REG_L2=1e-2

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_10.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.05cm_ntrack_10.h5

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
UUID=$(uuidgen)

LABEL=r13_stopp_fit_dE_dx0.01_stoc_noise_tgtsim_seed_${SEED_STRATEGY}_n_neigh${N_NEIGH}_${MODE}_e_sampling_${SAMPLING_STEP}cm_signalL${SIGNAL_LENGTH}_gradclip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbtach${MAX_NBATCH}_dtsd${DATA_SEED}_Adam_${LOSS}_de_lr${DE_LR}_reg${REG_L2}_${NORM}

nvidia-smi


# export JAX_LOG_COMPILES=1


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name fit_noise \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length ${SIGNAL_LENGTH} \
    --mode ${MODE} \
    --lut_file ../Data_selection/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --clip_from_range \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.98, \"init_value\" : 0, \"warmup_steps\": 2000}' \
    --shuffle_bt \
    --normalization_scheme ${NORM} \
    --fit_segment_de \
    --segment_de_mode 'segment-only'\
    --segment_de_optimizer Adam \
    --no_chop \
    --no_pad \
    --segment_de_lr ${DE_LR} \
    --segment_reg_l2 ${REG_L2} \
    --set_target_vals Ab 0.8 kb 0.0486 eField 0.50 tran_diff 8.8e-6 long_diff 4.0e-6 lifetime 2200 \
    #--probabilistic_sim \
    #--debug_nans \
    #--no_chop \
    #--profile \
    #--debug_nans \
    #--no-noise-guess \
    #--no-noise \
    #--print_input
"
