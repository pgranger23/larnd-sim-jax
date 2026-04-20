#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00
#SBATCH --array=0,1,2,3,4,5

# --- CONFIGURATION SELECTION ---
# Example format: A1-B1-C1-D1
# Change this variable or pass it as an argument
CONFIG="A3-B2-C2-D2"

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
BATCH_SIZE=200
MAX_NBATCH=50
ITERATIONS=50
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
#LOSS=mse_adc 
SEED_STRATEGY=same #different 
SAMPLING_STEP=0.01 
N_NEIGH=4
MODE="lut"
LR_SCHEDULER=warmup_exponential_decay_schedule
SIGNAL_LENGTH=200

# --- LOGIC FOR COMBINATIONS ---

# Parse CONFIG (A, B, C, D)
IFS='-' read -r CONF_A CONF_B CONF_C CONF_D <<< "$CONFIG"

# A: DX / TGT File / Chop Logic
if [ "$CONF_A" == "A1" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5
    CHOP_FLAG="" # No --no_chop for A1
    DX_LABEL="dxvaried"
elif [ "$CONF_A" == "A2" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
    CHOP_FLAG="--no_chop" # Add --no_chop for A2
    DX_LABEL="dx0.01"
elif [ "$CONF_A" == "A3" ]; then
    INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_1250070/job_25210996_0000/output_25210996_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.2cm.h5
    CHOP_FLAG="--no_chop" # Add --no_chop for A2
    DX_LABEL="dx0.1"
fi

# B: SIM Input File
if [ "$CONF_A" == "A1" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5
        B_LABEL="closure"
    else
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm_dEdx.h5
        B_LABEL="reco_dE"
    fi
elif [ "$CONF_A" == "A2" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
        B_LABEL="closure"
    elif [ "$CONF_B" == "B2" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.05cm.h5
        B_LABEL="reco_dE"
    elif [ "$CONF_B" == "B3" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_true_traj_start_end_reco_seg_step_0.01cm_range_0.05cm.h5
        B_LABEL="reco_traj_st_ed_pos_dE"
    elif [ "$CONF_B" == "B4" ]; then
	INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_max_evt_37815_reco_pos_dE_seg_step_0.01cm_range_0.05cm.h5
        B_LABEL="reco_pos_dE"
    fi
elif [ "$CONF_A" == "A3" ]; then
    if [ "$CONF_B" == "B1" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_1250070/job_25210996_0000/output_25210996_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.2cm.h5
        B_LABEL="closure"
    elif [ "$CONF_B" == "B2" ]; then
        INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_1250070/job_25210996_0000/output_25210996_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.2cm.h5
        B_LABEL="reco_dE"
    fi
fi

# C: Normalization / Params
if [ "$CONF_C" == "C1" ]; then
    NORM=divide
    PARAMS=optimize/scripts/param_list.yaml
    LR_LABEL="2e-3"
elif [ "$CONF_C" == "C2" ]; then
    NORM=sigmoid
    if [ "$CONF_B" == "B1" ]; then
    	PARAMS=optimize/scripts/param_list_main.yaml
	LR_LABEL="0.03"
    elif [ "$CONF_B" == "B2" ]; then
    	PARAMS=optimize/scripts/param_list_C2_B2.yaml
	LR_LABEL="0.1"
    elif [ "$CONF_B" == "B3" ]; then
    	PARAMS=optimize/scripts/param_list_main_reco.yaml
	LR_LABEL="1"
    elif [ "$CONF_B" == "B4" ]; then
    	PARAMS=optimize/scripts/param_list_main_reco.yaml
	LR_LABEL="1"
    fi
elif [ "$CONF_C" == "C3" ]; then
    NORM=exp_log
    PARAMS=optimize/scripts/param_list_main.yaml
    LR_LABEL="1"
fi

# D: Noise / Probabilistic Flag
if [ "$CONF_D" == "D1" ]; then
    PROB_FLAG=""
    D_LABEL="stoc_noise"
    LOSS=mse_adc
elif [ "$CONF_D" == "D2" ]; then
    PROB_FLAG="--probabilistic_sim"
    D_LABEL="prob_noise"
    LOSS=llhd
fi

#PARAMS=optimize/scripts/param_list_${CONFIG}.yaml

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

# Generate Label
LABEL="${PARAM}_${CONFIG}_stopp_${B_LABEL}_${DX_LABEL}_${D_LABEL}_tgtsim_seed_${SEED_STRATEGY}_n_neigh${N_NEIGH}_${MODE}_e_sampling_${SAMPLING_STEP}cm_signalL${SIGNAL_LENGTH}_gradclip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbtach${MAX_NBATCH}_dtsd${DATA_SEED}_adam_${LOSS}_${NORM}"

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

echo "Running Configuration: $CONFIG"
echo "Label: $LABEL"

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params ${PARAM} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type scan \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name scan \
    --scan_tgt_nom \
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
    --lr_kw '{\"decay_rate\" : 0.99, \"init_value\" : 0, \"warmup_steps\": 1000}' \
    --shuffle_bt \
    --normalization_scheme ${NORM} \
    ${PROB_FLAG} \
    ${CHOP_FLAG} \
    --print_input
"
