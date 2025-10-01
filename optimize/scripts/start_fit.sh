#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:cider-ml
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim
#SBATCH --output=logs/fit_noise/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=8:00:00
#SBATCH --array=1,2,3,4,5

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=400
ITERATIONS=10000
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
LOSS=mse_adc
SEED_STRATEGY=random #different_epoch
SAMPLING_STEP=0.01 # cm
N_NEIGH=4
MODE="lut"  #"parametrized"
LR_SCHEDULER=warmup_exponential_decay_schedule

### true proton
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm.h5
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm_dEdx.h5

### true stopping muon
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_dEdx_mod0.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_new.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_new.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0.h5
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0_dEdx_gaus1smear.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.1-cm_mod0_dEdx_gaus10smear.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx+20.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_mod0_dEdx+20.h5

INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_CSDA_dEdx+20.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_CSDA_dEdx+20.h5

##INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_new.h5
##INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_new.h5
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_force_agree_1MeVcm_new.h5
#
## full truth
##INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_new.h5
## 'reco' dE/dx
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_dEdx_new.h5
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_new.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_force_agree_1MeVcm_new.h5

## true through going muon
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
UUID=$(uuidgen)
#LABEL=stopping_mu_range_dEdx_6par_n_neigh${N_NEIGH}_mode_${MODE}_noise_e_sampling_${SAMPLING_STEP}cm_seed_strategy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}
#LABEL=stopping_mu_range_0.5-5cm_dEdx_force_agree_1MeVcm_6par_no_noise_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}
#LABEL=stopping_mu_range_0.5-5cm_dEdx_force_agree_1MeVcm_6par_no_noise_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_proton_range_0.1cm_dEdx_6par_no_noise_guess_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_stopmu_range_0.2-cm_dEdx+20_edep_mod0_6par_noise_tgt_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
LABEL=true_stopmu_range_0.2-cm_dEdx+20_edep_CSDA_6par_noise_tgtsim_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}
#LABEL=true_stopmu_range_0.5-5cm_dEdx_CSDA_6par_noise_tgt_n_neigh${N_NEIGH}_mode_${MODE}_e_sampling_${SAMPLING_STEP}cm_seed_stgy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_${LR_SCHEDULER}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}
#DECLARATIONS

nvidia-smi


# export JAX_LOG_COMPILES=1


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 200 \
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
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length 150 \
    --mc_diff \
    --mode ${MODE} \
    --lut_file ../Data_selection/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --clip_from_range \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.98, \"init_value\" : 0, \"warmup_steps\": 200}' \
    #--no-noise-guess \
    #--no-noise \
    #--lr_scheduler exponential_decay \
    #--lr_kw '{\"decay_rate\" : 0.99}' \
    #--live_selection
    #--chamfer_match_z \
    #--print_input
    # --loss_fn SDTW \
    # --lut_file /home/pgranger/larnd-sim/jit_version/original/build/lib/larndsim/bin/response_44.npy
    # --keep_in_memory
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized'
    # --profile_gradient 
    # --loss_fn space_match
"

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \
