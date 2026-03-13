#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
#SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_scan
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=2:00:00
##SBATCH --array=3,4
#SBATCH --array=0-5

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
# PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=200
ITERATIONS=50
DATA_SEED=1
N_NEIGH=4
ELEC_RESOLUTION=0.01
SEED_STRATEGY=different #random #different
LOSS=mse_adc
SIGNAL_LENGTH=400 #150
N_BATCH=400

### true proton
INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm.h5
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_proton_edep_2cm_range_0.1-cm_dEdx.h5

## true stopping muon
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_CSDA.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_CSDA.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_mod0.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx_scale5times.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_mod0_dEdx_scale5times.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx+10.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_mod0_dEdx+10.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx+20.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_mod0_dEdx+20.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_CSDA_dEdx+20.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_dEdx_CSDA_dEdx+20.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0.h5
##INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx_gaus1smear.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx_gaus3smear.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx_scale5times.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_ending_muon_edep_5cm_vol2cm_range_0.2-cm_mod0_dEdx_scale5times.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_new.h5
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_new.h5
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_force_agree_1MeVcm_new.h5
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_force_agree_1MeVcm_new.h5
# full truth
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_new.h5
# 'reco' dE/dx
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_dEdx_new.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_new.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5cm_dEdx_force_agree_1MeVcm_new.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_stopping_muon_edep_5cm_vol2cm_range0.5-5cm_dEdx_force_agree_1MeVcm_new.h5

## true through going muon
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/diffsim_input/true_through_muon_edep_10cm_vol1cm.h5

# data
#INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/M1_data_mockup_seg_2022_02_07_23_09_05_CET_2000ev.h5
#INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/M1_hits_target_2022_02_07_23_09_05_CET_2000ev.npz
#INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/M1_data_mockup_seg_2022_02_07_23_09_05_CET_2000ev.h5


SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
UUID=$(uuidgen)
#DECLARATIONS

nvidia-smi

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}
#LABEL=${PARAM}_true_proton_edep_range_0.1-cm_dEdx_noise_tgt_sim_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
#LABEL=${PARAM}_true_stopping_muon_edep_range_0.2-cm_dEdx_CSDA_noise_tgt_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
#LABEL=${PARAM}_true_stopping_muon_edep_range_0.2-cm_dEdx_edep_mod0_gaus10smear_no_noise_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
#LABEL=${PARAM}_true_stopping_muon_closure_range_0.2-cm_dEdx*5_noise_tgtsim_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
#LABEL=${PARAM}_true_stopping_muon_closure_range_0.2-cm_dEdx+10_no_noise_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
#LABEL=${PARAM}_true_stopping_muon_closure_range_0.2-cm_dEdx+20_no_noise_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
#LABEL=${PARAM}_true_stopping_muon_closure_range_0.2-cm_dEdx+20_CSDA_noise_tgtsim_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
#LABEL=${PARAM}_true_stopping_muon_range_0.2-cm_dEdx_gaus3smear_no_noise_nominal_tgt_non_prob_hits_sampled_seed_strategy_${SEED_STRATEGY}_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}
LABEL=${PARAM}_true_stopping_proton_reco_dEdx_range_0.1-cm_noise_tgtsim_nominal_tgt_non_prob_hits_n_neigh_${N_NEIGH}_seed_${SEED_STRATEGY}_signal_length${SIGNAL_LENGTH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}

# singularity exec --bind /sdf,$SCRATCH python-jax.sif python3 -m optimize.example_run \
# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install .
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${N_BATCH} \
    --params ${PARAM} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name scan\
    --seed ${TARGET_SEED} \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --electron_sampling_resolution ${ELEC_RESOLUTION} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length ${SIGNAL_LENGTH} \
    --mode 'lut' \
    --lut_file src/larndsim/detector_properties/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --fit_type 'scan' \
    --non_deterministic \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --scan_tgt_nom \
    --detector_props src/larndsim/detector_properties/module0.yaml \
    #--no-noise \
    #--no-noise-guess \
    #--no-noise \
    #--live_selection \
    #--chamfer_match_z
    #--random_ntrack \
    #--read_target \
    #--max_clip_norm_val 1 \
    #--mc_diff \
    #--scan_tgt_nom \
    #--print_input
    # --loss_fn SDTW \
    # --keep_in_memory
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized' 
    # --loss_fn space_match
    #--lr_scheduler exponential_decay \
    #--lr_kw '{"decay_rate" : 0.98}' \
"

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \
