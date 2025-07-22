#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:cider-nu
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_scan
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1:00:00
#SBATCH --array=0-8

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
# PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=100
ITERATIONS=10
DATA_SEED=1
N_NEIGH=4
#LOSS=mse_adc
LOSS=chamfer_3d


## muon
INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5

# proton 5000 events
#INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5
#INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5
# proton 1E5 events
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2826844_p_1E5/job_64452404_0001/output_64452404_0001-edepsim.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2826844_p_1E5/job_64452404_0001/output_64452404_0001-edepsim.h5

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
#LABEL=${PARAM}_loss_muon_data_closure_target_nom_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_chamfer_${UUID}
LABEL=${PARAM}_loss_muon_edep_no_noise_closure_target_nom_n_neigh_${N_NEIGH}_bt${BATCH_SIZE}_dtsd${DATA_SEED}_${LOSS}_${UUID}


# singularity exec --bind /sdf,$SCRATCH python-jax.sif python3 -m optimize.example_run \
# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install .
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 20 \
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
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length 150 \
    --mode 'lut' \
    --lut_file src/larndsim/detector_properties/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --fit_type 'scan' \
    --diffusion_in_current_sim \
    --non_deterministic \
    --sim_seed_strategy 'different' \
    --no-noise \
    --mc_diff \
    --scan_tgt_nom \
    --detector_props src/larndsim/detector_properties/module0.yaml \
    --live_selection \
    #--chamfer_match_z
    #--random_ntrack \
    #--read_target \
    #--max_clip_norm_val 1 \
    #--mc_diff \
    #--scan_tgt_nom \
    #--print_input
    # --loss_fn SDTW \
    # --lut_file /home/pgranger/larnd-sim/jit_version/original/build/lib/larndsim/bin/response_44.npy
    # --keep_in_memory
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized' 
    # --loss_fn space_match
    #--lr_scheduler exponential_decay \
    #--lr_kw '{"decay_rate" : 0.98}' \
"

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \
