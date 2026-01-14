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
#SBATCH --time=1:00:00
##SBATCH --array=1,2,3,4,5
##SBATCH --time=1:00:00
#SBATCH --array=0-8

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

PARAMS=("Ab" "kb" "eField" "tran_diff" "long_diff" "lifetime" "shift_z" "shift_x" "shift_y")
PARAM=${PARAMS[$SLURM_ARRAY_TASK_ID]}

TARGET_SEED=$SLURM_ARRAY_TASK_ID
# PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=500
ITERATIONS=400
MAX_CLIP_NORM_VAL=1
#MAX_CLIP_NORM_VAL=0.8
DATA_SEED=1
#LOSS=SDTW
LOSS=mse_adc
SEED_STRATEGY=same
SAMPLING_STEP=0.01 #0.0005 #0.005 cm
N_NEIGH=4
MODE="lut"  #"parametrized"

### proton 1E5 events
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2826844_p_1E5/job_64452404_0001/output_64452404_0001-edepsim.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2826844_p_1E5/job_64452404_0001/output_64452404_0001-edepsim.h5

#### proton 5000 events
#INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5
#INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5
##INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000_NIST_dEdx_cubic.h5

## muon 100 events
INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_noise.sif
UUID=$(uuidgen)
LABEL=mu_2par_Ab_kb_n_neigh${N_NEIGH}_mode_${MODE}_diff_curr_no_noise_e_sampling_${SAMPLING_STEP}cm_seed_strategy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}
#DECLARATIONS

nvidia-smi


# export JAX_LOG_COMPILES=1


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.example_run \
# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
#apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} pip install -e .; python3 -m optimize.example_run \
#apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 5 \
    --params ${PARAM} \
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
    --lr_scheduler exponential_decay \
    --lr_kw '{\"decay_rate\" : 0.95}' \
    --lr 0.01 \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length 150 \
    --mode ${MODE} \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --clip_from_range
    --probabilistic-target
    # --live_selection
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
