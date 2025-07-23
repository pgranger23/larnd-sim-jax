#!/bin/bash

#SBATCH --partition=ampere

#SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim
#SBATCH --output=logs/fit_noise/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1

#SBATCH --time=3:00:00
#SBATCH --array=1

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=1000
ITERATIONS=2000
MAX_CLIP_NORM_VAL=1
#MAX_CLIP_NORM_VAL=0.8
DATA_SEED=1
#LOSS=SDTW
LOSS=chamfer_3d
SEED_STRATEGY=different
SAMPLING_STEP=0.0005 #0.0005 #0.005 cm
ADC_NORM=10.

INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/M1_data_mockup_seg_2022_02_07_23_09_05_CET_2000ev.h5
#INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/M1_data_mockup_seg_2022_02_07_23_09_05_CET_2000ev_dune_dtype_dEdx3.h5
INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/M1_hits_target_2022_02_07_23_09_05_CET_2000ev.npz
#INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/M1_hits_target_2022_02_07_23_09_05_CET_2000ev_hit_merge.npz

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
UUID=$(uuidgen)
#DECLARATIONS

nvidia-smi

# export JAX_LOG_COMPILES=1

LABEL=M1_muon_data_signal_lut_vcm_vref_threshold5e3_3par_loss_Q_pos_match_t_e_sampling_${SAMPLING_STEP}cm_seed_strategy_${SEED_STRATEGY}_grad_clip${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_${UUID}

# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.example_run \
# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
#apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} pip install -e .; python3 -m optimize.example_run \
#apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch -1 \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 40 \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name fit_data \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --lr_scheduler exponential_decay \
    --lr_kw '{\"decay_rate\" : 0.99}' \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors 0 \
    --signal_length 191 \
    --mode 'lut' \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --read_target \
    --clip_from_range \
    --chamfer_adc_norm ${ADC_NORM} \
    --no-noise-guess \
    --chamfer_match_z \
    --diffusion_in_current_sim \
    --mc_diff \
    --lut_file /sdf/group/neutrino/cyifan/diff_sim_playground/Data_selection/response_44_v2a_full_tick.npz \
    #--lut_file /sdf/group/neutrino/cyifan/diff_sim_playground/larndsim_dune/larndsim/bin/response_44.npy \
    #--lut_file /sdf/group/neutrino/cyifan/diff_sim_playground/larndsim_dune/larndsim/bin/response_44_v2a_full.npz
    #--match_z
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
