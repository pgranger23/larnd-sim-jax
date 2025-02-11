#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --account=neutrino:ml-dev
#SBATCH --job-name=diffsim
#SBATCH --output=logs/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1:00:00
#SBATCH --array=1

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=800
ITERATIONS=2
DATA_SEED=1
INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5
INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000_NIST_dEdx_cubic.h5
#INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
UUID=$(uuidgen)
#DECLARATIONS

nvidia-smi



# export JAX_LOG_COMPILES=1


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.example_run \
# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch -1 \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --no-noise-guess \
    --data_seed ${DATA_SEED} \
    --num_workers 0 \
    --out_label p_noise_tgt_sd${TARGET_SEED}__data_sd${DATA_SEED}_adam_chamfer3D_${UUID} \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --link-vdrift-eField \
    --lr_scheduler exponential_decay \
    --lr_kw '{"decay_rate" : 0.99}' \
    --track_z_bound 28 \
    --max_clip_norm_val 1 \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 0 \
    --signal_length 191 \
    --mode 'parametrized' \
    --loss_fn chamfer_3d
    # --loss_fn SDTW \
    # --lut_file /home/pgranger/larnd-sim/jit_version/original/build/lib/larndsim/bin/response_44.npy
    # --keep_in_memory
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized'
    # --profile_gradient 
    # --loss_fn space_match


# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \
