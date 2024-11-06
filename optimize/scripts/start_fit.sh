#!/bin/bash

#SBATCH --partition=ampere
#SBATCH --account=neutrino
#
#SBATCH --job-name=larndsim
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus a100:1
#SBATCH --time=3:00:00

#BASE DECLARATIONS
TARGET_SEED=3
PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=300
ITERATIONS=500
DATA_SEED=1
INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
SIF_FILE=/sdf/home/p/pgranger/larndsim-jax.sif
UUID=$(uuidgen)
#DECLARATIONS



# export JAX_LOG_COMPILES=1
# singularity exec --bind /sdf,$SCRATCH python-jax.sif python3 -m optimize.example_run \

singularity exec --bind /sdf,$SCRATCH ${SIF_FILE} python3 -m optimize.example_run \
    --print_input \
    --data_sz -1 \
    --max_nbatch 40 \
    --params ${PARAMS} \
    --input_file ${INPUT_FILE} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --no-noise \
    --data_seed ${DATA_SEED} \
    --num_workers 0 \
    --out_label seed${TARGET_SEED}_tdiff-vdrift_ds${DATA_SEED}_adam_SDTW_lr1e-2_5trk_test_${UUID} \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --link-vdrift-eField \
    --lr_scheduler exponential_decay \
    --lr_kw '{"decay_rate" : 0.95}' \
    --track_z_bound 28 \
    --max_clip_norm_val 1 \
    --loss_fn SDTW \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 3 \
    --signal_length 300 \
    --mode 'parametrized'
    # --lut_file /home/pgranger/larnd-sim/jit_version/original/build/lib/larndsim/bin/response_44.npy
    # --keep_in_memory
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized'
    # --profile_gradient 
    # --loss_fn space_match


# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \