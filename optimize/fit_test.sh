#!/bin/bash

#BASE DECLARATIONS

TARGET_SEED=3
PARAMS=optimize/scripts/params_test.yaml
BATCH_SIZE=200
ITERATIONS=200
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
LOSS=chamfer_3d

### proton 5000 events
INPUT_FILE_TGT=prepared_data/input_1.h5
#INPUT_FILE_TGT=output/jax_ref/output_parametrized_1.h5.npz
INPUT_FILE_SIM=prepared_data/input_1.h5

#DECLARATIONS

python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 5 \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --no-noise-guess \
    --no-noise-target \
    --data_seed ${DATA_SEED} \
    --out_label fit_test_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_target \
    --test_name fit_noise \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --lr_scheduler exponential_decay \
    --lr_kw '{"decay_rate" : 0.97}' \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 0 \
    --signal_length 191 \
    --mode 'parametrized' \
    --loss_fn ${LOSS} \
    --fit_type 'chain' \
    --cpu_only \
    #--read_target
# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \

