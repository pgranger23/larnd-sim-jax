#!/bin/bash

#BASE DECLARATIONS

TARGET_SEED=1
PARAMS=optimize/scripts/params_test.yaml
BATCH_SIZE=100
ITERATIONS=10
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
LOSS=chamfer_3d

### proton 5000 events
INPUT_FILE_TGT=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5
INPUT_FILE_SIM=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim_p_only_active_vol_5000.h5

#DECLARATIONS

python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 2 \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --no-noise-guess \
    --no-noise-target \
    --data_seed ${DATA_SEED} \
    --num_workers 0 \
    --out_label p_5E3_6par_noise_tgt_grad_clip${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_target \
    --test_name fit_noise \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 0 \
    --signal_length 191 \
    --mode 'parametrized' \
    --loss_fn ${LOSS} \
    --fit_type 'scan' \
    --cpu_only
# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \

