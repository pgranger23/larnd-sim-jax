#!/bin/bash

#BASE DECLARATIONS

TARGET_SEED=1
PARAMS=optimize/scripts/params_test.yaml
BATCH_SIZE=100
ITERATIONS=20
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
LOSS=mse_adc

GPU=FALSE

#Read in some input arguments
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --gpu)
            GPU=TRUE
            shift # past argument
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
    esac
done

### proton 5000 events
INPUT_FILE_TGT=prepared_data/input_1.h5
INPUT_FILE_SIM=prepared_data/input_1.h5

#DECLARATIONS

#ONLY TESTING WITH 2 NEIGHBORS FOR FASTER CALCS

python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch 2 \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --no-noise-target \
    --data_seed ${DATA_SEED} \
    --out_label scan_test${MAX_CLIP_NORM_VAL}_bt${BATCH_SIZE}_tgtsd${TARGET_SEED}_dtsd${DATA_SEED}_adam_${LOSS}_target \
    --test_name fit_noise \
    --seed ${TARGET_SEED} \
    --random_ntrack \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution 0.01 \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --number_pix_neighbors 2 \
    --signal_length 150 \
    --mode 'lut' \
    --loss_fn ${LOSS} \
    --fit_type 'scan' \
    --sim_seed_strategy 'same' \
    $( [ "$GPU" == "FALSE" ] && echo "--cpu_only" ) \
    --scan_tgt_nom \
    --mc_diff
# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \

