#!/bin/bash

NFILES=4
ls -lht output
for ifile in $(seq 0 ${NFILES}); do
    INPUT_FILE=prepared_data/input_${ifile}.h5
    python3 -m optimize.simulate \
        --input_file ${INPUT_FILE} \
        --output_file output/output_${ifile}.h5 \
        --electron_sampling_resolution 0.005 \
        --number_pix_neighbors 3 \
        --signal_length 100 \
        --mode 'lut' \
        --lut_file larndsim/detector_properties/response_44.npy
done
# for ifile in $(seq 0 ${NFILES}); do
#     INPUT_FILE=prepared_data/input_${ifile}.h5
#     python3 -m optimize.simulate \
#         --input_file ${INPUT_FILE} \
#         --output_file output/output_${ifile}.h5 \
#         --electron_sampling_resolution 0.005 \
#         --number_pix_neighbors 0 \
#         --signal_length 191 \
#         --mode 'parametrized'
# done


python3 -m optimize.comparison \
    --ref_output output/jax_ref_local/output_ \
    --output output/output_ \
    --n_files $((${NFILES}+1))