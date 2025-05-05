INPUT_FILE=/home/pgranger/larnd-sim-latest/edepsim-output.h5

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown
python3 -m optimize.simulate \
    --input_file ${INPUT_FILE} \
    --output_file output/output-test.h5 \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 0 \
    --signal_length 191 \
    --mode 'parametrized' \
    --noise \
    --jac \
    --gpu