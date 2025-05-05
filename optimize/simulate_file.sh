#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:cider-nu
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_jac
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00
#SBATCH --array=0

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

INPUT_FILE=/sdf/group/neutrino/pgranger/larndsim-jax/data/mixed_sample/edepsim-output.h5

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install .
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
"