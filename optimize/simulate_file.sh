#!/bin/bash

#SBATCH --partition=ampere

#SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_jac
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00
#SBATCH --array=6


SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax/larndsim-jax_noise.sif

INPUT_FILE=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2826844_p_1E5/job_64452404_0001/output_64452404_0001-edepsim.h5
RESOLUTIONS=(0.001 0.002 0.003 0.004 0.005 0.0075 0.01 0.015 0.02 0.03 0.04 0.05)
# RESOLUTIONS=(0.001 0.002 0.003 0.004)
RESOLUTION=${RESOLUTIONS[$SLURM_ARRAY_TASK_ID]}

UUID=$(uuidgen)

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown
# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.simulate \

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install .
python3 -m optimize.simulate \
    --input_file ${INPUT_FILE} \
    --output_file output/output-${RESOLUTION}-${UUID}-noise-ldiff.h5 \
    --electron_sampling_resolution ${RESOLUTION} \
    --number_pix_neighbors 4 \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --signal_length 150 \
    --noise \
    --mode 'lut' \
    --n_events 100 \
    --gpu
"