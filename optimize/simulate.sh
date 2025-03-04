#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino
#SBATCH --job-name=larndsim
#SBATCH --output=logs/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1:00:00
##SBATCH --array=0
#SBATCH --array=0,1,2,3,4,5,6,7,8

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

# INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
INPUT_FILE=prepared_data/input_0.h5

echo $PWD/optimize
ls -lht $PWD


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.simulate \
python3 -m optimize.simulate \
    --input_file ${INPUT_FILE} \
    --output_file output/output_0.h5 \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 0 \
    --signal_length 191 \
    --mode 'parametrized'
    # --lut_file /home/pgranger/larnd-sim/jit_version/original/build/lib/larndsim/bin/response_44.npy
    # --number_pix_neighbors 0 \
    # --signal_length 191 \
    # --mode 'parametrized' 
