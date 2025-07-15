#!/bin/bash
#SBATCH --partition=ampere

#SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=larndsim
#SBATCH --output=logs/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=1:00:00
#SBATCH --array=1
##SBATCH --array=0,1

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

# INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
INPUT_FILE=prepared_data/input_${SLURM_ARRAY_TASK_ID}.h5
#INPUT_FILE=../Data_selection/event122.h5


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.simulate \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.simulate \
    --input_file ${INPUT_FILE} \
    --output_file output/output_parametrized_${SLURM_ARRAY_TASK_ID}.h5 \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 0 \
    --signal_length 150 \
    --mode 'parametrized' \
    --diffusion_in_current_sim \
    --out_np \
    --mc_diff; \

python3 -m optimize.simulate \
    --input_file ${INPUT_FILE} \
    --output_file output/output_lut_${SLURM_ARRAY_TASK_ID}.h5 \
    --electron_sampling_resolution 0.005 \
    --number_pix_neighbors 4 \
    --signal_length 150 \
    --mode 'lut' \
    --lut_file src/larndsim/detector_properties/response_44.npy \
    --out_np \
    --mc_diff

"
