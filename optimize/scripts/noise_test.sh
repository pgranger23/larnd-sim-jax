#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:nu-ml-dev
##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:cider-nu
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_scan
#SBATCH --output=logs/scan/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00
#SBATCH --array=1-8

#BASE DECLARATIONS

THRESHOLDS=(3000 4000 5000 6000 7000 8000 9000 10000)

THRESHOLD=${THRESHOLDS[$SLURM_ARRAY_TASK_ID-1]}

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif


nvidia-smi

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install .
python3 optimize/analysis_scripts/noise_test.py --threshold $THRESHOLD -n 10000 --no-noise
"

# nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop-shutdown python3 -m optimize.example_run \
