#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=taylor_scan
#SBATCH --output=tests/taylor/logs/taylor_scan-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00

cd /sdf/home/c/cjesus/LArTPC/larnd-sim-jax

mkdir -p tests/taylor/logs

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

nvidia-smi

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install . 2>&1 | tail -1
PYTHONPATH=. python3 tests/taylor/run_taylor_scan.py
"
