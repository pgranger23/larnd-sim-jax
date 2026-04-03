#!/bin/bash
# Submit Taylor scan jobs.
#
# Usage:
#   bash tests/taylor/submit_taylor_scan.sh 0        # single input
#   bash tests/taylor/submit_taylor_scan.sh all       # all inputs (one job each)

set -e

REPO_DIR=/sdf/home/c/cjesus/LArTPC/larnd-sim-jax
SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

submit_one() {
    local INPUT_ID=$1
    local INPUT_FILE="${REPO_DIR}/prepared_data/input_${INPUT_ID}.h5"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Skipping input_${INPUT_ID}: file not found"
        return
    fi

    mkdir -p "${REPO_DIR}/tests/taylor/logs"
    mkdir -p "${REPO_DIR}/tests/taylor/results"

    sbatch <<SBATCH_EOF
#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=taylor_${INPUT_ID}
#SBATCH --output=${REPO_DIR}/tests/taylor/logs/taylor_scan_${INPUT_ID}-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00

cd ${REPO_DIR}

nvidia-smi

apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip install . 2>&1 | tail -1
PYTHONPATH=. python3 tests/taylor/run_taylor_scan.py --input_id ${INPUT_ID}
"
SBATCH_EOF

    echo "Submitted input_${INPUT_ID}"
}

if [ -z "$1" ]; then
    echo "Usage: bash tests/taylor/submit_taylor_scan.sh <input_id|all>"
    exit 1
fi

if [ "$1" = "all" ]; then
    for f in ${REPO_DIR}/prepared_data/input_*.h5; do
        # Skip shifted variants
        if [[ "$f" == *"_shifted"* ]]; then
            continue
        fi
        # Extract ID from filename
        ID=$(basename "$f" | sed 's/input_//;s/\.h5//')
        submit_one "$ID"
    done
else
    submit_one "$1"
fi
