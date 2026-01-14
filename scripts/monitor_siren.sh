#!/bin/bash
# SIREN Training Monitor Script
# Usage:
#   ./scripts/monitor_siren.sh          # Tail the log file
#   ./scripts/monitor_siren.sh --plot   # Regenerate training plot
#   ./scripts/monitor_siren.sh --status # Show job status and summary

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"
OUTPUT_DIR="${PROJECT_DIR}/siren_training"
CONTAINER="/sdf/group/neutrino/pgranger/larnd-sim-jax.sif"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

cd "${PROJECT_DIR}"

case "$1" in
    --plot)
        echo -e "${BLUE}Regenerating training plot...${NC}"
        if [ ! -f "${OUTPUT_DIR}/history.json" ]; then
            echo "Error: No history.json found in ${OUTPUT_DIR}"
            exit 1
        fi

        singularity exec -B /sdf ${CONTAINER} python3 -c "
import sys
sys.path.insert(0, '.')
from src.siren.analysis.visualize import plot_training_history
import matplotlib
matplotlib.use('Agg')
plot_training_history('${OUTPUT_DIR}/history.json', '${OUTPUT_DIR}/training_progress.png')
"
        echo -e "${GREEN}Plot saved to: ${OUTPUT_DIR}/training_progress.png${NC}"
        ;;

    --status)
        echo -e "${BLUE}=== SIREN Training Status ===${NC}"
        echo ""

        # Check for running jobs
        echo -e "${YELLOW}Active jobs:${NC}"
        squeue -u $USER --name=siren_train 2>/dev/null || echo "No jobs found"
        echo ""

        # Check output directory
        if [ -d "${OUTPUT_DIR}" ]; then
            echo -e "${YELLOW}Output directory: ${OUTPUT_DIR}${NC}"

            # Latest checkpoint
            LATEST_CKPT=$(ls -t ${OUTPUT_DIR}/checkpoint_step_*.npz 2>/dev/null | head -1)
            if [ -n "$LATEST_CKPT" ]; then
                echo "  Latest checkpoint: $(basename $LATEST_CKPT)"
            fi

            # History file
            if [ -f "${OUTPUT_DIR}/history.json" ]; then
                STEPS=$(singularity exec -B /sdf ${CONTAINER} python3 -c "
import json
with open('${OUTPUT_DIR}/history.json') as f:
    h = json.load(f)
print(f\"Steps: {h['step'][-1] if h['step'] else 0}\")
print(f\"Train loss: {h['train_loss'][-1]:.6f}\" if h['train_loss'] else '')
print(f\"Val loss: {h['val_loss'][-1]:.6f}\" if h['val_loss'] else '')
" 2>/dev/null)
                echo "  $STEPS"
            fi

            # Log files
            LOG_COUNT=$(ls ${OUTPUT_DIR}/job-*.out 2>/dev/null | wc -l)
            echo "  Log files: $LOG_COUNT"
        else
            echo "  Output directory not found"
        fi
        ;;

    *)
        # Default: tail log file
        LOG_FILE=$(ls -t ${OUTPUT_DIR}/job-*.out 2>/dev/null | head -1)

        if [ -z "$LOG_FILE" ]; then
            echo "No log file found in ${OUTPUT_DIR}"
            echo ""
            echo "Usage:"
            echo "  $0          # Tail the log file"
            echo "  $0 --plot   # Regenerate training plot"
            echo "  $0 --status # Show job status"
            exit 1
        fi

        echo -e "${GREEN}Tailing: ${LOG_FILE}${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo -e "Run '${0} --plot' in another terminal to refresh plots"
        echo "---"
        tail -f "$LOG_FILE"
        ;;
esac
