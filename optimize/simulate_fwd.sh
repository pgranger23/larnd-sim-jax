#!/bin/bash
#SBATCH --partition=ampere

##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
#SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=larndsim
#SBATCH --output=logs/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=5:00:00
#SBATCH --array=1

#BASE DECLARATIONS
SAMPLING_STEP=0.1 # cm

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

# INPUT_FILE=/sdf/group/neutrino/cyifan/muon-sim/fake_data_S1/edepsim-output.h5
#INPUT_FILE=prepared_data/input_${SLURM_ARRAY_TASK_ID}.h5
#INPUT_FILE=../Data_selection/event122.h5

# dx 1mm

INPUT_FILE=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1_seg_0.h5
#OUTPUT_FILE=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-larndsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1_seg_0.h5

# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} python3 -m optimize.simulate \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.simulate \
    --input_file ${INPUT_FILE} \
    --output_file ${OUTPUT_FILE} \
    --number_pix_neighbors 4 \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --signal_length 400 \
    --mode 'lut' \
    --lut_file src/larndsim/detector_properties/response_44_v2a_full_tick.npz \
    --save_wfs \
    #--noise \
    #--n_events 2 \
    # --chop \
    # --noise \
    #--seed 9 \
    #--out_np \
    #--mc_diff
    #--electron_sampling_resolution 0.005 \

"
