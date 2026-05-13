#!/bin/bash

#SBATCH --partition=ampere

##SBATCH --account=mli:cider-ml
##SBATCH --account=neutrino:dune-ml
#SBATCH --account=neutrino:cider-nu
##SBATCH --account=neutrino:ml-dev

#SBATCH --job-name=diffsim_dE
#SBATCH --output=logs/fit_noise/job-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=4:00:00
#SBATCH --array=1

#BASE DECLARATIONS

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

TARGET_SEED=$SLURM_ARRAY_TASK_ID
PARAMS=optimize/scripts/param_list.yaml
BATCH_SIZE=200
MAX_NBATCH=500
ITERATIONS=10000
MAX_CLIP_NORM_VAL=1
DATA_SEED=1
LOSS=llhd #sobolev_adc #wasserstein_1d #nll #llhd #mmd_adc
SEED_STRATEGY=different #random #different_epoch
SAMPLING_STEP=0.01 # cm
N_NEIGH=4
MODE="lut"  #"parametrized"
LR_SCHEDULER=warmup_exponential_decay_schedule
SIGNAL_LENGTH=400
NORM=sigmoid #divide
DE_LR=5e-5 #1e-4
# SIGMA=500 #250 #500

# ProbabilisticLossStrategy defaults (optimize/strategies.py lines 312-322)
SOB_EPS=1e-10 #1e-6
SOB_W3D_GRAD=0.01
SOB_GAUSS_RADIUS_CM=0.3
SOB_GAUSS_SIGMA_CM=0.1
SOB_POOL_MED_X=60
SOB_POOL_MED_Z=30
SOB_POOL_GLB_X=20
SOB_POOL_GLB_Z=10
SOB_NZ_LOCAL=1999

SOB_POOL_LAYER_BALANCE=weights #running
SOB_POOL_WEIGHT_LOCAL=0.05
SOB_POOL_WEIGHT_MEDIUM=1.
SOB_POOL_WEIGHT_GLOBAL=50.

COORD="_x"

# dx = 0.1mm
INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1.h5
# INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1_shifted${COORD}_big.h5
INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1_shifted${COORD}.h5

# dx = 1mm, the rest are 0.1mm
#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.05cm.h5

# INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1.h5

# INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1.h5
# INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1_shifted${COORD}.h5
# INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1_shifted.h5

# INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.05cm_ntrack_1.h5

#INPUT_FILE_TGT=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm_ntrack_1_seg_0.h5
#INPUT_FILE_SIM=/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_2043327/job_25174367_0000/output_25174367_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_reco_dE_range_0.05cm_ntrack_1_seg_0.h5

SIF_FILE=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
UUID=$(uuidgen)

# LABEL=sobolev_output6
LABEL=stopp_fit_pos_shifted${COORD}_sm_trk1_dx0.1mm_n_neigh${N_NEIGH}_${MODE}_signalL${SIGNAL_LENGTH}_${LR_SCHEDULER}_bt${BATCH_SIZE}_nbtach${MAX_NBATCH}_dtsd${DATA_SEED}_Adam_${LOSS}_pos_lr${DE_LR}_eps${SOB_EPS}_w3dg${SOB_W3D_GRAD}_gssg_${SOB_GAUSS_SIGMA_CM}cm_gsr_${SOB_GAUSS_RADIUS_CM}cm_medx${SOB_POOL_MED_X}_glbx${SOB_POOL_GLB_X}_${NORM}

nvidia-smi


# export JAX_LOG_COMPILES=1


# apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node --capture-range-end=stop python3 -m optimize.example_run \
apptainer exec --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SIF_FILE} /bin/bash -c "
pip3 install .; \
python3 -m optimize.example_run \
    --data_sz -1 \
    --max_nbatch ${MAX_NBATCH} \
    --params ${PARAMS} \
    --input_file_sim ${INPUT_FILE_SIM} \
    --input_file_tgt ${INPUT_FILE_TGT} \
    --non_deterministic \
    --fit_type chain \
    --track_len_sel 2 \
    --max_abs_costheta_sel 0.966 \
    --min_abs_segz_sel 15. \
    --data_seed ${DATA_SEED} \
    --out_label ${LABEL} \
    --test_name fit_noise \
    --seed ${TARGET_SEED} \
    --optimizer_fn Adam \
    --iterations ${ITERATIONS} \
    --max_batch_len ${BATCH_SIZE} \
    --track_z_bound 28 \
    --max_clip_norm_val ${MAX_CLIP_NORM_VAL} \
    --electron_sampling_resolution ${SAMPLING_STEP} \
    --number_pix_neighbors ${N_NEIGH} \
    --signal_length ${SIGNAL_LENGTH} \
    --mode ${MODE} \
    --lut_file ../Data_selection/response_44_v2a_full_tick.npz \
    --loss_fn ${LOSS} \
    --sim_seed_strategy ${SEED_STRATEGY} \
    --clip_from_range \
    --lr_scheduler ${LR_SCHEDULER} \
    --lr_kw '{\"decay_rate\" : 0.999, \"init_value\" : 0, \"warmup_steps\": 1000}' \
    --target_gaussian_3d_radius_cm ${SOB_GAUSS_RADIUS_CM} \
    --target_gaussian_3d_sigma_cm ${SOB_GAUSS_SIGMA_CM} \
    --nz_local ${SOB_NZ_LOCAL} \
    --eps ${SOB_EPS} \
    --w_sobolev_3d_grad ${SOB_W3D_GRAD} \
    --sobolev_pool_nbin_x_medium ${SOB_POOL_MED_X} \
    --sobolev_pool_nbin_z_medium ${SOB_POOL_MED_Z} \
    --sobolev_pool_nbin_x_global ${SOB_POOL_GLB_X} \
    --sobolev_pool_nbin_z_global ${SOB_POOL_GLB_Z} \
    --sobolev_pool_layer_balance ${SOB_POOL_LAYER_BALANCE} \
    --sobolev_pool_weight_local ${SOB_POOL_WEIGHT_LOCAL} \
    --sobolev_pool_weight_medium ${SOB_POOL_WEIGHT_MEDIUM} \
    --sobolev_pool_weight_global ${SOB_POOL_WEIGHT_GLOBAL} \
    --shuffle_bt \
    --normalization_scheme ${NORM} \
    --fit_track_shift_xyz \
    --track_shift_mode 'segment-only'\
    --track_shift_optimizer Adam \
    --no_chop \
    --no_pad \
    --track_shift_lr ${DE_LR} \
    --set_target_vals Ab 0.8 kb 0.0486 eField 0.50 tran_diff 8.8e-6 long_diff 4.0e-6 lifetime 2200 \
    --probabilistic_sim \
    # --mmd_sigma ${SIGMA} \
    #--segment_reg_l2 ${REG_L2} \
    #--segment_reg_track_total ${REG_TOT} \
    #--segment_reg_smooth ${REG_SMOOTH} \
    #--debug_nans \
    #--no_chop \
    #--profile \
    #--debug_nans \
    #--no-noise-guess \
    #--no-noise \
    #--print_input
"
