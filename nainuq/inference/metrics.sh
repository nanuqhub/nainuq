#!/bin/bash
#OAR -n nextsim_emulator
#OAR -l core=64,walltime=1:10:00
#OAR --stdout log/%jobid%.out
#OAR --stderr log/%jobid%.err
#OAR --project pr-sasip

# Set environment
cd /bettik/PROJECTS/pr-sasip/ducharlo/nanuk/
source /applis/environments/conda.sh
conda activate nextsim_surrogate

# Test parameters
export SAVE_DIR="results/experiment_nanuk1_1h_all_small_lr2/"
export DATA_PATH="/bettik/PROJECTS/pr-sasip/ducharlo/nanuk/nanuk1_1h/"
export CHECKPOINT_NAME="last.ckpt"
export N_CYCLE=690
export K=360
export OCEAN=0
export TIMESTEP=1
export SAVE_PRED=True
export NOISE=0
export NOISE_INIT=False

export NN_size=32
# Sea ice variables
export SEA_ICE_VARIABLES="sit sic siu siv snt" 
export POST_PROCESSING=1

#Metrics
export METRICS="RMSE bias IIEE"
#Ocean variables
export OCEAN_VARIABLES=0
export OCEAN_UNDER=0
python inference/compute_metrics.py \
    --save_dir $SAVE_DIR \
    --data_path $DATA_PATH \
    --post_processing $POST_PROCESSING \
    --n_cycle $N_CYCLE \
    --metrics $METRICS \
    --sea_ice_variables $SEA_ICE_VARIABLES \
    --ocean $OCEAN \
    --k $K \
    --timestep $TIMESTEP \
    --noise $NOISE \
    --noise_init $NOISE_INIT
