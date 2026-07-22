#!/bin/bash
#OAR -n nextsim_emulator
#OAR -l /nodes=1/core=10/gpu=1,walltime=0:30:00
#OAR -p gpumodel='V100'
#OAR --stdout log/%jobid%.out
#OAR --stderr log/%jobid%.err
#OAR --project pr-sasip

# Set environment
cd /bettik/PROJECTS/pr-sasip/ducharlo/nanuk/
source /applis/environments/cuda_env.sh bigfoot 10.2
source /applis/environments/conda.sh
conda activate nextsim_surrogate

# Test parameters
export SAVE_DIR="results/experiment_nanuk1_1h_all_small_no_mask_lambda_no_ocean_new_archi/"
export DATA_PATH="/bettik/PROJECTS/pr-sasip/ducharlo/nanuk/nanuk1_1h_wo_ocean/"
export CHECKPOINT_NAME="last.ckpt"
export N_CYCLE=1
export K=8640
export OCEAN=0
export TIMESTEP=1
export FREQUENCY=12
export SAVE_PRED=True
export NOISE=0
export NOISE_INIT=False

export PCONV_USE=True
export NN_size=32
# Sea ice variables
export SEA_ICE_VARIABLES="sit sic siu siv snt"
export POST_PROCESSING=1

#Ocean variables
export OCEAN_VARIABLES=0
export OCEAN_UNDER=0
export use_ocean_as_forcings=0


# Run inference
python inference/test.py \
    --save_dir $SAVE_DIR \
    --data_path $DATA_PATH \
    --NN_size $NN_size \
    --frequency $FREQUENCY \
    --pconv_use $PCONV_USE \
    --post_processing $POST_PROCESSING \
    --checkpoint_name $CHECKPOINT_NAME \
    --n_cycle $N_CYCLE \
    --ocean_variables $OCEAN_VARIABLES\
    --ocean_under $OCEAN_UNDER\
    --use_ocean_as_forcings $use_ocean_as_forcings\
    --sea_ice_variables $SEA_ICE_VARIABLES \
    --ocean $OCEAN \
    --k $K \
    --timestep $TIMESTEP \
    --save_pred $SAVE_PRED \
    --noise $NOISE \
    --noise_init $NOISE_INIT
