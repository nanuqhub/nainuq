#!/bin/bash

#OAR -n nextsim_emulator
#OAR -l /nodes=1/gpu=1,walltime=1:00:00
#OAR -p gpumodel='A100'
#OAR --stdout log/%jobid%.out
#OAR --stderr log/%jobid%.err
#OAR --project pr-sasip

# Set environment
cd /bettik/PROJECTS/pr-sasip/ducharlo/LR/pytorch/
source /applis/environments/cuda_env.sh bigfoot 10.2
source /applis/environments/conda.sh
conda activate nextsim_surrogate

# Test parameters
export SAVE_DIR="results/experiment2/"
export DATA_PATH="../../../postprocessed/"
export CHECKPOINT_NAME="last-v1.ckpt"
export N_CYCLE=1320
export K=61
export TIMESTEP=1
export SAVE_PRED=False
export NOISE=0
export NOISE_INIT=False

# Run inference
python inference/test.py \
    --save_dir $SAVE_DIR \
    --data_path $DATA_PATH \
    --checkpoint_name $CHECKPOINT_NAME \
    --n_cycle $N_CYCLE \
    --k $K \
    --timestep $TIMESTEP \
    --save_pred $SAVE_PRED \
    --noise $NOISE \
    --noise_init $NOISE_INIT
