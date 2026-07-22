#!/bin/bash

#SBATCH -A wbh@v100
#SBATCH --job-name="train nanuk" # Job name
#SBATCH --ntasks=1 # nbr of tasks (= nbr of GPU)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # nbr of GPU per node
#SBATCH --partition=gpu_p2
#SBATCH --cpus-per-task=12 # nbr of cores per task
#SBATCH --hint=nomultithread # physical core
#SBATCH --time=20:00:00 # Max exec time
#SBATCH --output="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%j.out" # out file name
#SBATCH --error="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%j.err" # error file name

module purge
# Set environment
cd /linkhome/rech/genrea01/ucm13rr/nanuk1
module load miniforge/24.9.0
conda activate nextsim_surrogate


# Training parameters
export BATCH_SIZE=128
export NUM_EPOCHS=500
export LEARNING_RATE=5e-5
export WEIGHT_DECAY=1e-6
export LAMBDA_BIAS=0
export LAMBDA_PINN=0.1
export LAMBDA_TV=0
export SAVE_DIR="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/results/2026/run_UNet_nanuk1_24h_reduced/"
export DATA_PATH="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/dataset/nanuk1_24h_reduced/"
export PCONV_USE=0

# Model parameters
export MODEL_ARCHITECTURE="unet"
export BASE_FEATURES=32
export KERNEL_SIZE=3

# Sea ice variables
export SEA_ICE_VARIABLES="sit sic siu siv snt"

#Ocean variables
export OCEAN_VARIABLES=0
export OCEAN_UNDER=0
export use_ocean_as_forcings=1

# Run training
python -u src/train_emulator.py \
    --model_architecture $MODEL_ARCHITECTURE\
    --ocean_variables $OCEAN_VARIABLES\
    --ocean_under $OCEAN_UNDER\
    --pconv_use $PCONV_USE\
    --use_ocean_as_forcings $use_ocean_as_forcings\
    --sea_ice_variables $SEA_ICE_VARIABLES \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lambda_bias $LAMBDA_BIAS \
    --lambda_TV $LAMBDA_TV \
    --lambda_PINN $LAMBDA_PINN \
    --save_dir $SAVE_DIR \
    --data_path $DATA_PATH \
    --base_features $BASE_FEATURES \
    --kernel_size $KERNEL_SIZE
