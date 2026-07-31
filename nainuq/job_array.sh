#!/bin/bash
#SBATCH -A wbh@v100
#SBATCH --job-name="train_nanuk_grid_array" # Job name
#SBATCH --array=0-17 # Number of tasks = total combinations (3x3x3=27)
#SBATCH --ntasks=1 # nbr of tasks (= nbr of GPU)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # nbr of GPU per node
#SBATCH --partition=gpu_p2
#SBATCH --cpus-per-task=3 # nbr of cores per task
#SBATCH --hint=nomultithread # physical core
#SBATCH --time=20:00:00 # Max exec time
#SBATCH --output="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%A_%a.out" # out file name
#SBATCH --error="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%A_%a.err" # error file name

module purge
# Set environment
cd /linkhome/rech/genrea01/ucm13rr/nanuk1
module load miniforge/24.9.0
conda activate nextsim_surrogate

# Fixed parameters
export BATCH_SIZE=128
export NUM_EPOCHS=500
export LEARNING_RATE=5e-5
export WEIGHT_DECAY=1e-6
export BASE_DIR="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/results/2026/grid_1h_new_ts_2/"
export DATA_PATH="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/dataset/nanuk1_1h_with_ocean_under/"

export PCONV_USE=0

# Model parameters
export MODEL_ARCHITECTURE="unet"
export BASE_FEATURES=32
export KERNEL_SIZE=3

# Sea ice variables
export SEA_ICE_VARIABLES="sit sic siu siv snt"

# Ocean variables
export OCEAN_VARIABLES=0
export OCEAN_UNDER=0
export use_ocean_as_forcings=1

# Define the grid of hyperparameters
lambda_bias_values=(0 1 5)
lambda_TV_values=(0.05 0.07)
lambda_PINN_values=(1 5 10)

# Calculate the indices for this task
idx=$SLURM_ARRAY_TASK_ID
n_lambda_bias=${#lambda_bias_values[@]}
n_lambda_TV=${#lambda_TV_values[@]}
n_lambda_PINN=${#lambda_PINN_values[@]}

# Map the array index to the combination
lambda_bias=${lambda_bias_values[$((idx / (n_lambda_TV * n_lambda_PINN) ))]}
lambda_TV=${lambda_TV_values[$(( (idx / n_lambda_PINN) % n_lambda_TV ))]}
lambda_PINN=${lambda_PINN_values[$((idx % n_lambda_PINN))]}

# Create a unique SAVE_DIR for this combination
export SAVE_DIR="${BASE_DIR}lambda_bias_${lambda_bias}_lambda_TV_${lambda_TV}_lambda_PINN_${lambda_PINN}/"
mkdir -p "$SAVE_DIR"

# Run training
python -u src/train_emulator.py \
    --model_architecture $MODEL_ARCHITECTURE \
    --ocean_variables $OCEAN_VARIABLES \
    --ocean_under $OCEAN_UNDER \
    --pconv_use $PCONV_USE \
    --use_ocean_as_forcings $use_ocean_as_forcings \
    --sea_ice_variables $SEA_ICE_VARIABLES \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --lambda_bias $lambda_bias \
    --lambda_TV $lambda_TV \
    --lambda_PINN $lambda_PINN \
    --save_dir $SAVE_DIR \
    --data_path $DATA_PATH

