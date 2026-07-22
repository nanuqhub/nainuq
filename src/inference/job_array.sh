#!/bin/bash
#SBATCH -A wbh@v100
#SBATCH --job-name="test_nanuk_grid_array" # Job name
#SBATCH --array=0-17 # Number of tasks = total combinations (3x3x3=27)
#SBATCH --ntasks=1 # nbr of tasks (= nbr of GPU)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # nbr of GPU per node
#SBATCH --partition=gpu_p2
#SBATCH --cpus-per-task=10 # nbr of cores per task
#SBATCH --hint=nomultithread # physical core
#SBATCH --time=1:20:00 # Max exec time
#SBATCH --output="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%A_%a.out" # out file name
#SBATCH --error="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%A_%a.err" # error file name

module purge
# Set environment
cd /linkhome/rech/genrea01/ucm13rr/nanuq1
module load miniforge/24.9.0
conda activate nextsim_surrogate

# Fixed parameters
export BASE_DIR="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/results/2026/grid_1h_new_ts_2/"
export DATA_PATH="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/dataset/nanuk1_1h_with_ocean_under/"


# Sea ice variables
export SEA_ICE_VARIABLES="sit sic siu siv snt"

export CHECKPOINT_NAME="last.ckpt"
export N_CYCLE=600
export K=260

export OCEAN=0
export TIMESTEP=1
export FREQUENCY=12
export SAVE_PRED=0
export NOISE=0
export NOISE_INIT=False

export NN_size=32
export POST_PROCESSING=1
#Ocean variables
export OCEAN_VARIABLES=0
export OCEAN_UNDER=0
export use_ocean_as_forcings=1

export use_pconv=0
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

python src/inference/test.py \
    --save_dir $SAVE_DIR \
    --data_path $DATA_PATH \
    --NN_size $NN_size \
    --use_pconv $use_pconv \
    --frequency $FREQUENCY \
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

