#!/bin/bash

#SBATCH -A wbh@v100
#SBATCH --job-name="test nanuk" # Job name
#SBATCH --ntasks=1 # nbr of tasks (= nbr of GPU)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # nbr of GPU per node
#SBATCH --partition=gpu_p2
#SBATCH --cpus-per-task=10 # nbr of cores per task
#SBATCH --hint=nomultithread # physical core
#SBATCH --time=0:40:00 # Max exec time
#SBATCH --output="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%j.out" # out file name
#SBATCH --error="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/log/%j.err" # error file name

module purge
# Set environment
cd /linkhome/rech/genrea01/ucm13rr/nanuq1
#cd /lustre/fswork/projects/rech/wvg/ucm13rr/nanuk/
module load miniforge/24.9.0
conda activate nextsim_surrogate


export SAVE_DIR="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/results/TV/run_UNet_nanuk1_24h_bulk_ocean_avg/"
export DATA_PATH="/lustre/fswork/projects/rech/wbh/ucm13rr/nanuk/dataset/nanuk1_24h_avg/"

# Sea ice variables
export SEA_ICE_VARIABLES="sit sic siu siv snt"

export CHECKPOINT_NAME="last.ckpt"
export N_CYCLE=1
export K=360

export OCEAN=0
export TIMESTEP=24
export FREQUENCY=12
export SAVE_PRED=1
export NOISE=0
export NOISE_INIT=False

export NN_size=32
export POST_PROCESSING=1
#Ocean variables
export OCEAN_VARIABLES=0
export OCEAN_UNDER=0
export use_ocean_as_forcings=1

export use_pconv=0
# Run inference
python src/inference/test_topaz.py \
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
