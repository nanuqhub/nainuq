# Training



## Launching training

The training of the emulators are launched using SLURM script `nanuk.slurm` on Jean-Zay. The script can be updated to fit any computer. We describe inthe following the different parameters. 

```
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
```

## Hyperparameters grid search

The grid search to find the best parameters for $\lambda_{\mathrm{bias}}$, $\lambda_{\mathrm{TV}}$ and $\lambda_{\mathrm{PINNs}}$ is launched from the `job_array.sh` script.

```
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
```