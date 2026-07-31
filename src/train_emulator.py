#Check if it works

import argparse

import glob
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.TFRecordDataset import Sea_ice_dataset
from layers.full_UNet import UNetModel
from layers.full_UNet_with_PConv import PConv_UNetModel
import torch
import numpy as np
import torch.multiprocessing as mp
import warnings
import logging
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet model for sea ice prediction')
    parser.add_argument('--model_architecture', type=str, default = "unet")
    parser.add_argument(
        "--sea_ice_variables",
        nargs="+",
        default=["sit",'sic',"siu", "siv", "snt"],
        help="List of sea ice names"
    )
    parser.add_argument('--use_ocean_as_forcings', type=str2bool, default=False)
    parser.add_argument('--ocean_under', type=str2bool, default=False)
    parser.add_argument('--ocean_variables', type=str2bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lambda_PINN', type=float, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--lambda_bias', type=float, default=100)
    parser.add_argument('--lambda_TV', type=float, default=1)
    parser.add_argument('--save_dir', type=str, default='results/experiment1')
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--base_features', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--pconv_use', type=str2bool, default=False)
    return parser.parse_args()

def train_model(args):
    print(args)
    # Set random seeds for reproducibility
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')
    print(args.use_ocean_as_forcings) 
    #Get number of sea ice variables
    len_variables = len(args.sea_ice_variables)
    N_ocean = 0
    print(len_variables)
    N_under = 0
    if args.use_ocean_as_forcings==True:
        print('here')
        N_ocean = 5
    else:
        print('no ocean in forcings')
        N_ocean = 0
    if args.ocean_under==False:
        N_under = 0
    #Define input and output size of the emulator
    in_channels = 4 + N_under + N_ocean + len_variables
    out_channels = len_variables
    if args.ocean_variables:
        out_channels +=5
    print(in_channels)
    print(out_channels)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_files = sorted(glob.glob(f"{args.data_path}train/data_20*.tfrecords.*"))
    val_files = sorted(glob.glob(f"{args.data_path}val/*.tfrecords.*"))
    
    # Debug print to check files found
    print("=== Dataset Files ===")
    print(f"Training files found: {len(train_files)}")
    print(f"First training file: {train_files[0] if train_files else 'None'}")
    print(f"Validation files found: {len(val_files)}")
    print(f"First validation file: {val_files[0] if val_files else 'None'}")
    print(f"Data path used: {args.data_path}")
    
    if not train_files or not val_files:
        raise FileNotFoundError(f"No data files found in {args.data_path}")
    
    train_dataset = Sea_ice_dataset(train_files, args.sea_ice_variables, N_ocean, N_under,args.use_ocean_as_forcings)
    val_dataset = Sea_ice_dataset(val_files, args.sea_ice_variables, N_ocean, N_under,args.use_ocean_as_forcings)
    
    # Debug print dataset sizes
    print(f"\n=== Dataset Sizes ===")
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=False,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Get the next batch from the validation loader
    #next_batch = next(iter(val_loader))

    
    # If your dataset returns a tuple of (inputs, targets)
    #inputs, targets = next_batch
    
    # Initialize model
    print("Start training UNet")
    if args.pconv_use:
        model = PConv_UNetModel(
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=args.base_features,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            lambda_=args.lambda_bias,
            save_dir=str(save_dir)
            )
    else:
        model = UNetModel(
            data_path = args.data_path,
            in_channels=in_channels,
            out_channels=out_channels,
            base_features=args.base_features,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            lambda_bias=args.lambda_bias,
            lambda_TV = args.lambda_TV,
            lambda_PINN = args.lambda_PINN,
            save_dir=str(save_dir)
            )
    print(model.summary())
    print(model)
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(save_dir / 'checkpoints'),
        filename='model-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_last=True,
        save_top_k=3
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup trainer
    trainer = pl.Trainer(
        default_root_dir=str(save_dir),
        max_epochs=args.num_epochs,
        num_sanity_val_steps=0,
        accelerator='gpu',
        gradient_clip_val=1.0,        # clip gradients with norm > 1.0
        gradient_clip_algorithm="norm",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        enable_progress_bar=True
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == '__main__':
    # Set start method to spawn
    mp.set_start_method('spawn', force=True)
    
    # Parse arguments and train model
    args = parse_args()
    train_model(args) 
