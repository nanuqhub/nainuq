#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from tqdm import trange
from skimage.measure import block_reduce
import os
import sys
import json

# Add the parent directory to the path so we can import from layers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.UNet import UNetModel
from inference.test_utils import Test

def load_hyperparameters(save_dir):
    """Load hyperparameters from the experiment directory"""
    config_path = os.path.join(save_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        # Default parameters if config doesn't exist
        return {
            'in_channels': 10,
            'out_channels': 1,
            'base_features': 32,
            'lr': 1e-4,
            'weight_decay': 1e-3,
            'lambda_': 100
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Test UNet model for sea ice prediction')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint_name', type=str, required=True)
    parser.add_argument('--n_cycle', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--timestep', type=int, required=True)
    parser.add_argument('--save_pred', type=bool, default=True)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--noise_init', type=bool, default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and process mask
    os.chdir('/bettik/PROJECTS/pr-sasip/ducharlo/LR/pytorch/')
    mask = np.load('mask.npy')
    print(np.shape(mask))
    mask = 1 - mask
    # Reshape mask to match expected format [B, 1, H, W]
    mask = torch.from_numpy(mask).float()
    mask = mask.permute(2, 0, 1).unsqueeze(0)  # Add batch and channel dimensions
    mask = mask.to(device)

    # Load hyperparameters from experiment directory
    config = load_hyperparameters(args.save_dir)

    # Initialize model with loaded hyperparameters
    model = UNetModel(
        in_channels=config['in_channels'],
        out_channels=config['out_channels'],
        base_features=config['base_features'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        lambda_=config['lambda_'],
        save_dir=args.save_dir
    ).to(device)

    # Load weights
    checkpoint_path = os.path.join(args.save_dir, 'checkpoints', args.checkpoint_name)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Initialize test class
    test = Test(
        model=model,
        season='all',
        mask=mask,
        k=args.k,
        N_cycle=args.n_cycle,
        save_pred=args.save_pred,
        timestep=args.timestep,
        path_to_save=args.save_dir,
        path_to_data=args.data_path,
        noise=args.noise,
        noise_init=args.noise_init
    )

    # Run test
    fs, fs_pers, bias = test.test_model()
    print(fs)
    print(fs_pers)
    # Save results
    results_dir = os.path.join(args.save_dir, 'test_results')
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, f'clip_cycle_{args.n_cycle}_bias_mean.npy'), bias)
    np.save(os.path.join(results_dir, f'clip_cycle_{args.n_cycle}_fs_mean.npy'), fs)
    np.save(os.path.join(results_dir, f'clip_cycle_{args.n_cycle}_fs_mean_pers.npy'), fs_pers)

if __name__ == '__main__':
    main()
