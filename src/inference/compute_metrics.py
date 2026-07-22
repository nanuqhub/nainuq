#!/usr/bin/env python3

import xarray as xr

import argparse
import numpy as np
from tqdm import trange
from skimage.measure import block_reduce
import os
import sys
import json

# Add the parent directory to the path so we can import from layers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
            'in_channels': 17,
            'out_channels': 5,
            'base_features': 32,
            'lr': 1e-4,
            'weight_decay': 1e-3,
            'lambda_': 100
        }

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
    parser = argparse.ArgumentParser(description='Test UNet model for sea ice prediction')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--n_cycle', type=int, required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--timestep', type=int, required=True)
    parser.add_argument('--noise', type=float, default=0)
    parser.add_argument('--ocean', type=str2bool, default=True)
    parser.add_argument('--noise_init', type=bool, default=True)
    parser.add_argument(
        "--sea_ice_variables",
        nargs="+",
        default=["sit", "sic","siu", "siv", "snt"],
        help="List of sea ice names"
    )
    parser.add_argument('--post_processing', type=str2bool, default=True)
    parser.add_argument('--metrics', nargs="+", default=['RMSE', 'bias', 'IIEE', 'PSD'])
    return parser.parse_args()


def rmse(x,y):
    return np.sqrt(np.mean((x - y)**2, axis = (3, 4)))

def bias(x,y):
    return np.mean((x - y), axis = (3, 4))

def iiee(truth, pred, threshold=0.15):
    
    area = np.load("area_nanuk1.npy")
    # Boolean masks
    ice_pred = pred > threshold
    ice_truth = truth > threshold

    # Overestimate: model ice where no observed ice
    over = (ice_pred & ~ice_truth) * area

    # Underestimate: observed ice where no model ice
    under = (ice_truth & ~ice_pred) * area

    # Integrated Ice Edge Error
    IIEE_value = np.sum(over) + np.sum(under)
    return IIEE_value
def main():
    #Size of image
    N_x = 128

    #Read arguments
    args = parse_args()

    #Number of variables
    N_var = len(args.sea_ice_variables)

    #Open mask for valid pixels in full Arctic
    mask = np.load("mask2_nanuk1.npy")

    #Open mask for Central Arctic
    mask_CA = np.load('mask_reduced_CA.npy')

    #Open mask for MIZ
    mask_MIZ = np.load("mask_reduced_MIZ.npy")

    #Compute grid cells ratio for Central Arctic
    N_CA = np.sum(mask_CA)
    ratio_CA = 128*128/N_CA

    #Compute grid cells ratio for MIZ
    N_MIZ = np.sum(mask_MIZ)
    ratio_MIZ = 128*128/N_MIZ

    #Compute grid cells ratio for full Arctic
    N_full = np.sum(mask)
    ratio_full = 128*128/N_full

    #Path directory
    path_dir = os.path.join(args.save_dir, 'test_results/')
    os.makedirs(path_dir, exist_ok=True)

    #Initialize forecast and truth loading
    truths = np.zeros((args.n_cycle, args.k, N_var, N_x, N_x))
    preds = np.zeros((args.n_cycle, args.k, N_var, N_x, N_x))

    print("Loading forecast fields and truth fields")
    #Load each sample in array
    for i in trange(args.n_cycle):
        #Open forecast field
        truths[i] = np.load(args.save_dir + f'post_process_{args.post_processing}_cycle_{args.n_cycle}_truth_{i}.npy')

        #Open truth field
        preds[i] = np.load(args.save_dir + f'post_process_{args.post_processing}_cycle_{args.n_cycle}_pred_{i}.npy')

    print("Finish loading")
    #Metrics computation 
    N_metrics = len(args.metrics)

    # Save results
    print('Create directory')
    results_dir = os.path.join(args.save_dir, 'metrics/')
    os.makedirs(results_dir, exist_ok=True)

    #Only compute SIC if SIC in sea ice variables
    if 'IIEE' in args.metrics and 'sic' not in args.sea_ice_variables:
        print('No IIEE computation')
        N_metrics-=1

    #Compute RMSE on 3 fields
    #axis 0: RMSE computation on full Arctic
    #axis 1: RMSE computation on Central Arctic
    #axis 2: RMSE computation on MIZ
    if 'RMSE' in args.metrics:
        print("Compute RMSE")
        rmse_res = np.zeros((args.n_cycle, args.k, N_var, 3))
        rmse_res[:,:,:,0] = rmse(truths*mask,preds*mask)*ratio_full
        rmse_res[:,:,:,1] = rmse(truths*mask_CA,preds*mask_CA)*ratio_CA
        rmse_res[:,:,:,2] = rmse(truths*mask_MIZ,preds*mask_MIZ)*ratio_MIZ

        rmse_res_pers = np.zeros((args.n_cycle, args.k, N_var, 3))
        rmse_res_pers[:,:,:,0] = rmse(np.expand_dims(truths[:,0], 1)*mask,truths*mask)*ratio_full
        rmse_res_pers[:,:,:,1] = rmse(np.expand_dims(truths[:,0], 1)*mask_CA,truths*mask_CA)*ratio_CA
        rmse_res_pers[:,:,:,2] = rmse(np.expand_dims(truths[:,0], 1)*mask_MIZ,truths*mask_MIZ)*ratio_MIZ

        print("Save RMSE file")
        np.save(os.path.join(results_dir,f'post_process_{args.post_processing}_cycle_{args.n_cycle}_rmse.npy'), rmse_res)
        np.save(os.path.join(results_dir,f'post_process_{args.post_processing}_cycle_{args.n_cycle}_rmse_persistence.npy'), rmse_res_pers)

    if 'bias' in args.metrics:
        print("Compute bias error")
        bias_res = np.zeros((args.n_cycle, args.k, N_var, 3))
        bias_res[:,:,:,0] = bias(truths*mask,preds*mask)*ratio_full
        bias_res[:,:,:,1] = bias(truths*mask_CA,preds*mask_CA)*ratio_CA
        bias_res[:,:,:,2] = bias(truths*mask_MIZ,preds*mask_MIZ)*ratio_MIZ

        bias_res_pers = np.zeros((args.n_cycle, args.k, N_var, 3))
        bias_res_pers[:,:,:,0] = bias(np.expand_dims(truths[:,0], 1)*mask,truths*mask)*ratio_full
        bias_res_pers[:,:,:,1] = bias(np.expand_dims(truths[:,0], 1)*mask_CA,truths*mask_CA)*ratio_CA
        bias_res_pers[:,:,:,2] = bias(np.expand_dims(truths[:,0], 1)*mask_MIZ,truths*mask_MIZ)*ratio_MIZ

        print("Save bias error file")
        np.save(os.path.join(results_dir,f'post_process_{args.post_processing}_cycle_{args.n_cycle}_bias.npy'), bias_res)
        np.save(os.path.join(results_dir,f'post_process_{args.post_processing}_cycle_{args.n_cycle}_bias_persistence.npy'), bias_res_pers)

    if 'IIEE' in args.metrics and 'sic' in args.sea_ice_variables:
        print("Compute IIEE")
        N_sic = args.sea_ice_variables.index('sic')
        area = np.load('area_nanuk1.npy')

        pred_sic = preds[:,:,N_sic]
        truth_sic = truths[:,:,N_sic]
        iiee_res = np.zeros((args.n_cycle, args.k))
        iiee_res_pers = np.zeros((args.n_cycle, args.k))
        for i in range(args.n_cycle):
            for j in range(args.k):
                iiee_res[i,j] = iiee(truth_sic[i,j]*mask, pred_sic[i,j]*mask)
                iiee_res_pers[i,j] = iiee((np.expand_dims(truth_sic[:,0], 1)*mask)[i,0],truth_sic[i,j]*mask)

        print("Save IIEE file")
        np.save(os.path.join(results_dir,f'post_process_{args.post_processing}_cycle_{args.n_cycle}_iiee.npy'), iiee_res)
        np.save(os.path.join(results_dir,f'post_process_{args.post_processing}_cycle_{args.n_cycle}_iiee_persistence.npy'), iiee_res_pers)

if __name__ == '__main__':
    main()
