import numpy as np
import torch
from tqdm import trange
from torch.utils.data import DataLoader
import sys
import os

# Add the parent directory to the path so we can import from datasets
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.TFRecordDataset import SICDataset  # Direct import

class Test:
    def __init__(self, model, mask, season, k, N_cycle, timestep, save_pred, 
                 path_to_save, path_to_data, noise_init, noise):
        self.model = model
        self.mask = mask
        self.path = path_to_save
        self.timestep = timestep
        self.N_cycle = N_cycle
        self.timestep_output = 2
        self.path_data = path_to_data
        self.noise_init = noise_init
        self.noise = noise
        self.save_pred = save_pred
        self.device = next(model.parameters()).device
        
        # Normalization constants
        self.mean_input = 0.38415005912623124
        self.std_input = 0.7710474897556033
        self.mean_output = -1.6270055141928728e-05
        self.std_output = 0.023030024601018818
        self.N_x = 512
        self.N_y = 512
        self.k = k
        self.season = season

    def normalize_input(self, x):
        return (x - self.mean_input) / self.std_input

    def reverse_normalize_output(self, x):
        return x * self.std_output + self.mean_output 

    def reverse_normalize_input(self, x):
        return x * self.std_input + self.mean_input 

    def add_noise(self, x):
        return x + torch.randn_like(x) * self.noise

    def FS(self, x_truth):
        x_truth = x_truth.cpu().numpy()
        min_truth = np.min(x_truth[0, 0, :, :])  # Changed indexing
        
        result = np.zeros((self.k))
        x_inputs = np.zeros((self.k, 4 * self.timestep + 6, self.N_x, self.N_y))  # Changed order
        x_pred = np.zeros((self.k, 1, self.N_x, self.N_y))  # Changed order
        
        # Add noise to initial condition
        x_truth[0, :self.timestep, :, :] = self.add_noise(torch.from_numpy(x_truth[0, :self.timestep, :, :])).numpy()
        x_inputs[0] = x_truth[0]  # No need to reshape as dimensions match
        x_pred[0] = x_truth[0, self.timestep - 1:self.timestep, :, :]  # Take correct slice
        
        for t in range(1, self.k):
            # Convert to tensor and move to device
            inputs = torch.from_numpy(x_inputs[t - 1]).float().unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                err = self.model(inputs, self.mask)[0]  # Only take first output
                err = err.cpu().numpy()  # Shape is now [C, H, W]
                err = self.reverse_normalize_output(err)
                x_pred[t] = self.reverse_normalize_input(x_pred[t - 1]) + err
                x_pred[t] = self.normalize_input(x_pred[t])
                x_pred[t] = np.clip(x_pred[t], a_min=min_truth, a_max=None)
            
            # Prepare next input
            if self.timestep != 1:
                x_inputs[t] = np.concatenate(
                    (x_inputs[t - 1, 1:self.timestep, :, :],  # Previous inputs
                     x_pred[t],  # Current prediction
                     x_truth[2 * t, self.timestep:, :, :]),  # Future forcings
                    axis=0
                )
            else:
                x_inputs[t] = np.concatenate(
                    (x_pred[t],  # Current prediction
                     x_truth[2 * t, self.timestep:, :, :]),  # Future forcings
                    axis=0
                )
        
        # Calculate metrics (adjust for channel-first format)
        mask_np = self.mask.cpu().numpy().squeeze()  # Remove extra dimensions
        x_pred_squeezed = x_pred[:, 0, :, :]  # Remove channel dimension for calculations
        x_truth_target = x_truth[0:2*self.k:2, self.timestep-1, :, :]
        
        # Calculate bias for each timestep (k steps)
        bias = np.zeros(self.k)
        for t in range(self.k):
            bias[t] = (x_pred_squeezed[t] * mask_np - x_truth_target[t] * mask_np).mean()
        
        # Calculate RMSE for each timestep
        result = np.zeros(self.k)
        for t in range(self.k):
            result[t] = np.sqrt(((x_pred_squeezed[t] * mask_np - x_truth_target[t] * mask_np) ** 2).mean())
        
        # Calculate persistence RMSE
        result_pers = np.zeros(self.k)
        for t in range(self.k):
            result_pers[t] = np.sqrt(((x_truth[0, self.timestep-1] * mask_np - x_truth_target[t] * mask_np) ** 2).mean())
        
        return result, result_pers, bias, x_pred

    def test_model(self):
        # Setup validation dataset
        val_files = [f"{self.path_data}val.tfrecords.{str(i).zfill(3)}" for i in range(24)]
        val_dataset = SICDataset(val_files, self.timestep)  # Use direct import
        val_loader = DataLoader(val_dataset, batch_size=1440, shuffle=False)
        
        fs = np.zeros((self.N_cycle, self.k))
        bias = np.zeros((self.N_cycle, self.k))
        fs_pers = np.zeros((self.N_cycle, self.k))
        x_pred = np.zeros((self.N_cycle, self.k, 512, 512, 1))  # Updated dimensions
        truth = np.zeros((self.N_cycle, self.k, 512, 512, 1))   # Updated dimensions
        
        # Get validation data
        x, y = next(iter(val_loader))
        x = x.to(self.device)
        
        self.model.eval()
        for i in trange(self.N_cycle):
            fs_res, fs_pers_res, bias_res, xpred = self.FS(x[i:i+2*self.k])
            bias[i] = bias_res
            fs[i] = fs_res
            fs_pers[i] = fs_pers_res
            
            if self.save_pred:
                tru = x[i:i+2*self.k:2,:,:,0].cpu().numpy()
                np.save(f"{self.path}clip_cycle_{self.N_cycle}_truth_{i}.npy", tru)
                np.save(f"{self.path}clip_cycle_{self.N_cycle}_pred_{i}.npy", xpred)
        
        return fs, fs_pers, bias