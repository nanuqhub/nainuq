import torch.nn.functional as F
import logging
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any


class DoubleConv(nn.Module):
    """
    Double convolution block with optional residual connection.
    Conv -> Mish -> Conv -> Mish + Residual
    """
    def __init__(self, in_channels, out_channels, use_mish=True, dropout=0.1):
        super().__init__()
        self.use_residual = (in_channels == out_channels)
        activation = nn.Mish() if use_mish else nn.ReLU(inplace=True)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            activation,
            nn.Dropout2d(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            activation,
            nn.Dropout2d(p=dropout)
        )

        # If in/out channels differ, use a 1×1 conv to match for residual
        if not self.use_residual:
            self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x):
        out = self.double_conv(x)
        res = self.res_conv(x)
        return out + res



class UNetModel(pl.LightningModule):

    def __init__(self,
            data_path,
            in_channels: int = 17,
            out_channels: int = 10,
            base_features: int = 64,
            lr: float = 1e-4,
            weight_decay: float = 1e-3,
            lambda_bias: float = 0.1,
            lambda_PINN : float = 0,
            lambda_TV : float = 1,
            save_dir: str = 'results'):
            
        super().__init__()
        
        
        # Create save directory structure
        self.save_dir = Path(save_dir)
        self.checkpoint_dir = self.save_dir / 'checkpoints'
        self.metrics_dir = self.save_dir / 'metrics'
        self.config_dir = self.save_dir / 'config'
        
        # Create directories if they don't exist
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        # Save hyperparameters
        self.save_hyperparameters()
        self.save_config()
        
        self.path_data =  data_path
        #Save mask

        self.N = 5

        self.mask = np.load('/linkhome/rech/genrea01/ucm13rr/nanuq1/src/mask2_nanuk1.npy')

        self.mask = torch.from_numpy(self.mask).float()
        self.mask = torch.reshape(self.mask, [1, 1, 128, 128])
        # Move mask to GPU if available
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'epoch': [],
            'train_loss': [], 'train_mse': [], 'train_bias': [],
            'val_loss': [], 'val_mse': [], 'val_bias': [],
            'train_bias_snt': [], 'val_bias_snt': [],
            'train_bias_sic': [], 'val_bias_sic': [],
            'train_bias_sit': [], 'val_bias_sit': [],
            'train_drift_loss': [], 'val_drift_loss': [],
            'train_A_loss': [], 'val_A_loss': [],
            'train_H_loss': [], 'val_H_loss': [],
            'train_tv_loss': [], 'val_tv_loss': [],
            'learning_rate': []
        }
        
        # Encoder features progression
        self.features = [base_features, base_features * 2, base_features * 4, base_features*8]  # [32, 64, 128]
        
        self.mean_input = np.load(f'{self.path_data}sea_ice_mean_input.npy')
        self.std_input = np.load(f'{self.path_data}sea_ice_std_input.npy')
        self.mean_output = np.load(f'{self.path_data}sea_ice_mean_output.npy')
        self.std_output = np.load(f'{self.path_data}sea_ice_std_output.npy')


        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_bias = lambda_bias
        self.lambda_PINN = lambda_PINN        
        self.lambda_TV = lambda_TV
        # Initialize loss parameters

        self.loss = nn.MSELoss()
        
        self.init_conv = DoubleConv(in_channels, self.features[0])                # base -> f0
        self.pool1 = nn.MaxPool2d(2)
        self.conv1  = DoubleConv(self.features[0], self.features[1])             # f0 -> f1
        self.pool2 = nn.MaxPool2d(2)
        self.conv2  = DoubleConv(self.features[1], self.features[2])             # f1 -> f2

        # Bottleneck (down)
        self.conv3 = DoubleConv(self.features[2], self.features[3])              # f2 -> f3

        # Bottleneck unwind (symmetric)
        self.conv4 = DoubleConv(self.features[3], self.features[2])              # f3 -> f2

        # Decoder (upsample + concat)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = DoubleConv(self.features[2] + self.features[1], self.features[1], dropout = 0.1)            # (upsampled f2 + skip f2) -> f1

        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv6 = DoubleConv(self.features[1] + self.features[0], self.features[0], dropout = 0.1)            # (upsampled f1 + skip f1) -> f0

        self.conv7 = DoubleConv(self.features[0], self.features[0], dropout = 0.2)
        self.final_conv = nn.Conv2d(self.features[0], out_channels, kernel_size=1)


    def normalize_input(self, x):
        std = self.std_input.reshape(1, self.N, 1, 1)
        mean = self.mean_input.reshape(1, self.N, 1, 1)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()
        return (x - mean) / std

    def reverse_normalize_output(self, x):
        std = self.std_output.reshape(1, self.N, 1, 1)
        mean = self.mean_output.reshape(1, self.N, 1, 1)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()
        return x * std + mean

    def reverse_normalize_input(self, x):
        std = self.std_input.reshape(1, self.N, 1, 1)
        mean = self.mean_input.reshape(1, self.N, 1, 1)
        mean = torch.from_numpy(mean).float()
        std = torch.from_numpy(std).float()
        return x * std + mean

    def save_config(self):
        """Save model configuration to JSON"""
        config = {
            'in_channels': self.hparams.in_channels,
            'out_channels': self.hparams.out_channels,
            'base_features': self.hparams.base_features,
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
            'lambda_bias': self.hparams.lambda_bias,
            'save_dir': str(self.save_dir)
        }
        
        config_path = self.config_dir / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

    def save_metrics(self):
        """Save training metrics to CSV"""
        # Ensure all lists have the same length by padding with None if necessary
        max_length = max(len(v) for v in self.metrics_history.values())
        
        # Pad all lists to the same length
        padded_metrics = {}
        for key, values in self.metrics_history.items():
            padded_metrics[key] = values + [None] * (max_length - len(values))
        
        df = pd.DataFrame(padded_metrics)
        
        # Remove rows where all metric values are None
        df = df.dropna(how='all', subset=['train_loss', 'val_loss'])
        
        metrics_path = self.metrics_dir / 'training_metrics.csv'
        df.to_csv(metrics_path, index=False)

    
    def forward(self, x, sic_threshold: float = -0.50738):

        #sic = x[:, 1, :, :]  # select the second channel
        #mask_sic = (sic > sic_threshold).float()  # 1 where SIC > threshold, else 0

        # Expand to all channels for broadcasting
        #mask = mask_sic.unsqueeze(1) # (B, 1, H, W)
        #mask = mask.expand(-1, 5, -1, -1)
        #x0 = self.init_conv(torch.cat([x[:,:-5], x[:,-5:]*mask], dim = 1))

        x0 = self.init_conv(x)          # (B, f0, H, W)
        x1 = self.pool1(x0)
        x1 = self.conv1(x1)             # (B, f1, H/2, W/2)
        
        x2 = self.pool2(x1)
        x2 = self.conv2(x2)             # (B, f2, H/4, W/4)

        x3 = self.conv3(x2)             # (B, f3, H/4, W/4)
        x4 = self.conv4(x3)             # (B, f2, H/4, W/4)

        x5 = self.up5(x4)               # (B, f2, H/2, W/2)  -- may require align
        # ensure match with x1
        x5 = torch.cat([x5, x1], dim=1) # (B, 2*f2, H/2, W/2)
        x5 = self.conv5(x5)             # (B, f1, H/2, W/2)

        x6 = self.up6(x5)               # (B, f1, H, W)
        x6 = torch.cat([x6, x0], dim=1) # (B, 2*f1, H, W)  -- careful: x0 has f0 channels; if shapes differ adapt

        x6 = self.conv6(x6)
        x7 = self.conv7(x6)
        out = self.final_conv(x7)*self.mask
        return out
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer

    def summary(self):
        """Print a summary of the UNet architecture"""
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nTotal number of parameters: {total_params:,}")
    def L_sat(self,x_prev, x_pred, threshold=0.01):
        """
        Velocity saturation loss, applied only where updated concentration < threshold.

        Args:
            x_prev (torch.Tensor): previous state [B, C, H, W]
            x_pred (torch.Tensor): predicted tendencies [B, C, H, W]
            threshold (float): concentration threshold

        Returns:
            torch.Tensor: scalar saturation loss
        """
        # Reconstruct updated state
        x_new = self.reverse_normalize_input(x_prev) + self.reverse_normalize_output(x_pred)

        A_new = x_new[:, 1, :, :]  # concentration
        u_new = x_new[:, 2, :, :]
        v_new = x_new[:, 3, :, :]
        H_new = x_new[:, 0, :, :]
        # Compute velocity magnitude
        vel2 = u_new**2 + v_new**2

        # Apply mask for low concentration
        mask = (A_new < threshold).float()

        # Apply only where A < threshold
        loss_sat = (vel2 * mask).mean()
        # Penalize negative thickness
        loss_H = F.relu(-H_new).mean()

        # Penalize concentration outside [0, 1]
        loss_A = (F.relu(A_new - 1) + F.relu(-A_new)).mean()

        return 1*loss_sat + 1*loss_H + 1*loss_A, loss_sat, loss_H, loss_A

    def compute_loss(self, y_pred, y_true, x):
        """
        Compute custom loss with MSE and bias constraint
        """
        
        # Compute MSE
        mse_loss = self.loss(y_pred*self.mask, y_true)
        tv_loss = torch.mean(torch.abs(y_pred[:, 1, :, :-1] - y_pred[:, 1, :, 1:])) + torch.mean(torch.abs(y_pred[:, 1, :-1, :] - y_pred[:, 1, 1:, :]))
 
        # Compute bias term (average difference across all pixels)
        pred_mean = torch.mean(y_pred[:,0])
        true_mean = torch.mean(y_true[:,0])
        bias_loss_sit = (pred_mean - true_mean) ** 2
        
        pred_mean_sic = torch.mean(y_pred[:,1])
        true_mean_sic = torch.mean(y_true[:,1])
        bias_loss_sic = (pred_mean_sic - true_mean_sic) ** 2

        pred_mean_snt = torch.mean(y_pred[:,4])
        true_mean_snt = torch.mean(y_true[:,4])
        bias_loss_snt = (pred_mean_snt - true_mean_snt) ** 2

        sat_loss, loss_sat, loss_H, loss_A = self.L_sat(x[:,:5].cpu(), y_pred.cpu())
        # Combine losses
        total_loss = mse_loss + self.lambda_bias * (bias_loss_sit + bias_loss_sic + bias_loss_snt)  + self.lambda_PINN*sat_loss + self.lambda_TV*tv_loss
        
        return total_loss, mse_loss, bias_loss_sit, bias_loss_snt, bias_loss_sic, tv_loss, loss_sat, loss_H, loss_A

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        y_hat = self(x)
            
        #print("compute loss")
        loss, mse, bias_sit, bias_snt, bias_sic, tv_loss, loss_sat, loss_H, loss_A = self.compute_loss(y_hat, y, x)
        
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)
        self.log('train_bias_sit', bias_sit, on_step=False, on_epoch=True)
        self.log('train_bias_snt', bias_snt, on_step=False, on_epoch=True)
        self.log('train_bias_sic', bias_sic, on_step=False, on_epoch=True)
        self.log('train_tv_loss', tv_loss, on_step=False, on_epoch=True)
        self.log('train_drift_loss', loss_sat, on_step=False, on_epoch=True)
        self.log('train_H_loss', loss_H, on_step=False, on_epoch=True)
        self.log('train_A_loss', loss_A, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Compute loss with components
        val_loss, val_mse, val_bias_sit, val_bias_snt, val_bias_sic, val_tv_loss, val_loss_sat, val_loss_H, val_loss_A = self.compute_loss(y_hat, y, x)

        # Log validation metrics
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mse', val_mse, on_step=True, on_epoch=True)
        self.log('val_bias_sit', val_bias_sit, on_step=False, on_epoch=True)
        self.log('val_bias_snt', val_bias_snt, on_step=False, on_epoch=True)
        self.log('val_bias_sic', val_bias_sic, on_step=False, on_epoch=True)
        self.log('val_tv_loss', val_tv_loss, on_step=False, on_epoch=True)
        self.log('val_drift_loss', val_loss_sat, on_step=False, on_epoch=True)
        self.log('val_H_loss', val_loss_H, on_step=False, on_epoch=True)
        self.log('val_A_loss', val_loss_A, on_step=False, on_epoch=True)
        return val_loss

    def on_train_epoch_end(self):
        # Get metrics from callback_metrics
        metrics = self.trainer.callback_metrics
        print(metrics)
        # Update metrics history only if metrics exist
        if 'train_loss_epoch' in metrics:
            self.metrics_history['epoch'].append(self.current_epoch)
            self.metrics_history['train_loss'].append(metrics['train_loss_epoch'].item())
            self.metrics_history['train_mse'].append(metrics['train_mse_epoch'].item())
            self.metrics_history['train_bias_sit'].append(metrics['train_bias_sit'].item())
            self.metrics_history['train_bias_snt'].append(metrics['train_bias_snt'].item())
            self.metrics_history['train_bias_sic'].append(metrics['train_bias_sic'].item())
            self.metrics_history['train_tv_loss'].append(metrics['train_tv_loss'].item())
            self.metrics_history['train_drift_loss'].append(metrics['train_drift_loss'].item())
            self.metrics_history['train_H_loss'].append(metrics['train_H_loss'].item())
            self.metrics_history['train_A_loss'].append(metrics['train_A_loss'].item())
            #self.metrics_history['train_bias'].append(metrics['train_bias_epoch'].item())
            
            # Get learning rate
            opt = self.optimizers()
            if opt is not None:
                lr = opt.param_groups[0]['lr']
                self.metrics_history['learning_rate'].append(lr)
                self.log('learning_rate', lr, on_epoch=True)
            
            # Save metrics after each epoch
            self.save_metrics()

    def on_validation_epoch_end(self):
        # Get metrics from callback_metrics
        metrics = self.trainer.callback_metrics
        print(metrics) 
        # Update metrics history only if metrics exist
        if 'val_loss_epoch' in metrics:
            # Ensure we have the same number of validation metrics as training metrics
            while len(self.metrics_history['val_loss']) < len(self.metrics_history['epoch']):
                self.metrics_history['val_loss'].append(None)
                self.metrics_history['val_mse'].append(None)
                #self.metrics_history['val_bias'].append(None)
            
            # Add current validation metrics
            self.metrics_history['val_loss'].append(metrics['val_loss_epoch'].item())
            self.metrics_history['val_mse'].append(metrics['val_mse_epoch'].item())
            self.metrics_history['val_bias_sit'].append(metrics['val_bias_sit'].item())
            self.metrics_history['val_bias_snt'].append(metrics['val_bias_snt'].item())
            self.metrics_history['val_bias_sic'].append(metrics['val_bias_sic'].item())
            self.metrics_history['val_tv_loss'].append(metrics['val_tv_loss'].item())
            self.metrics_history['val_drift_loss'].append(metrics['val_drift_loss'].item())
            self.metrics_history['val_H_loss'].append(metrics['val_H_loss'].item())
            self.metrics_history['val_A_loss'].append(metrics['val_A_loss'].item())
            #self.metrics_history['val_bias'].append(metrics['val_bias_epoch'].item())
            
            # Save metrics after each epoch
            self.save_metrics()

    def get_checkpoint_callback(self):
        """Create ModelCheckpoint callback with proper save directory"""
        return pl.callbacks.ModelCheckpoint(
            dirpath=self.checkpoint_dir,
            filename='model-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_last=True,
            verbose=True
        )

    def get_early_stopping_callback(self, patience=20):
        """Create EarlyStopping callback"""
        return pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min',
            verbose=True
        )
