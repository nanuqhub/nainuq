# System modules
import logging
import os
import json
import pandas as pd
from pathlib import Path
import numpy as np
# External modules
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any
from layers.ModuleBlocks import Encoder, Bottleneck, Decoder

class DebugCallback(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is starting!")
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        print(f"\nStarting batch {batch_idx}")
        try:
            x, y = batch
            print(f"Successfully unpacked batch")
            print(f"Batch shapes - x: {x.shape}, y: {y.shape}")
            print(f"x device: {x.device}, y device: {y.device}")
            print(f"pl_module device: {next(pl_module.parameters()).device}")
            print(f"mask device: {pl_module.mask.device}")
        except Exception as e:
            print(f"Error in batch start: {str(e)}")
        
    def setup(self, trainer, pl_module, stage):
        print(f"Setup called with stage: {stage}")
        
    def on_before_batch_transfer(self, trainer, pl_module, batch):
        print("Before batch transfer")
        return batch
        
    def on_after_batch_transfer(self, trainer, pl_module, batch):
        print("After batch transfer")
        return batch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        print(f"Finished batch {batch_idx}")
        
    def on_train_epoch_start(self, trainer, pl_module):
        print("\nEpoch starting!")
        print(f"Model device: {next(pl_module.parameters()).device}")
        print(f"Mask device: {pl_module.mask.device}")
        
    def on_train_epoch_end(self, trainer, pl_module):
        print("Epoch finished!")

class UNetModel(pl.LightningModule):
    def __init__(self,
            in_channels: int = 27,
            out_channels: int = 6,
            base_features: int = 64,
            lr: float = 1e-4,
            weight_decay: float = 1e-3,
            lambda_: float = 0.1,
            save_dir: str = 'results'):
            
        print("\nInitializing UNetModel...")
        super().__init__()
        
        print("Loading mask...")
        self.mask = np.load('data/mask_land.npy')
        #self.mask = 1 - self.mask
        self.mask = torch.from_numpy(self.mask).float()
        #self.mask = self.mask.permute(2, 0, 1).unsqueeze(0)
        print(self.mask)
        self.mask = torch.reshape(self.mask, [1, 1, 256, 256])
        # Move mask to GPU if available
        if torch.cuda.is_available():
            print("Moving mask to GPU")
            self.mask = self.mask.cuda()
        
        print(f"Mask shape after loading: {self.mask.shape}")
        print(f"Mask device: {self.mask.device}")
        
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
        
        # Initialize metrics tracking
        self.metrics_history = {
            'epoch': [],
            'train_loss': [], 'train_mse': [], 'train_bias': [],
            'val_loss': [], 'val_mse': [], 'val_bias': [],
            'learning_rate': []
        }
        
        # Encoder features progression
        features = [base_features, base_features * 2, base_features * 4]  # [32, 64, 128]
        
        self.encoder = Encoder(input_channels=in_channels, features=features)
        self.bottleneck = Bottleneck(features[-1], features[-1] * 2)  # 128 -> 256
        
        # Decoder features should match the encoder+bottleneck in reverse
        decoder_features = [
            features[-1] * 2,  # 256 (from bottleneck)
            features[-2],      # 64
            features[-3],      # 32
            out_channels       # 1 (final output)
        ]
        self.decoder = Decoder(features=decoder_features)  # [256, 64, 32] + final conv to 1
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_ = lambda_
        
        # Initialize loss parameters
        self.mse = nn.MSELoss()
        
        # Add debug prints in forward
        self.debug_callback = DebugCallback()
        
        print("Initialization complete!")

    def save_config(self):
        """Save model configuration to JSON"""
        config = {
            'in_channels': self.hparams.in_channels,
            'out_channels': self.hparams.out_channels,
            'base_features': self.hparams.base_features,
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
            'lambda_': self.hparams.lambda_,
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

    def encode(self,
            state_tensor: torch.Tensor,
            mask: torch.Tensor
            ):
        reduced_tensor, skip_features, reduced_mask, skip_masks = self.encoder(state_tensor, mask)
        return reduced_tensor, skip_features, reduced_mask, skip_masks

    def bottleneck_forward(self,
            reduced_tensor: torch.Tensor,
            reduced_mask: torch.Tensor):
        return self.bottleneck(reduced_tensor, reduced_mask)

    def decode(self,
            reduced_tensor: torch.Tensor,
            skip_features: list,
            skip_masks:list, 
            mask:torch.Tensor):
        return self.decoder(reduced_tensor, skip_features,  skip_masks, mask)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        #print('start') 
        # Encoding
        x, skip_features, mask, skip_masks = self.encode(x, mask)
        #print(x)
        #print("BN")
        # Bottleneck
        x, mask = self.bottleneck_forward(x, mask)
        #print(x)
        #print("decode")
        # Decoding
        x, mask = self.decode(x, skip_features, skip_masks, mask)
        return x, mask

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

    def compute_loss(self, y_pred, y_true):
        """
        Compute custom loss with MSE and bias constraint
        """
        
        # Compute MSE
        mse_loss = self.mse(y_pred, y_true)
        
        # Compute bias term (average difference across all pixels)
        pred_mean = torch.mean(y_pred)
        true_mean = torch.mean(y_true)
        bias_loss = (pred_mean - true_mean) ** 2
        
        # Combine losses
        total_loss = mse_loss + self.lambda_ * bias_loss
        
        return total_loss, mse_loss, bias_loss

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        #print('unpack')
        x, y = batch
        
        #print("predict")
        y_hat, mask_out = self(x, self.mask)
            
        #print("compute loss")
        loss, mse, bias = self.compute_loss(y_hat, y)
            
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)
        self.log('train_bias', bias, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        
        # Forward pass
        y_hat, mask_out = self(x, self.mask)
        
        # Compute loss with components
        val_loss, val_mse, val_bias = self.compute_loss(y_hat, y)
        
        # Log validation metrics
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mse', val_mse, on_step=True, on_epoch=True)
        self.log('val_bias', val_bias, on_step=True, on_epoch=True)
        
        return val_loss

    def on_train_epoch_end(self):
        # Get metrics from callback_metrics
        metrics = self.trainer.callback_metrics
        
        # Update metrics history only if metrics exist
        if 'train_loss_epoch' in metrics:
            self.metrics_history['epoch'].append(self.current_epoch)
            self.metrics_history['train_loss'].append(metrics['train_loss_epoch'].item())
            self.metrics_history['train_mse'].append(metrics['train_mse_epoch'].item())
            self.metrics_history['train_bias'].append(metrics['train_bias_epoch'].item())
            
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
        
        # Update metrics history only if metrics exist
        if 'val_loss_epoch' in metrics:
            # Ensure we have the same number of validation metrics as training metrics
            while len(self.metrics_history['val_loss']) < len(self.metrics_history['epoch']):
                self.metrics_history['val_loss'].append(None)
                self.metrics_history['val_mse'].append(None)
                self.metrics_history['val_bias'].append(None)
            
            # Add current validation metrics
            self.metrics_history['val_loss'].append(metrics['val_loss_epoch'].item())
            self.metrics_history['val_mse'].append(metrics['val_mse_epoch'].item())
            self.metrics_history['val_bias'].append(metrics['val_bias_epoch'].item())
            
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
