# System modules
import torch.nn.functional as F
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
class PConv2d(nn.Conv2d):
    """
    Partial 2D Convolution implementation in PyTorch, based on:
    'Image Inpainting for Irregular Holes Using Partial Convolutions' (Liu et al.)

    This layer extends the standard Conv2d to handle masked inputs. It applies the convolution
    only to valid (unmasked) pixels and updates the mask accordingly.

    The key features are:
    - Mask-aware convolution that only considers valid pixels
    - Dynamic mask updating based on convolution window coverage
    - Optional over-compensation to handle varying numbers of valid pixels

    Args:
        *args: Arguments passed to nn.Conv2d (channels, kernel_size, etc.)
        over_compensation (bool): Whether to apply window size compensation to handle
            varying numbers of valid pixels in each convolution window. When True,
            output values are scaled based on the ratio of valid pixels.
        **kwargs: Additional keyword arguments for nn.Conv2d (stride, padding, etc.)

    Example usage:
        pconv = PConv2d(3, 64, kernel_size=3, padding=1)
        output, new_mask = pconv(input_tensor, input_mask)
    """
    def __init__(self, *args, over_compensation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.over_compensation = over_compensation

        # Create mask kernel similar to the original implementation
        self.mask_kernel = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
        self.mask_kernel = self.mask_kernel.to(self.weight.device)

        # Calculate window size for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]

        # Register mask kernel as buffer (not a parameter)
        self.register_buffer('kernel_mask', self.mask_kernel)

    def forward(self, x, mask):
        """
        Forward pass of partial convolution.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            mask (torch.Tensor): Binary mask of shape [B, 1, H, W]

        Returns:
            tuple: (output, updated_mask)
        """
        mask_kernel = self.mask_kernel.to(x.device)
        x_masked = x*mask
        
        if self.over_compensation:
            with torch.no_grad():
                valid_count = F.conv2d(
                    mask,
                    mask_kernel,
                    bias=None,
                    stride=self.stride,
                    padding=self.padding,
                    dilation=self.dilation,
                )
                valid_count = torch.clamp(valid_count, min=1.0)  # avoid division by zero
        # Perform convolution on masked input
        out = F.conv2d(
            x_masked,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # Optional normalization by valid pixel fraction
        if self.over_compensation:
            out = out * (self.window_size / valid_count)
        
        out = out * mask

        return out
        #with torch.no_grad():
         #   padding = [p for p in reversed(self.padding) for _ in range(2)]
          #  padded_mask = F.pad(mask, padding, mode='replicate')

            # Compute mask ratio
           # mask_kernel = self.kernel_mask.expand(mask.shape[1], 1, -1, -1)
            #valid_pixels = F.conv2d(
             #   padded_mask,
              #  mask_kernel,
               # padding = 0,
                #stride=self.stride,
                #dilation=self.dilation,
                #groups=mask.shape[1]
            #)
            #if self.over_compensation:
#
 #               mask_ratio = self.window_size * mask / (valid_pixels + 1e-8)
  #          else:
   #             mask_ratio = mask
        # Apply mask to input
        #padded_x = F.pad(x * mask, padding, mode='replicate')
        #print(padded_x.shape)
        # Perform partial convolution
        #out = super().forward(x * mask)
        # Apply mask ratio
        #out = out * mask_ratio

        ## Apply activation if specified
        #if hasattr(self, 'activation') and self.activation is not None:
         #   out = self.activation(out)
          #  print(self.activation)
        #return out, mask

    def to(self, device):
        """
        Handles device transfer of internal mask kernel.

        Args:
            device: The target device (CPU/GPU)

        Returns:
            PConv2d: Returns self for method chaining
        """
        super().to(device)
        self.mask_kernel = self.mask_kernel.to(device)
        return self



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
        
    def on_train_epoch_end(self, trainer, pl_module):
        print("Epoch finished!")
#class DoubleConv(nn.Module):
 #   def __init__(self, in_channels, out_channels, use_mish=True):
  #      super().__init__()
   #     activation = nn.Mish() if use_mish else nn.ReLU(inplace=True)
    #    self.double_conv = nn.Sequential(
     #       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
     #       nn.GroupNorm(8, out_channels),
      #      activation,
       #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #    nn.GroupNorm(8, out_channels),
         #   activation,
        #)

    #def forward(self, x):
     #   return self.double_conv(x)
#class DoubleConv(nn.Module):
 #   """
  #  Double convolution block with optional residual connection.
   # Conv -> Mish -> Conv -> Mish + Residual
    #"""
    #def __init__(self, in_channels, out_channels, use_mish=True, dropout=0.1):
     #   super().__init__()
      #  self.use_residual = (in_channels == out_channels)
       # activation = nn.Mish() if use_mish else nn.ReLU(inplace=True)

        #self.double_conv = nn.Sequential(
         #   PConv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
          #  nn.GroupNorm(8, out_channels),
           # activation,
            #nn.Dropout2d(p=dropout),
            #PConv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            #nn.GroupNorm(8, out_channels),
            #activation,
            #nn.Dropout2d(p=dropout)
        #)

        ## If in/out channels differ, use a 1×1 conv to match for residual
        #if not self.use_residual:
        #    self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #else:
        #    self.res_conv = nn.Identity()

    #def forward(self, x):
     #   out = self.double_conv(x)
      #  res = self.res_conv(x)
       # return out + res


class DoubleConv(nn.Module):
    """
    Two successive Partial Convolution + GroupNorm + Mish + Dropout layers.
    Propagates both the feature map and the mask.
    """

    def __init__(self, in_channels, out_channels, dropout=0.1, num_groups=8, use_mish=True):
        super().__init__()
        activation = nn.Mish() if use_mish else nn.ReLU(inplace=True)

        # First partial convolution block
        self.pconv1 = PConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act1 = activation
        self.drop1 = nn.Dropout2d(p=dropout)

        # Second partial convolution block
        self.pconv2 = PConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.act2 = activation
        self.drop2 = nn.Dropout2d(p=dropout)

    def forward(self, x, mask):
        # First PConv
        x = self.pconv1(x, mask)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)

        # Second PConv
        x = self.pconv2(x, mask)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)

        return x
        
class PConv_UNetModel(pl.LightningModule):

    def __init__(self,
            in_channels: int = 17,
            out_channels: int = 10,
            base_features: int = 64,
            lr: float = 1e-4,
            weight_decay: float = 1e-3,
            lambda_: float = 0.1,
            save_dir: str = 'results'):
            
        print("\nInitializing UNetModel...")
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
        
        #Save mask
        self.mask = np.load('/linkhome/rech/genrea01/ucm13rr/nanuq1/src/mask2_nanuk1.npy')
        self.mask = torch.from_numpy(self.mask).float()
        #self.mask = self.mask.permute(2, 0, 1).unsqueeze(0)
        print(self.mask)
        self.mask = torch.reshape(self.mask, [1, 1, 128, 128])
        # Move mask to GPU if available
        if torch.cuda.is_available():
            print("Moving mask to GPU")
            self.mask = self.mask.cuda()
        
        # Initialize metrics tracking
        self.metrics_history = {
            'epoch': [],
            'train_loss': [], 'train_mse': [], 'train_bias': [],
            'val_loss': [], 'val_mse': [], 'val_bias': [],
            'learning_rate': []
        }
        
        # Encoder features progression
        self.features = [base_features, base_features * 2, base_features * 4, base_features*8]  # [32, 64, 128]
        
        #self.encoder = Encoder(input_channels=in_channels, features=features)
        #self.bottleneck = Bottleneck(features[-1], features[-1] * 2)  # 128 -> 256
        
        # Decoder features should match the encoder+bottleneck in reverse
        self.decoder_features = [
            self.features[-1],  # 256 (from bottleneck)
            self.features[-2],      # 64
            self.features[-3],
            self.features[-4], # 32
            out_channels       # 1 (final output)
        ]
        #self.decoder = Decoder(features=decoder_features)  # [256, 64, 32] + final conv to 1
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_ = lambda_
        
        # Initialize loss parameters
        self.loss = nn.MSELoss()
        
        # Add debug prints in forward
        self.debug_callback = DebugCallback()
        
        self.init_conv = DoubleConv(in_channels, self.features[0])                # base -> f0
        self.pool1 = nn.MaxPool2d(2)
        self.pool1_2 = nn.MaxPool2d(2)
        self.conv1  = DoubleConv(self.features[0], self.features[1])             # f0 -> f1
        self.pool2 = nn.MaxPool2d(2)
        self.pool2_2 = nn.MaxPool2d(2)
        self.conv2  = DoubleConv(self.features[1], self.features[2])             # f1 -> f2

        # Bottleneck (down)
        self.conv3 = DoubleConv(self.features[2], self.features[3])              # f2 -> f3

        # Bottleneck unwind (symmetric)
        self.conv4 = DoubleConv(self.features[3], self.features[2])              # f3 -> f2

        # Decoder (upsample + concat)
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = DoubleConv(self.features[2] + self.features[1], self.features[1], dropout = 0.2)            # (upsampled f2 + skip f2) -> f1

        self.up6 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv6 = DoubleConv(self.features[1] + self.features[0], self.features[0], dropout = 0.2)            # (upsampled f1 + skip f1) -> f0

        self.conv7 = DoubleConv(self.features[0], self.features[0], dropout = 0.2)
        self.final_conv = PConv2d(self.features[0], out_channels, kernel_size=1)
        #print("Initialization complete!")
        
        #Init
        #self.init_conv = DoubleConv(in_channels, self.features[0]) 

        #Block 1
        #self.pool1 = nn.MaxPool2d(2)
        #self.conv1 = DoubleConv(self.features[0], self.features[1])

        #Block 2
        #self.pool2 = nn.MaxPool2d(2)
        #self.conv2 = DoubleConv(self.features[1], self.features[2])
        
        #Bottleneck
        #self.conv3 = DoubleConv(self.features[2], self.features[3])
        #self.conv4 = DoubleConv(self.features[3], self.features[1])
    
        #Block 3
        #self.up5 = nn.Upsample(scale_factor = 2, mode = "bilinear",  align_corners=False)
        #self.conv5 = DoubleConv(self.features[2], self.features[0], align_corners=False) 
        #self.conv5 = DoubleConv(self.features[1]*2, self.features[0]) 
        #Block 4 
        #self.up6 = nn.Upsample(scale_factor = 2, mode = 'bilinear',  align_corners=False)
        #self.conv6 = DoubleConv(self.features[0]*2, self.features[0])

        #Final
        #self.conv7 = DoubleConv(self.features[0], self.features[0])
        #self.final_conv = PConv2d(self.features[0], out_channels, kernel_size=1)
        print("End init")     
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

    
    #def forward(self, x: torch.Tensor, sic_threshold: float = -0.50738):
        # Encoding
        
        #sic = x[:, 1, :, :]  # select the second channel
        #mask_sic = (sic > sic_threshold).float()  # 1 where SIC > threshold, else 0

        # Expand to all channels for broadcasting
        #mask = mask_sic.unsqueeze(1) # (B, 1, H, W)
        #mask = mask.expand(-1, 5, -1, -1) 
        #x = self.init_conv(torch.cat([x[:,:-5], x[:,-5:]*mask], dim = 1))
        
        #x = self.init_conv(x)
        
        #Block1
        #x1 = self.pool1(x)

        #x1_2 = self.conv1(x1)

        #Block2 
        #x2 = self.pool2(x1_2)
        

        #x2_2 = self.conv2(x2)

        #Bottleneck
        #x3 = self.conv3(x2_2)
        #x4 = self.conv4(x3)

        #Block3
        #x5 = self.up5(x4)
        #x5_2 = torch.cat([x5, x1_2], dim = 1)
        #x5_3 = self.conv5(x5_2)

        #Block 4
        #x6= self.up6(x5_3)
        #x6 = F.interpolate(x6, size=x.shape[-2:], mode='bilinear', align_corners=False)
        #x6_2 = torch.cat([x6, x], dim =1)
        #x6_3 = self.conv6(x6_2)
        
        #Block 5
        #x7 = self.conv7(x6_3)
        #x_final = self.final_conv(x7)#*self.mask
        
        #return x_final
    def forward(self, x):
        x0 = self.init_conv(x, self.mask)          # (B, f0, H, W)
        x1 = self.pool1(x0)
        mask1 = self.pool1_2(self.mask)
        
        x1 = self.conv1(x1, mask1)             # (B, f1, H/2, W/2)
        
        x2 = self.pool2(x1)
        mask2 = self.pool2_2(mask1)

        x2 = self.conv2(x2, mask2)             # (B, f2, H/4, W/4)

        x3 = self.conv3(x2, mask2)             # (B, f3, H/4, W/4)
        x4 = self.conv4(x3, mask2)             # (B, f2, H/4, W/4)

        x5 = self.up5(x4)               # (B, f2, H/2, W/2)  -- may require align
        # ensure match with x1
        x5 = torch.cat([x5, x1], dim=1) # (B, 2*f2, H/2, W/2)
        x5 = self.conv5(x5, mask1)             # (B, f1, H/2, W/2)

        x6 = self.up6(x5)               # (B, f1, H, W)
        x6 = torch.cat([x6, x0], dim=1) # (B, 2*f1, H, W)  -- careful: x0 has f0 channels; if shapes differ adapt

        x6 = self.conv6(x6, self.mask)
        x7 = self.conv7(x6, self.mask)
        out = self.final_conv(x7, self.mask)
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

    def compute_loss(self, y_pred, y_true):
        """
        Compute custom loss with MSE and bias constraint
        """
        
        # Compute MSE
        mse_loss = self.loss(y_pred, y_true)
        tv_loss = torch.mean(torch.abs(y_pred[:, :, :, :-1] - y_pred[:, :, :, 1:])) + torch.mean(torch.abs(y_pred[:, :, :-1, :] - y_pred[:, :, 1:, :]))
 
        # Compute bias term (average difference across all pixels)
        pred_mean = torch.mean(y_pred[:,0])
        true_mean = torch.mean(y_true[:,0])
        bias_loss_sit = (pred_mean - true_mean) ** 2
        
        pred_mean_sic = torch.mean(y_pred[:,1])
        true_mean_sic = torch.mean(y_true[:,1])
        bias_loss_sic = (pred_mean_sic - true_mean_sic) ** 2

        # Combine losses
        total_loss = mse_loss + self.lambda_ * (bias_loss_sit + bias_loss_sic) + 1e-2 * tv_loss
        
        return total_loss, mse_loss, bias_loss_sit, bias_loss_sic

    def training_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        y_hat = self(x)
            
        #print("compute loss")
        loss, mse, bias_sit, bias_sic = self.compute_loss(y_hat, y)
            
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True)
        self.log('train_bias_sit', bias_sit, on_step=False, on_epoch=True)
        self.log('train_bias_sic', bias_sic, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack the batch
        x, y = batch
        
        # Forward pass
        y_hat = self(x)
        
        # Compute loss with components
        val_loss, val_mse, val_bias_sit, val_bias_sic = self.compute_loss(y_hat, y)
        
        # Log validation metrics
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_mse', val_mse, on_step=True, on_epoch=True)
        self.log('val_bias_sit', val_bias_sit, on_step=False, on_epoch=True)
        self.log('val_bias_sic', val_bias_sic, on_step=False, on_epoch=True)
        
        return val_loss

    def on_train_epoch_end(self):
        # Get metrics from callback_metrics
        metrics = self.trainer.callback_metrics
        
        # Update metrics history only if metrics exist
        if 'train_loss_epoch' in metrics:
            self.metrics_history['epoch'].append(self.current_epoch)
            self.metrics_history['train_loss'].append(metrics['train_loss_epoch'].item())
            self.metrics_history['train_mse'].append(metrics['train_mse_epoch'].item())
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
