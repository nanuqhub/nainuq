import torch
import torch.nn as nn
import torch.nn.functional as F

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
            
        # Validate input channels
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input to have {self.in_channels} channels, "
                f"but got {x.shape[1]} channels instead. "
                f"Input shape: {x.shape}"
            )
            
        
        with torch.no_grad():
            padding = [p for p in reversed(self.padding) for _ in range(2)]
            padded_mask = F.pad(mask, padding, mode='replicate')
            
            # Compute mask ratio
            mask_kernel = self.kernel_mask.expand(padded_mask.shape[1], 1, -1, -1)
            valid_pixels = F.conv2d(
                padded_mask, 
                mask_kernel,
                stride=self.stride,
                padding=0,
                dilation=self.dilation,
                groups=mask.shape[1]
            )
            
        mask_ratio = self.window_size * mask / (valid_pixels + 1e-8)
                
        # Apply mask to input
        #padded_x = x * mask
        # Perform partial convolution
        out = super().forward(x)
        # Apply mask ratio
        out = out * mask_ratio
        
        # Apply activation if specified
        if hasattr(self, 'activation') and self.activation is not None:
            out = self.activation(out)
            
        return out, mask

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
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_mish=True):
        super().__init__()
        activation = nn.Mish() if use_mish else nn.ReLU(inplace=True)
        
        self.double_conv = PConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.double_conv2 = PConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        #self.double_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        #self.double_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, x, mask):
        if mask is not None:
            if mask.shape[1] > mask.shape[-1]:
                mask = mask.permute(0, 3, 1, 2)
            #x = x * mask
        #x = self.double_conv(x)
        #x = self.double_conv2(x)
        x, mask = self.double_conv(x, mask)
        x, _ = self.double_conv2(x, mask)
        return x, mask

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, mask):
        x = self.maxpool(x)
        mask = self.maxpool(mask)
        x, mask = self.conv(x, mask)
        return x, mask

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        concat_channels = out_channels * 2
        self.conv = DoubleConv(concat_channels, out_channels, use_mish=True)

    def forward(self, x1, x2, mask):
        x1 = self.up(x1)
        
        #if mask1 is not None:
         #   mask1 = F.interpolate(mask1, size=x2.shape[2:], mode='nearest')
        
        #diff_y = x2.size()[2] - x1.size()[2]
        #diff_x = x2.size()[3] - x1.size()[3]
        
        #x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
         #              diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x1, x2], dim=1)
        #mask = None
        #if mask1 is not None and mask2 is not None:
         #   mask = mask1 * mask2
            
        x, mask = self.conv(x, mask)
        return x, mask

class Encoder(nn.Module):
    def __init__(self, input_channels, features=[64, 128, 256]):
        super().__init__()
        self.first_conv = DoubleConv(input_channels, features[0])
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(features[i], features[i + 1])
            for i in range(len(features) - 1)
        ])

    def forward(self, x, mask):
        features = []
        masks = []
        x, curr_mask = self.first_conv(x, mask = mask)
        features.append(x)
        masks.append(curr_mask)
        
        for block in self.encoder_blocks:
            x, curr_mask = block(x, curr_mask)
            features.append(x)
            masks.append(curr_mask)
        
        return x, features, curr_mask, masks

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottleneck = DoubleConv(in_channels, out_channels)

    def forward(self, x, mask):
        x, mask = self.bottleneck(x, mask)
        return x, mask
    
class Decoder(nn.Module):
    def __init__(self, features=[256, 128, 64, 1]):
        super().__init__()
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_channels=features[i], out_channels=features[i + 1])
            for i in range(len(features) - 2)
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[-2], features[-1], kernel_size=1),
            nn.Conv2d(features[-1], features[-1], kernel_size=1)
            )

    def forward(self, x, encoder_features, skip_masks, mask):
        encoder_features = encoder_features[:-1]
        encoder_features = encoder_features[::-1]
        skip_masks = skip_masks[:-1]
        skip_masks = skip_masks[::-1]
        for i, block in enumerate(self.decoder_blocks):

            skip_feature = encoder_features[i]
            skip_mask = skip_masks[i]
            x, mask = block(x, skip_feature, skip_mask)
        
        x = self.final_conv(x)
        return x, mask
