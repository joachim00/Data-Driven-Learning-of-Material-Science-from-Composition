"""
Variational Autoencoder (VAE) decoder for latent diffusion models.

This module implements the decoder component of a VAE that reconstructs
images from latent representations.
"""

import torch
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    """
    Self-attention block for VAE encoder/decoder.
    
    Applies self-attention over spatial dimensions with a residual connection.
    Used at the bottleneck of the VAE for capturing long-range dependencies.
    
    Args:
        channels: Number of input/output channels.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention with residual connection.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
        
        Returns:
            Output tensor of shape (B, C, H, W).
        """
        residue = x 
        x = self.groupnorm(x)

        n, c, h, w = x.shape
        
        # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        
        # Self-attention
        x = self.attention(x)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))
        
        return x + residue


class VAE_ResidualBlock(nn.Module):
    """
    Residual block with GroupNorm and SiLU activation.
    
    Uses pre-activation residual connection pattern with optional
    channel projection for the skip connection.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (B, in_channels, H, W).
        
        Returns:
            Output tensor of shape (B, out_channels, H, W).
        """
        residue = x
        
        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        
        return x + self.residual_layer(residue)


class VAE_Decoder(nn.Sequential):
    """
    VAE decoder that reconstructs images from latent representations.
    
    The decoder upsamples latent representations by a factor of 8 and
    reconstructs RGB images. It mirrors the encoder architecture with
    residual blocks and self-attention.
    
    Architecture:
        - Input: (B, 4, H/8, W/8) latent representation
        - 3 upsampling stages (each 2x increase)
        - Multiple residual blocks with decreasing channels (512 -> 256 -> 128)
        - Self-attention at the bottleneck
        - Output: (B, 3, H, W) RGB images
    
    Example:
        >>> decoder = VAE_Decoder()
        >>> latent = torch.randn(1, 4, 8, 8)
        >>> image = decoder(latent)
        >>> image.shape
        torch.Size([1, 3, 64, 64])
    """
    
    def __init__(self):
        super().__init__(
            # Input projection
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            
            # Bottleneck with attention
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Upsample 1: (B, 512, H/8, W/8) -> (B, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Upsample 2: (B, 512, H/4, W/4) -> (B, 256, H/2, W/2)
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            VAE_ResidualBlock(512, 256), 
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            
            # Upsample 3: (B, 256, H/2, W/2) -> (B, 128, H, W)
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            VAE_ResidualBlock(256, 128), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 
            
            # Output projection
            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            nn.Conv2d(128, 3, kernel_size=3, padding=1), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.
        
        Args:
            x: Latent tensor of shape (B, 4, H/8, W/8).
        
        Returns:
            Reconstructed image of shape (B, 3, H, W).
        """
        for module in self:
            x = module(x)
        return x