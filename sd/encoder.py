"""
Variational Autoencoder (VAE) encoder for latent diffusion models.

This module implements the encoder component of a VAE that compresses
images into a lower-dimensional latent space suitable for diffusion modeling.
"""

import torch
from torch import nn
from torch.nn import functional as F
from sd.decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    """
    VAE encoder that compresses images to latent representations.
    
    The encoder downsamples input images by a factor of 8 and produces
    a latent representation with 4 channels. It uses residual blocks
    and self-attention for high-quality compression.
    
    Architecture:
        - Input: (B, 3, H, W) RGB images
        - 3 downsampling stages (each 2x reduction)
        - Multiple residual blocks with increasing channels (128 -> 256 -> 512)
        - Self-attention at the bottleneck
        - Output: (B, 4, H/8, W/8) latent representation
    
    The encoder outputs mean and log-variance for the VAE's reparameterization trick.
    
    Example:
        >>> encoder = VAE_Encoder()
        >>> image = torch.randn(1, 3, 64, 64)
        >>> noise = torch.randn(1, 4, 8, 8)
        >>> latent, mean, log_var = encoder(image, noise)
        >>> latent.shape
        torch.Size([1, 4, 8, 8])
    """
    
    def __init__(self):
        super().__init__(
            # Initial convolution: (B, 3, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            # First residual blocks at 128 channels
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            # Downsample 1: (B, 128, H, W) -> (B, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),
            
            # Second stage at 256 channels
            VAE_ResidualBlock(128, 256), 
            VAE_ResidualBlock(256, 256), 
            
            # Downsample 2: (B, 256, H/2, W/2) -> (B, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 
            
            # Third stage at 512 channels
            VAE_ResidualBlock(256, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Downsample 3: (B, 512, H/4, W/4) -> (B, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 
            
            # Bottleneck with attention
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_AttentionBlock(512), 
            VAE_ResidualBlock(512, 512), 
            
            # Output projection
            nn.GroupNorm(32, 512), 
            nn.SiLU(), 
            nn.Conv2d(512, 8, kernel_size=3, padding=1),  # 8 = 2 * latent_channels (mean + log_var)
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images to latent space using the reparameterization trick.
        
        Args:
            x: Input images of shape (B, 3, H, W).
            noise: Random noise for reparameterization, shape (B, 4, H/8, W/8).
        
        Returns:
            Tuple of (latent, mean, log_variance) where:
                - latent: Sampled latent representation (B, 4, H/8, W/8)
                - mean: Mean of the latent distribution (B, 4, H/8, W/8)
                - log_variance: Log variance of the latent distribution (B, 4, H/8, W/8)
        """
        for module in self:
            # Apply asymmetric padding before strided convolutions
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # Split into mean and log-variance
        mean, log_variance = torch.chunk(x, 2, dim=1)
        
        # Clamp log variance for numerical stability
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        
        # Reparameterization trick: z = mean + std * noise
        x = mean + stdev * noise
        
        return x, mean, log_variance