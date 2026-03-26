"""
U-Net diffusion model for noise prediction.

This module implements the U-Net architecture for denoising diffusion
probabilistic models (DDPM). The model predicts noise given a noisy
latent, timestep embedding, and optional context embedding.

Architecture follows the Stable Diffusion U-Net with:
- Sinusoidal time embeddings
- ResNet blocks with time conditioning
- Self-attention and cross-attention layers
- Skip connections between encoder and decoder
"""

import torch
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention, CrossAttention


class TimeEmbedding(nn.Module):
    """
    MLP for processing sinusoidal timestep embeddings.
    
    Transforms the initial timestep embedding through a 2-layer MLP
    with SiLU activation, expanding the dimension by 4x.
    
    Args:
        n_embd: Input embedding dimension (typically 320).
    """
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform timestep embedding.
        
        Args:
            x: Timestep embedding of shape (B, n_embd).
        
        Returns:
            Transformed embedding of shape (B, 4 * n_embd).
        """
        x = self.linear_1(x)
        x = F.silu(x) 
        x = self.linear_2(x)
        return x


class UNET_ResidualBlock(nn.Module):
    """
    Residual block with timestep conditioning.
    
    Combines spatial features with timestep information through
    addition after a linear projection of the time embedding.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n_time: Dimension of time embedding (default: 1280).
    """
    
    def __init__(self, in_channels: int, out_channels: int, n_time: int = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time conditioning.
        
        Args:
            feature: Spatial features of shape (B, in_channels, H, W).
            time: Time embedding of shape (B, n_time).
        
        Returns:
            Output features of shape (B, out_channels, H, W).
        """
        residue = feature
        
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        
        time = F.silu(time)
        time = self.linear_time(time)
        
        # Add time embedding (broadcast over spatial dimensions)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    """
    Transformer block with self-attention, cross-attention, and FFN.
    
    Applies a full transformer layer over spatial features:
    1. Self-attention for spatial relationships
    2. Cross-attention for context conditioning
    3. Feed-forward network with GeGLU activation
    
    Args:
        n_head: Number of attention heads.
        n_embd: Embedding dimension per head.
        d_context: Dimension of context embeddings (default: 768).
    """
    
    def __init__(self, n_head: int, n_embd: int, d_context: int = 768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block with cross-attention.
        
        Args:
            x: Spatial features of shape (B, C, H, W).
            context: Context embeddings of shape (B, seq_len, d_context).
        
        Returns:
            Transformed features of shape (B, C, H, W).
        """
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # Flatten spatial dimensions for attention
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        
        # Self-attention
        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short
        
        # Cross-attention with context
        residue_short = x
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residue_short
        
        # Feed-forward with GeGLU
        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        
        # Reshape back to spatial
        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


class Upsample(nn.Module):
    """
    2x upsampling with convolution.
    
    Args:
        channels: Number of input/output channels.
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2x using nearest neighbor interpolation."""
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)


class SwitchSequential(nn.Sequential):
    """
    Sequential container that routes inputs based on layer type.
    
    Automatically passes the appropriate arguments to each layer:
    - UNET_AttentionBlock: receives (x, context)
    - UNET_ResidualBlock: receives (x, time)
    - Other layers: receive only x
    """
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with routing based on layer type."""
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    """
    U-Net architecture for diffusion noise prediction.
    
    Encoder-decoder architecture with skip connections. Each stage
    consists of residual blocks and attention layers, with downsampling
    in the encoder and upsampling in the decoder.
    
    Args:
        d_context: Dimension of context embeddings from composition encoder.
    
    Architecture:
        - Encoder: 4 stages with channels [320, 640, 1280, 1280]
        - Bottleneck: ResNet + Attention + ResNet at 1280 channels
        - Decoder: 4 stages mirroring encoder with skip connections
    """
    
    def __init__(self, d_context: int = 64):
        super().__init__()
        self.encoders = nn.ModuleList([
            # Stage 1: 320 channels at H/8
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40, d_context=d_context)),
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # Stage 2: 640 channels at H/16
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80, d_context=d_context)),
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # Stage 3: 1280 channels at H/32
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160, d_context=d_context)),
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # Stage 4: 1280 channels at H/64 (no attention)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280), 
            UNET_AttentionBlock(8, 160, d_context=d_context), 
            UNET_ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # Stage 4 decoder
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # Stage 3 decoder
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160, d_context=d_context), Upsample(1280)),
            
            # Stage 2 decoder
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80, d_context=d_context), Upsample(640)),
            
            # Stage 1 decoder
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, d_context=d_context)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40, d_context=d_context)),
        ])

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through U-Net.
        
        Args:
            x: Noisy latent of shape (B, 4, H/8, W/8).
            context: Context embeddings of shape (B, seq_len, d_context).
            time: Time embedding of shape (B, 1280).
        
        Returns:
            Predicted noise of shape (B, 320, H/8, W/8).
        """
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    """
    Final output projection layer.
    
    Projects U-Net output to the latent channel dimension.
    
    Args:
        in_channels: Input channels from U-Net (typically 320).
        out_channels: Output channels (typically 4 for latent space).
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to latent channels."""
        x = self.groupnorm(x)
        x = F.silu(x)
        x = self.conv(x)
        return x


class Diffusion(nn.Module):
    """
    Complete diffusion model combining time embedding, U-Net, and output projection.
    
    This is the main model class that should be instantiated for training
    and inference. It handles:
    - Timestep embedding expansion
    - U-Net forward pass with context conditioning
    - Final projection to predicted noise
    
    Args:
        d_context: Dimension of context embeddings (default: 64).
            Should match the output dimension of the composition encoder.
    
    Example:
        >>> model = Diffusion(d_context=512)
        >>> latent = torch.randn(2, 4, 8, 8)
        >>> context = torch.randn(2, 10, 512)
        >>> time = torch.randn(2, 320)
        >>> noise_pred = model(latent, context, time)
        >>> noise_pred.shape
        torch.Size([2, 4, 8, 8])
    """
    
    def __init__(self, d_context: int = 64):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET(d_context=d_context)
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise from noisy latent.
        
        Args:
            latent: Noisy latent of shape (B, 4, H/8, W/8).
            context: Context embeddings of shape (B, seq_len, d_context).
            time: Sinusoidal timestep embedding of shape (B, 320).
        
        Returns:
            Predicted noise of shape (B, 4, H/8, W/8).
        """
        time = self.time_embedding(time)
        output = self.unet(latent, context, time)
        output = self.final(output)
        return output