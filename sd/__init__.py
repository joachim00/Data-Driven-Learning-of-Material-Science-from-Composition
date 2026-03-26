"""
Composition-conditioned diffusion model for materials microstructure generation.

This package implements a VAE + DDPM architecture with CLIP-inspired
composition conditioning for generating microstructure images from
chemical compositions.

Modules:
    encoder: VAE encoder (image -> latent)
    decoder: VAE decoder (latent -> image)
    diffusion: U-Net noise predictor with cross-attention
    ddpm: DDPM noise scheduler and sampling
    attention: Self-attention and cross-attention layers
    true_ccip: True CCIP composition encoder (transformer-based)
    ccip: Original CCIP composition encoder
    pipeline: Inference pipeline utilities
    data_helper: HDF5 dataset loading
    true_ccip_data: True CCIP dataset loader

Example:
    >>> from sd.ddpm import DDPMSampler
    >>> from sd.diffusion import Diffusion
    >>> from sd.decoder import VAE_Decoder
    >>> from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer
"""

from sd.ddpm import DDPMSampler
from sd.diffusion import Diffusion
from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from sd.attention import SelfAttention, CrossAttention
from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer

__all__ = [
    # Core models
    "DDPMSampler",
    "Diffusion",
    "VAE_Encoder", 
    "VAE_Decoder",
    # Attention
    "SelfAttention",
    "CrossAttention",
    # Composition encoding
    "TrueCCIP",
    "CCIPCompositionTokenizer",
]

__version__ = "1.0.0"
