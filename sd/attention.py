"""
Attention mechanisms for diffusion models.

This module implements self-attention and cross-attention layers used
throughout the U-Net architecture for capturing spatial and contextual
dependencies.
"""

import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    """
    Multi-head self-attention layer.
    
    Computes scaled dot-product attention over the input sequence,
    allowing each position to attend to all positions.
    
    Args:
        n_heads: Number of attention heads.
        d_embed: Embedding dimension (must be divisible by n_heads).
        in_proj_bias: Whether to use bias in input projection.
        out_proj_bias: Whether to use bias in output projection.
    
    Example:
        >>> attn = SelfAttention(n_heads=8, d_embed=512)
        >>> x = torch.randn(2, 64, 512)  # (batch, seq_len, dim)
        >>> output = attn(x)  # (2, 64, 512)
    """
    
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask: bool = False) -> torch.Tensor:
        """
        Apply self-attention to input sequence.
        
        Args:
            x: Input tensor of shape (B, seq_len, d_embed).
            causal_mask: If True, apply causal masking to prevent
                attending to future positions.
        
        Returns:
            Output tensor of shape (B, seq_len, d_embed).
        """
        input_shape = x.shape 
        batch_size, sequence_length, d_embed = input_shape 
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head) 

        # Project to Q, K, V
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # Compute attention weights
        weight = q @ k.transpose(-1, -2)
        
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1) 
            weight.masked_fill_(mask, -torch.inf) 
        
        weight /= math.sqrt(self.d_head) 
        weight = F.softmax(weight, dim=-1) 

        # Apply attention to values
        output = weight @ v
        output = output.transpose(1, 2) 
        output = output.reshape(input_shape) 
        output = self.out_proj(output) 
        
        return output


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention layer.
    
    Computes attention between a query sequence and a context sequence,
    allowing the model to attend to conditioning information (e.g.,
    composition embeddings).
    
    Args:
        n_heads: Number of attention heads.
        d_embed: Query embedding dimension.
        d_cross: Context embedding dimension.
        in_proj_bias: Whether to use bias in input projections.
        out_proj_bias: Whether to use bias in output projection.
    
    Example:
        >>> attn = CrossAttention(n_heads=8, d_embed=512, d_cross=768)
        >>> x = torch.randn(2, 64, 512)      # query (latent features)
        >>> ctx = torch.randn(2, 10, 768)    # context (composition embedding)
        >>> output = attn(x, ctx)  # (2, 64, 512)
    """
    
    def __init__(
        self,
        n_heads: int,
        d_embed: int,
        d_cross: int,
        in_proj_bias: bool = True,
        out_proj_bias: bool = True
    ):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention between query and context.
        
        Args:
            x: Query tensor (latent features) of shape (B, seq_len_q, d_embed).
            y: Context tensor (conditioning) of shape (B, seq_len_kv, d_cross).
        
        Returns:
            Output tensor of shape (B, seq_len_q, d_embed).
        """
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)
        
        # Project Q from latent, K and V from context
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        # Reshape for multi-head attention
        q = q.view(interim_shape).transpose(1, 2) 
        k = k.view(interim_shape).transpose(1, 2) 
        v = v.view(interim_shape).transpose(1, 2) 
        
        # Compute attention weights
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        # Apply attention to values
        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)

        return output