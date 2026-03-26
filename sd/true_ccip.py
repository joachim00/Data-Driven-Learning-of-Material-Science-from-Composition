"""
True CLIP-style Composition-Conditioned Image Prior (TrueCCIP).

This module implements a CLIP-inspired composition encoder that treats
chemical elements as tokens, similar to how CLIP treats words. This enables:
- Variable-length composition inputs
- Learned element interactions via self-attention
- Normalized embeddings for contrastive learning

The architecture follows the CLIP text encoder pattern with:
- Element token embeddings + fraction embeddings
- Transformer self-attention layers
- Global pooling and L2 normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class CCIPCompositionTokenizer:
    """
    CLIP-style tokenizer for chemical compositions.
    
    Converts composition dictionaries (e.g., {"Fe": 0.5, "Al": 0.5}) into
    token sequences where each element is a separate token, analogous to
    words in natural language.
    
    Attributes:
        max_length: Maximum sequence length including special tokens.
        elements: List of 95 supported chemical elements.
        vocab_size: Total vocabulary size (elements + special tokens).
        PAD_TOKEN: Padding token ID (0).
        START_TOKEN: Sequence start token ID (1).
        END_TOKEN: Sequence end token ID (2).
    
    Example:
        >>> tokenizer = CCIPCompositionTokenizer(max_length=20)
        >>> tokens, fracs, mask = tokenizer.tokenize({"Al": 0.33, "Fe": 0.67})
        >>> tokens.shape, fracs.shape
        (torch.Size([20]), torch.Size([20]))
    """
    
    def __init__(self, max_length: int = 20, fraction_bins: int = 100):
        """
        Initialize the tokenizer.
        
        Args:
            max_length: Maximum sequence length for compositions.
            fraction_bins: Number of bins for discretizing atomic fractions
                (used only for bin_to_fraction/fraction_to_bin methods).
        """
        self.max_length = max_length
        self.fraction_bins = fraction_bins
        
        # Standard periodic table elements (95 elements)
        self.elements = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U', 'Np', 'Pu', 'Am'
        ]
        
        # Element-to-ID mapping (IDs 3+ for elements)
        self.element_to_id = {elem: i + 3 for i, elem in enumerate(self.elements)}
        self.id_to_element = {i + 3: elem for i, elem in enumerate(self.elements)}
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
        self.vocab_size = len(self.elements) + 3
        
    def fraction_to_bin(self, fraction: float) -> int:
        """Convert continuous fraction [0,1] to discrete bin index."""
        return min(int(fraction * self.fraction_bins), self.fraction_bins - 1)
    
    def bin_to_fraction(self, bin_idx: int) -> float:
        """Convert discrete bin index back to fraction (bin center)."""
        return (bin_idx + 0.5) / self.fraction_bins
    
    def tokenize(
        self,
        composition: Dict[str, float]
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        """
        Convert composition dict to token sequence.
        
        Elements are sorted by fraction (descending) for consistent ordering.
        
        Args:
            composition: Dict mapping element symbols to atomic fractions,
                e.g., {"Al": 0.33, "Fe": 0.33, "Ni": 0.34}.
            
        Returns:
            tokens: Element token IDs of shape [max_length].
            fractions: Atomic fractions of shape [max_length].
            attention_mask: Boolean mask where True = real token.
        """
        # Sort elements by fraction (descending) for consistency
        sorted_elements = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize sequences with START token
        tokens = [self.START_TOKEN]
        fractions = [0.0]
        
        # Add element tokens
        for element, fraction in sorted_elements:
            if element in self.element_to_id and len(tokens) < self.max_length - 1:
                tokens.append(self.element_to_id[element])
                fractions.append(fraction)
        
        # Add END token
        if len(tokens) < self.max_length:
            tokens.append(self.END_TOKEN)
            fractions.append(0.0)
        
        # Create attention mask
        attention_mask = [True] * len(tokens)
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.PAD_TOKEN)
            fractions.append(0.0)
            attention_mask.append(False)
        
        return (
            torch.LongTensor(tokens),
            torch.FloatTensor(fractions),
            torch.BoolTensor(attention_mask)
        )
    
    def tokenize_batch(
        self,
        compositions: List[Dict[str, float]]
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        """
        Tokenize a batch of compositions.
        
        Args:
            compositions: List of composition dictionaries.
        
        Returns:
            Tuple of stacked (tokens, fractions, attention_mask) tensors.
        """
        batch_tokens = []
        batch_fractions = []
        batch_masks = []
        
        for comp in compositions:
            tokens, fractions, mask = self.tokenize(comp)
            batch_tokens.append(tokens)
            batch_fractions.append(fractions)
            batch_masks.append(mask)
        
        return (
            torch.stack(batch_tokens),
            torch.stack(batch_fractions),
            torch.stack(batch_masks)
        )
    
    def decode(
        self,
        tokens: torch.LongTensor,
        fractions: torch.FloatTensor, 
        attention_mask: torch.BoolTensor
    ) -> Dict[str, float]:
        """
        Convert tokens back to composition dict.
        
        Args:
            tokens: Token IDs of shape [seq_len].
            fractions: Atomic fractions of shape [seq_len].
            attention_mask: Boolean mask of shape [seq_len].
        
        Returns:
            Composition dictionary mapping elements to fractions.
        """
        composition = {}
        
        for token_id, fraction, is_real in zip(tokens, fractions, attention_mask):
            if not is_real or token_id.item() in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                continue
            if token_id.item() in self.id_to_element:
                element = self.id_to_element[token_id.item()]
                composition[element] = fraction.item()
        
        return composition


class TrueCCIPEmbedding(nn.Module):
    """
    Embedding layer combining element tokens with atomic fractions.
    
    Produces embeddings by summing:
    - Element token embeddings (learned per element)
    - Fraction embeddings (projected continuous values)
    - (Optional) Positional embeddings
    
    Args:
        vocab_size: Number of tokens in vocabulary.
        embedding_dim: Dimension of embeddings.
        max_length: Maximum sequence length.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.fraction_projection = nn.Linear(1, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        tokens: torch.LongTensor,
        fractions: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Compute embeddings for tokenized composition.
        
        Args:
            tokens: Token IDs of shape (B, seq_len).
            fractions: Atomic fractions of shape (B, seq_len).
        
        Returns:
            Embeddings of shape (B, seq_len, embedding_dim).
        """
        batch_size, seq_len = tokens.shape
        
        # Element embeddings
        token_emb = self.token_embedding(tokens)
        
        # Fraction embeddings
        frac_emb = self.fraction_projection(fractions.unsqueeze(-1))
        
        # Combine (positional embeddings not used currently)
        embeddings = token_emb + frac_emb
        embeddings = self.layer_norm(embeddings)
        
        return embeddings


class TrueCCIPTransformerLayer(nn.Module):
    """
    Transformer layer for learning element interactions.
    
    Uses multi-head self-attention to capture relationships between
    elements in a composition (e.g., synergistic or competing effects).
    
    Args:
        embedding_dim: Hidden dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.ff_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: Optional[torch.BoolTensor] = None
    ) -> torch.FloatTensor:
        """
        Apply transformer layer.
        
        Args:
            x: Input of shape (B, seq_len, embedding_dim).
            attention_mask: Boolean mask where True = real token.
        
        Returns:
            Output of shape (B, seq_len, embedding_dim).
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        
        key_padding_mask = ~attention_mask if attention_mask is not None else None
        attn_output, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + attn_output
        
        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = residual + self.ff_net(x)
        
        return x


class TrueCCIP(nn.Module):
    """
    True CLIP-style composition encoder.
    
    Encodes chemical compositions into fixed-length embeddings by treating
    each element as a token (like words in CLIP). Uses transformer attention
    to learn element interactions and produces L2-normalized outputs.
    
    Args:
        vocab_size: Vocabulary size (default: 98 = 95 elements + 3 special).
        embedding_dim: Internal embedding dimension (default: 128).
        max_length: Maximum sequence length (default: 20).
        n_layers: Number of transformer layers (default: 6).
        n_heads: Number of attention heads (default: 8).
        output_dim: Output embedding dimension (default: 512).
        dropout: Dropout probability (default: 0.1).
    
    Example:
        >>> model = TrueCCIP(output_dim=512)
        >>> tokenizer = CCIPCompositionTokenizer()
        >>> tokens, fracs, mask = tokenizer.tokenize({"Fe": 0.5, "Al": 0.5})
        >>> embedding = model(tokens.unsqueeze(0), fracs.unsqueeze(0), mask.unsqueeze(0))
        >>> embedding.shape
        torch.Size([1, 512])
    """
    
    def __init__(
        self, 
        vocab_size: int = 98,
        embedding_dim: int = 128,
        max_length: int = 20,
        n_layers: int = 6,
        n_heads: int = 8,
        output_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.output_dim = output_dim
        
        self.embedding = TrueCCIPEmbedding(vocab_size, embedding_dim, max_length)
        
        self.transformer_layers = nn.ModuleList([
            TrueCCIPTransformerLayer(embedding_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.projection = nn.Linear(embedding_dim, output_dim)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using CLIP-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        tokens: torch.LongTensor,
        fractions: torch.FloatTensor, 
        attention_mask: torch.BoolTensor
    ) -> torch.FloatTensor:
        """
        Encode composition to normalized embedding.
        
        Args:
            tokens: Element token IDs of shape (B, seq_len).
            fractions: Atomic fractions of shape (B, seq_len).
            attention_mask: Boolean mask of shape (B, seq_len).
        
        Returns:
            L2-normalized composition embeddings of shape (B, output_dim).
        """
        # Embed tokens
        x = self.embedding(tokens, fractions)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Masked average pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = x.mean(dim=1)
        
        # Project and normalize
        composition_features = self.projection(pooled)
        composition_features = F.normalize(composition_features, p=2, dim=-1)
        
        return composition_features
    
    def encode_composition(
        self,
        composition: Dict[str, float],
        tokenizer: CCIPCompositionTokenizer
    ) -> torch.FloatTensor:
        """
        Convenience method to encode a single composition.
        
        Args:
            composition: Composition dict, e.g., {"Al": 0.33, "Fe": 0.67}.
            tokenizer: CCIPCompositionTokenizer instance.
        
        Returns:
            Normalized embedding of shape (output_dim,).
        """
        tokens, fractions, attention_mask = tokenizer.tokenize(composition)
        
        tokens = tokens.unsqueeze(0)
        fractions = fractions.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        
        with torch.no_grad():
            features = self.forward(tokens, fractions, attention_mask)
        
        return features.squeeze(0)


if __name__ == "__main__":
    print("Testing True CLIP-style CCIP Implementation")
    print("=" * 50)
    
    tokenizer = CCIPCompositionTokenizer(max_length=20)
    model = TrueCCIP(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=128,
        max_length=20,
        n_layers=6,
        n_heads=8,
        output_dim=512
    )
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Max Sequence Length: {tokenizer.max_length}")
    
    test_compositions = [
        {"Al": 0.33, "Fe": 0.33, "Ni": 0.34},
        {"Cu": 0.5, "Zn": 0.5},
        {"Ti": 0.6, "O": 0.4},
        {"C": 1.0}
    ]
    
    print("\nTesting Tokenization:")
    for i, comp in enumerate(test_compositions):
        tokens, fractions, mask = tokenizer.tokenize(comp)
        print(f"  Composition {i+1}: {comp}")
        print(f"    Tokens: {tokens[:8]}")
        print(f"    Fractions: {[f'{f:.3f}' for f in fractions[:8]]}")
        print(f"    Real tokens: {mask.sum().item()}")
        
        decoded = tokenizer.decode(tokens, fractions, mask)
        print(f"    Decoded: {decoded}")
        print()
    
    print("Testing Batch Processing:")
    batch_tokens, batch_fractions, batch_masks = tokenizer.tokenize_batch(test_compositions)
    print(f"  Batch shapes:")
    print(f"    Tokens: {batch_tokens.shape}")
    print(f"    Fractions: {batch_fractions.shape}")
    print(f"    Masks: {batch_masks.shape}")
    
    print("\nTesting Model Forward Pass:")
    model.eval()
    with torch.no_grad():
        features = model(batch_tokens, batch_fractions, batch_masks)
        print(f"  Output shape: {features.shape}")
        print(f"  Output norms: {torch.norm(features, dim=1)}")
        print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    print("\nTesting Composition Similarity:")
    similarities = torch.mm(features, features.t())
    print(f"  Similarity matrix:")
    for i in range(len(test_compositions)):
        row = [f"{sim:.3f}" for sim in similarities[i]]
        print(f"    {row}")
    
    print("\nTrue CLIP-style CCIP implementation complete!")
    print("Key features:")
    print("   - Each element is a separate token (like words in CLIP)")
    print("   - Self-attention learns element interactions")
    print("   - Handles variable composition lengths")
    print("   - L2-normalized outputs for contrastive learning")
