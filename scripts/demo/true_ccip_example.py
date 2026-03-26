import torch
import torch.nn as nn
import torch.nn.functional as F

class TrueCCIPEmbedding(nn.Module):
    """
    True CLIP-style embedding for compositions.
    Each element is a separate token, like words in CLIP.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        # Element embeddings (like word embeddings in CLIP)
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        
        # Optional: fraction embeddings for hybrid approach
        self.fraction_projection = nn.Linear(1, embedding_dim)
        
    def forward(self, tokens, fractions=None, attention_mask=None):
        """
        Args:
            tokens: [batch_size, seq_len] - element token IDs
            fractions: [batch_size, seq_len] - optional fraction values
            attention_mask: [batch_size, seq_len] - mask for padding
        """
        batch_size, seq_len = tokens.shape
        
        # Token embeddings (each element gets its own embedding)
        token_emb = self.token_embedding(tokens)  # [batch_size, seq_len, embedding_dim]
        
        # Positional embeddings  
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)
        
        # Combine token + position embeddings
        embeddings = token_emb + pos_emb
        
        # Optional: add fraction information
        if fractions is not None:
            frac_emb = self.fraction_projection(fractions.unsqueeze(-1))
            embeddings = embeddings + frac_emb
        
        return embeddings


class TrueCCIP(nn.Module):
    """
    True CLIP-style CCIP model.
    Uses sequence of element tokens instead of weighted averages.
    """
    def __init__(self, vocab_size=95, embedding_dim=128, max_length=20, 
                 n_layers=6, n_heads=8, output_dim=512):
        super().__init__()
        
        self.embedding = TrueCCIPEmbedding(vocab_size, embedding_dim, max_length)
        
        # Transformer encoder (like CLIP text encoder)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer norm
        self.layernorm = nn.LayerNorm(embedding_dim)
        
        # Projection to output space (like CLIP)
        self.projection = nn.Linear(embedding_dim, output_dim)
        
        self.output_dim = output_dim
    
    def forward(self, tokens, fractions=None, attention_mask=None):
        """
        Forward pass for composition encoding.
        
        Args:
            tokens: [batch_size, seq_len] element token IDs
            fractions: [batch_size, seq_len] optional fractions
            attention_mask: [batch_size, seq_len] padding mask
        """
        # Get embeddings for each element token
        embeddings = self.embedding(tokens, fractions, attention_mask)
        # Shape: [batch_size, seq_len, embedding_dim]
        
        # Transpose for transformer (expects [seq_len, batch_size, embedding_dim])
        embeddings = embeddings.transpose(0, 1)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = ignore)
            mask = ~attention_mask.bool()
        else:
            mask = None
        
        # Apply transformer layers
        transformed = self.transformer(embeddings, src_key_padding_mask=mask)
        # Shape: [seq_len, batch_size, embedding_dim]
        
        # Transpose back
        transformed = transformed.transpose(0, 1)
        # Shape: [batch_size, seq_len, embedding_dim]
        
        # Global pooling (various strategies)
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (transformed * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-8)
        else:
            # Simple average pooling
            pooled = transformed.mean(dim=1)
        # Shape: [batch_size, embedding_dim]
        
        # Layer norm
        pooled = self.layernorm(pooled)
        
        # Project to output space
        output = self.projection(pooled)
        # Shape: [batch_size, output_dim]
        
        # L2 normalize (like CLIP)
        output = F.normalize(output, p=2, dim=-1)
        
        return output


# Example usage showing the difference
def compare_approaches():
    batch_size = 4
    
    # Current approach: weighted vectors
    print("=== CURRENT APPROACH (Weighted Vectors) ===")
    current_input = torch.randn(batch_size, 16)  # Pre-computed weighted vectors
    print(f"Input shape: {current_input.shape}")
    print(f"Input represents: Weighted combination of element embeddings")
    
    # True CLIP approach: element sequences
    print("\n=== TRUE CLIP APPROACH (Element Sequences) ===")
    tokens = torch.randint(3, 30, (batch_size, 20))  # Element token IDs
    fractions = torch.rand(batch_size, 20)  # Fraction values
    attention_mask = torch.ones(batch_size, 20)
    attention_mask[:, 15:] = 0  # Mask padding tokens
    
    print(f"Tokens shape: {tokens.shape}")
    print(f"Fractions shape: {fractions.shape}")
    print(f"Each position represents: Individual element + fraction")
    print(f"Sample tokens: {tokens[0][:10]}")
    print(f"Sample fractions: {fractions[0][:10]}")
    
    # Test the model
    model = TrueCCIP(vocab_size=95, embedding_dim=128, output_dim=64)
    
    with torch.no_grad():
        output = model(tokens, fractions, attention_mask)
        print(f"\nOutput shape: {output.shape}")
        print(f"Output norm: {torch.norm(output, dim=1)}")  # Should be ~1 (normalized)
    
    print("\n=== ADVANTAGES OF TRUE CLIP APPROACH ===")
    print("✅ Each element is a separate 'word' like in CLIP")
    print("✅ Self-attention between elements (Al-Fe interactions)")
    print("✅ Position-aware (majority vs minority elements)")
    print("✅ Handles variable compositions naturally")
    print("✅ Can learn complex compositional rules")
    print("✅ Better alignment with CLIP's design philosophy")

if __name__ == "__main__":
    compare_approaches()
