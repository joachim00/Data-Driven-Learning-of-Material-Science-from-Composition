import torch 
from torch import nn
from torch.nn import functional as F
from sd.attention import SelfAttention

class CCIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_length, embedding_dim)
        self.max_length = max_length
        self.embedding_dim = embedding_dim

    def forward(self, tokens):
        # tokens shape: (batch_size, seq_len)
        batch_size, seq_len = tokens.shape
        
        # Token embeddings
        x = self.token_embedding(tokens)  # (batch_size, seq_len, embedding_dim)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # (batch_size, seq_len, embedding_dim)
        
        x = x + pos_emb
        return x
        

class CCIPLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm2 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(n_embd, n_embd * 4)
        self.linear2 = nn.Linear(n_embd * 4, n_embd)

    def forward(self, x):
        residue = x
        x = self.layernorm1(x)
        x = self.attention(x)
        x = x + residue

        residue = x
        x = self.layernorm2(x)
        x = self.linear1(x)
        x = x * torch.sigmoid(1.702 * x)
        x = self.linear2(x)
        x = x + residue
        return x




class CCIP(nn.Module):
    def __init__(self, vocab_size=95, embedding_dim=128, max_length=20, n_layers=6, n_heads=8, output_dim=512):
        super().__init__()
        self.embedding = CCIPEmbedding(vocab_size, embedding_dim, max_length)
        
        # Use ModuleList to store transformer layers
        self.layers = nn.ModuleList([
            CCIPLayer(n_heads, embedding_dim) for _ in range(n_layers)
        ])
        
        self.layernorm = nn.LayerNorm(embedding_dim)
        
        # Project to final output dimension (similar to CLIP)
        self.projection = nn.Linear(embedding_dim, output_dim)
        
        self.output_dim = output_dim

    def forward(self, tokens: torch.LongTensor, attention_mask=None) -> torch.FloatTensor:
        # tokens shape: (batch_size, seq_len)
        tokens = tokens.type(torch.long)
        
        # Get embeddings
        state = self.embedding(tokens)  # (batch_size, seq_len, embedding_dim)
        
        # Apply transformer layers
        for layer in self.layers:
            state = layer(state)
        
        # Final layer norm
        state = self.layernorm(state)
        
        # Global average pooling over sequence dimension
        if attention_mask is not None:
            # Mask out padding tokens
            mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            state = state * mask
            pooled = state.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        else:
            pooled = state.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Project to output space
        output = self.projection(pooled)  # (batch_size, output_dim)
        
        # Normalize for contrastive learning (like CLIP)
        output = F.normalize(output, p=2, dim=-1)
        
        return output