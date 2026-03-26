#!/usr/bin/env python3
"""
Demonstration of how a dictionary gets encoded by True CCIP.
"""

import torch
import sys
import os
sys.path.append('/Users/joachim/MasterThesisProject/stable_diffusion_from_scratch')

from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer

def demonstrate_ccip_encoding():
    """Show how a composition dictionary gets encoded by True CCIP."""
    
    print("=" * 80)
    print("DEMONSTRATION: How True CCIP Encodes Composition Dictionaries")
    print("=" * 80)
    
    # Create the tokenizer
    tokenizer = CCIPCompositionTokenizer(max_length=20)
    print(f"Created tokenizer with max_length={tokenizer.max_length}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Special tokens: PAD={tokenizer.PAD_TOKEN}, START={tokenizer.START_TOKEN}, END={tokenizer.END_TOKEN}")
    print()
    
    # Example composition dictionaries
    compositions = [
        {"Fe": 0.5, "O": 0.5},  # Simple iron oxide
        {"Fe": 0.7, "Ni": 0.2, "Cr": 0.1},  # Steel-like alloy
        {"Al": 0.6, "O": 0.4},  # Aluminum oxide
        {"Si": 0.33, "O": 0.67},  # Silicon dioxide
        {"Ca": 0.2, "Fe": 0.3, "Si": 0.3, "O": 0.2},  # Complex composition
    ]
    
    print("Step 1: Raw Composition Dictionaries")
    print("-" * 40)
    for i, comp in enumerate(compositions):
        print(f"Composition {i+1}: {comp}")
    print()
    
    # Tokenize each composition
    print("Step 2: Tokenization Process")
    print("-" * 40)
    
    tokenized_data = []
    for i, comp in enumerate(compositions):
        print(f"\nComposition {i+1}: {comp}")
        
        # Tokenize
        tokens, fractions, attention_mask = tokenizer.tokenize(comp)
        tokenized_data.append((tokens, fractions, attention_mask))
        
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Fractions shape: {fractions.shape}")
        print(f"  Attention mask shape: {attention_mask.shape}")
        
        # Show the actual token values
        print(f"  Raw tokens: {tokens}")
        print(f"  Raw fractions: {fractions}")
        print(f"  Attention mask: {attention_mask}")
        
        # Decode tokens to show elements
        decoded_elements = []
        for j in range(len(tokens)):
            token_id = tokens[j].item()
            if token_id == tokenizer.PAD_TOKEN:
                decoded_elements.append("PAD")
            elif token_id == tokenizer.START_TOKEN:
                decoded_elements.append("START")
            elif token_id == tokenizer.END_TOKEN:
                decoded_elements.append("END")
            elif token_id in tokenizer.id_to_element:
                decoded_elements.append(tokenizer.id_to_element[token_id])
            else:
                decoded_elements.append(f"UNK({token_id})")
        
        print(f"  Decoded tokens: {decoded_elements}")
        
        # Show non-zero fractions
        active_positions = attention_mask.bool()
        active_tokens = tokens[active_positions]
        active_fractions = fractions[active_positions]
        active_elements = [decoded_elements[j] for j in range(len(decoded_elements)) if attention_mask[j]]
        
        print(f"  Active elements: {list(zip(active_elements, active_fractions.tolist()))}")
    
    print("\n" + "=" * 80)
    print("Step 3: True CCIP Model Forward Pass")
    print("=" * 80)
    
    # Create True CCIP model
    model = TrueCCIP(
        vocab_size=98,
        embedding_dim=128,
        max_length=20,
        n_layers=6,
        n_heads=8,
        output_dim=512,
        dropout=0.1
    )
    
    print(f"Created True CCIP model:")
    print(f"  Vocab size: {model.vocab_size}")
    print(f"  Embedding dim: {model.embedding_dim}")
    print(f"  Max length: {model.max_length}")
    print(f"  Output dim: {model.output_dim}")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Process each composition through the model
    print("\nForward pass results:")
    print("-" * 40)
    
    with torch.no_grad():
        for i, (tokens, fractions, attention_mask) in enumerate(tokenized_data):
            # Add batch dimension
            tokens_batch = tokens.unsqueeze(0)
            fractions_batch = fractions.unsqueeze(0)
            attention_mask_batch = attention_mask.unsqueeze(0)
            
            # Forward pass
            output = model(tokens_batch, fractions_batch, attention_mask_batch)
            
            print(f"\nComposition {i+1}: {compositions[i]}")
            print(f"  Input shapes: tokens={tokens_batch.shape}, fractions={fractions_batch.shape}, mask={attention_mask_batch.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Output norm: {torch.norm(output).item():.4f}")
            print(f"  Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"  Output mean: {output.mean().item():.4f}")
            print(f"  Output std: {output.std().item():.4f}")
    
    print("\n" + "=" * 80)
    print("Step 4: Batch Processing")
    print("=" * 80)
    
    # Show batch processing
    all_tokens = torch.stack([data[0] for data in tokenized_data])
    all_fractions = torch.stack([data[1] for data in tokenized_data])
    all_attention_masks = torch.stack([data[2] for data in tokenized_data])
    
    print(f"Batch processing {len(compositions)} compositions:")
    print(f"  Batch tokens shape: {all_tokens.shape}")
    print(f"  Batch fractions shape: {all_fractions.shape}")
    print(f"  Batch attention masks shape: {all_attention_masks.shape}")
    
    with torch.no_grad():
        batch_output = model(all_tokens, all_fractions, all_attention_masks)
        
    print(f"  Batch output shape: {batch_output.shape}")
    print(f"  Each composition → {batch_output.shape[1]}D embedding")
    
    # Show similarity between compositions
    print("\nSimilarity matrix (cosine similarity):")
    similarities = torch.mm(batch_output, batch_output.t())
    print(f"Similarity matrix shape: {similarities.shape}")
    
    print("\nPairwise similarities:")
    for i in range(len(compositions)):
        for j in range(i+1, len(compositions)):
            sim = similarities[i, j].item()
            print(f"  Comp {i+1} vs Comp {j+1}: {sim:.4f}")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print("✓ Dictionary → Tokenization: Compositions are converted to (tokens, fractions, attention_mask)")
    print("✓ Tokenization → Embedding: Each token gets embedded and combined with fraction info")
    print("✓ Transformer Processing: Multi-head attention processes the sequence")
    print("✓ Global Pooling: Masked average pooling creates a single representation")
    print("✓ Projection: Linear layer maps to final output dimension")
    print("✓ Normalization: L2 normalization for contrastive learning")
    print(f"✓ Final Output: {batch_output.shape[1]}D normalized embeddings ready for similarity comparison")
    
    return True

if __name__ == "__main__":
    demonstrate_ccip_encoding()
