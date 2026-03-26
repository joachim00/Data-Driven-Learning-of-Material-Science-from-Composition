#!/usr/bin/env python3
"""
Show how True CCIP encoding works within the training pipeline.
"""

import torch
import sys
import os
sys.path.append('/Users/joachim/MasterThesisProject/stable_diffusion_from_scratch')

from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer
from sd.data_helper import HDF5CompositionImageDataset
from torchvision import transforms

def show_training_pipeline_encoding():
    """Demonstrate how True CCIP encoding works in the training pipeline."""
    
    print("=" * 80)
    print("True CCIP Encoding in Training Pipeline")
    print("=" * 80)
    
    # Create dataset like in training
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    tokenizer = CCIPCompositionTokenizer(max_length=20)
    
    print("Step 1: Dataset Creation")
    print("-" * 40)
    print(f"Created tokenizer with max_length={tokenizer.max_length}")
    
    try:
        dataset = HDF5CompositionImageDataset(
            h5_path='data/dataset_comp_image_spectra.h5',
            transform=image_transforms,
            comp_format='dict',
            tokenizer=tokenizer
        )
        print(f"Created dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Could not create dataset: {e}")
        return
    
    print("\nStep 2: Sample Data Loading")
    print("-" * 40)
    
    # Get a few samples
    for i in range(3):
        try:
            sample = dataset[i]
            comp_data, image = sample
            
            print(f"\nSample {i+1}:")
            print(f"  Original composition dict: {dataset.compositions[i]}")
            print(f"  Tokenized data type: {type(comp_data)}")
            
            if isinstance(comp_data, tuple) and len(comp_data) == 3:
                tokens, fractions, attention_mask = comp_data
                print(f"  Tokens shape: {tokens.shape}")
                print(f"  Fractions shape: {fractions.shape}")
                print(f"  Attention mask shape: {attention_mask.shape}")
                print(f"  Image shape: {image.shape}")
                
                # Show a few token values
                print(f"  First 10 tokens: {tokens[:10]}")
                print(f"  First 10 fractions: {fractions[:10]}")
                print(f"  First 10 attention mask: {attention_mask[:10]}")
                
            else:
                print(f"  Unexpected data format: {comp_data}")
                
        except Exception as e:
            print(f"  Error loading sample {i}: {e}")
    
    print("\nStep 3: Custom Collate Function")
    print("-" * 40)
    
    # Show how the custom collate function works
    def true_ccip_collate_fn(batch):
        """Custom collate function for True CCIP data format."""
        # Separate composition data and images
        comp_data, images = zip(*batch)
        
        # Check if composition data is tuples (True CCIP format)
        if isinstance(comp_data[0], tuple) and len(comp_data[0]) == 3:
            # Extract tokens, fractions, attention_mask from each sample
            tokens_list, fractions_list, attention_mask_list = zip(*comp_data)
            
            # Stack them into batch tensors
            tokens_batch = torch.stack(tokens_list, dim=0)
            fractions_batch = torch.stack(fractions_list, dim=0)
            attention_mask_batch = torch.stack(attention_mask_list, dim=0)
            
            # Return as tuple of (tokens, fractions, attention_mask), images
            return (tokens_batch, fractions_batch, attention_mask_batch), torch.stack(images, dim=0)
        else:
            # Standard format: just stack composition vectors and images
            return torch.stack(comp_data, dim=0), torch.stack(images, dim=0)
    
    # Create a small batch
    batch_size = 4
    try:
        batch = [dataset[i] for i in range(batch_size)]
        collated = true_ccip_collate_fn(batch)
        
        (tokens_batch, fractions_batch, attention_mask_batch), images_batch = collated
        
        print(f"Batch collation results:")
        print(f"  Tokens batch shape: {tokens_batch.shape}")
        print(f"  Fractions batch shape: {fractions_batch.shape}")
        print(f"  Attention mask batch shape: {attention_mask_batch.shape}")
        print(f"  Images batch shape: {images_batch.shape}")
        
    except Exception as e:
        print(f"Error in batch collation: {e}")
        return
    
    print("\nStep 4: True CCIP Model Forward Pass")
    print("-" * 40)
    
    # Create True CCIP model (like in training)
    model = TrueCCIP(
        vocab_size=98,
        embedding_dim=128,
        max_length=20,
        n_layers=6,
        n_heads=8,
        output_dim=512,
        dropout=0.1
    )
    
    print(f"Created True CCIP model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Forward pass
    try:
        with torch.no_grad():
            embeddings = model(tokens_batch, fractions_batch, attention_mask_batch)
            
        print(f"Model forward pass:")
        print(f"  Input shapes: tokens={tokens_batch.shape}, fractions={fractions_batch.shape}, mask={attention_mask_batch.shape}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Output range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
        print(f"  Output norms: {torch.norm(embeddings, dim=1)}")
        
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        return
    
    print("\nStep 5: Projection Layer (as in training)")
    print("-" * 40)
    
    # Show how the projection layer works
    class TrueCCIPWithProjection(torch.nn.Module):
        def __init__(self, true_ccip_module, output_dim):
            super().__init__()
            self.true_ccip = true_ccip_module
            self.projection = torch.nn.Linear(true_ccip_module.output_dim, output_dim)
        
        def forward(self, tokens, fractions, attention_mask):
            embeddings = self.true_ccip(tokens, fractions, attention_mask)
            return self.projection(embeddings)
    
    # Create wrapper (like in training)
    context_embed_dim = 64  # Target dimension for diffusion model
    wrapped_model = TrueCCIPWithProjection(model, context_embed_dim)
    
    try:
        with torch.no_grad():
            projected_embeddings = wrapped_model(tokens_batch, fractions_batch, attention_mask_batch)
            
        print(f"Projection layer:")
        print(f"  Input dimension: {model.output_dim}")
        print(f"  Output dimension: {context_embed_dim}")
        print(f"  Projected embeddings shape: {projected_embeddings.shape}")
        print(f"  Projected range: [{projected_embeddings.min():.4f}, {projected_embeddings.max():.4f}]")
        
    except Exception as e:
        print(f"Error in projection: {e}")
    
    print("\nStep 6: Training Loop Format")
    print("-" * 40)
    
    # Show how it's used in training loop
    print("In the training loop, the data flows like this:")
    print("  1. DataLoader provides: ((tokens, fractions, attention_mask), images)")
    print("  2. Context embedder processes: tokens, fractions, attention_mask → context_embeddings")
    print("  3. Context embeddings are unsqueezed: [batch, output_dim] → [batch, 1, output_dim]")
    print("  4. Diffusion model uses: (noisy_latents, context_embeddings, time_embeddings) → predicted_noise")
    
    # Show the unsqueeze operation
    context_embeddings = projected_embeddings.unsqueeze(1)  # Add sequence dimension
    print(f"  Context embeddings final shape: {context_embeddings.shape}")
    print(f"  Ready for diffusion model input!")
    
    print("\n" + "=" * 80)
    print("Summary of True CCIP Encoding Process:")
    print("=" * 80)
    print("1. Dictionary → Tokenization: {'Fe': 0.5, 'O': 0.5} → (tokens, fractions, attention_mask)")
    print("2. Batch Collation: List of tuples → Batched tensors")
    print("3. True CCIP Forward: (tokens, fractions, mask) → 512D embeddings")
    print("4. Projection: 512D → 64D (context_embed_dim)")
    print("5. Unsqueeze: [batch, 64] → [batch, 1, 64] for diffusion model")
    print("6. Diffusion Training: Uses context embeddings to condition noise prediction")
    print("\n✓ The pretrained True CCIP provides rich compositional representations!")
    print("✓ The projection layer adapts to the diffusion model's expected input size!")
    print("✓ Training preserves the pretrained knowledge while learning diffusion!")

if __name__ == "__main__":
    show_training_pipeline_encoding()
