#!/usr/bin/env python3
"""
Example usage of the DDPM + True CCIP generation script with different tokenizer embeddings.

This script demonstrates how to use the tokenizer-only mode with various element embeddings.
"""

import os
import subprocess
import sys

def run_generation_example(embedding_type, compositions):
    """Run a generation example with a specific embedding type."""
    
    embedding_path = f"embedding/tokenizers/{embedding_type}-embedding.json"
    
    if not os.path.exists(embedding_path):
        print(f"❌ Embedding file not found: {embedding_path}")
        return False
    
    print(f"🧪 Testing generation with {embedding_type} embeddings...")
    
    # Create a simple composition string
    comp_string = ";".join([f"{k}:{v}" for comp in compositions for k, v in comp.items()])
    
    # Build command
    cmd = [
        sys.executable, "generate_ddpm_true_ccip.py",
        "--ddpm_checkpoint", "dummy_ddpm.pth",  # This would be a real checkpoint
        "--tokenizer_only",
        "--tokenizer_embedding_path", embedding_path,
        "--custom_compositions", comp_string,
        "--num_samples", "2",
        "--batch_size", "1",
        "--output_dir", f"test_output_{embedding_type}",
        "--help"  # Just show help for now since we don't have real checkpoints
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if "--tokenizer_embedding_path" in result.stdout:
            print(f"✅ {embedding_type} embeddings: Command structure valid")
            return True
        else:
            print(f"❌ {embedding_type} embeddings: Command structure invalid")
            return False
    except Exception as e:
        print(f"❌ {embedding_type} embeddings: Error - {e}")
        return False

def main():
    """Demonstrate different embedding types."""
    print("🚀 Demonstrating DDPM + True CCIP generation with different embeddings\n")
    
    # Test compositions
    test_compositions = [
        {"Fe": 0.5, "O": 0.5},
        {"Al": 0.3, "Si": 0.4, "O": 0.3}
    ]
    
    # Available embedding types
    embedding_types = [
        "megnet16",
        "word2vec", 
        "cgcnn",
        "matscholar",
        "elem",
        "onehot"
    ]
    
    print("Available embedding types:")
    for i, emb_type in enumerate(embedding_types, 1):
        embedding_path = f"embedding/tokenizers/{emb_type}-embedding.json"
        status = "✅" if os.path.exists(embedding_path) else "❌"
        print(f"  {i}. {emb_type} {status}")
    
    print("\nExample usage commands:\n")
    
    for emb_type in embedding_types:
        embedding_path = f"embedding/tokenizers/{emb_type}-embedding.json"
        if os.path.exists(embedding_path):
            print(f"# Using {emb_type} embeddings:")
            print(f"python generate_ddpm_true_ccip.py \\")
            print(f"    --ddpm_checkpoint checkpoints/ddpm_model.pth \\")
            print(f"    --vae_checkpoint checkpoints/vae_epoch_9.pth \\")
            print(f"    --tokenizer_only \\")
            print(f"    --tokenizer_embedding_path {embedding_path} \\")
            print(f"    --custom_compositions 'Fe:0.5,O:0.5;Al:0.3,Si:0.4,O:0.3' \\")
            print(f"    --output_dir generated_samples_{emb_type} \\")
            print(f"    --annotate\n")
    
    print("💡 Tips:")
    print("- megnet16: Good general-purpose embeddings for materials")
    print("- word2vec: Learned from materials literature")
    print("- cgcnn: Crystal graph-based embeddings")
    print("- matscholar: Materials science literature embeddings")
    print("- elem: Simple elemental property embeddings")
    print("- onehot: One-hot encoded element representations")
    
    print("\n🔧 Tokenizer-only mode benefits:")
    print("- Faster startup (no True CCIP model loading)")
    print("- Lower memory usage")
    print("- Can work with any element embedding type")
    print("- Good for rapid prototyping and testing")

if __name__ == "__main__":
    main()
