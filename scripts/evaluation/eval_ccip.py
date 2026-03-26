"""
CCIP Evaluation and Inference Script

This script evaluates a trained CCIP model and provides functions for:
1. Computing embeddings for compositions and images
2. Finding similar images for given compositions
3. Finding similar compositions for given images
4. Evaluating retrieval performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from train_ccip_simple import ImageEncoder, CompositionDataset, load_data_splits
from sd.ccip import CCIP
from sd.data_helper import HDF5CompositionImageDataset
from embedding.element_tokenizer import CompositionTokenizer


def custom_collate_fn(batch):
    """Custom collate function to handle composition dictionaries"""
    # Separate the composition_dict from other fields
    composition_tokens = torch.stack([item['composition_tokens'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    images = torch.stack([item['image'] for item in batch])
    composition_dicts = [item['composition_dict'] for item in batch]  # Keep as list
    indices = torch.tensor([item['index'] for item in batch])
    
    return {
        'composition_tokens': composition_tokens,
        'attention_mask': attention_mask,
        'image': images,
        'composition_dict': composition_dicts,  # List of dicts
        'index': indices
    }


class CCIPEvaluator:
    """Evaluator for trained CCIP models"""
    
    def __init__(self, composition_encoder, image_encoder, device):
        self.composition_encoder = composition_encoder
        self.image_encoder = image_encoder
        self.device = device
        
        # Set to eval mode
        self.composition_encoder.eval()
        self.image_encoder.eval()
    
    def encode_compositions(self, composition_tokens, attention_masks):
        """Encode compositions to embedding space"""
        with torch.no_grad():
            composition_tokens = composition_tokens.to(self.device)
            attention_masks = attention_masks.to(self.device)
            embeddings = self.composition_encoder(composition_tokens, attention_masks)
        return embeddings
    
    def encode_images(self, images):
        """Encode images to embedding space"""
        with torch.no_grad():
            images = images.to(self.device)
            embeddings = self.image_encoder(images)
        return embeddings
    
    def compute_similarity(self, comp_embeddings, img_embeddings):
        """Compute cosine similarity between composition and image embeddings"""
        # Both embeddings are already normalized in the models
        similarities = torch.matmul(comp_embeddings, img_embeddings.T)
        return similarities
    
    def find_similar_images(self, composition_embedding, image_embeddings, top_k=5):
        """Find top-k most similar images for a given composition"""
        similarities = torch.matmul(composition_embedding.unsqueeze(0), image_embeddings.T)
        similarities = similarities.squeeze(0)
        
        top_k_indices = torch.topk(similarities, k=top_k, dim=0).indices
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()
    
    def find_similar_compositions(self, image_embedding, composition_embeddings, top_k=5):
        """Find top-k most similar compositions for a given image"""
        similarities = torch.matmul(image_embedding.unsqueeze(0), composition_embeddings.T)
        similarities = similarities.squeeze(0)
        
        top_k_indices = torch.topk(similarities, k=top_k, dim=0).indices
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices.cpu().numpy(), top_k_scores.cpu().numpy()
    
    def evaluate_retrieval(self, comp_embeddings, img_embeddings, k_values=[1, 5, 10]):
        """
        Evaluate retrieval performance using Recall@K
        
        Args:
            comp_embeddings: (N, embed_dim) composition embeddings
            img_embeddings: (N, embed_dim) image embeddings  
            k_values: List of k values for Recall@K computation
        
        Returns:
            Dictionary with recall scores
        """
        N = comp_embeddings.shape[0]
        
        # Compute similarity matrix
        similarities = self.compute_similarity(comp_embeddings, img_embeddings)  # (N, N)
        
        results = {}
        
        # Image retrieval (composition -> image)
        for k in k_values:
            correct = 0
            for i in range(N):
                # Get top-k most similar images for composition i
                top_k_indices = torch.topk(similarities[i], k=k, dim=0).indices
                # Check if the correct image (index i) is in top-k
                if i in top_k_indices:
                    correct += 1
            
            recall_i2t = correct / N
            results[f'recall_i2t@{k}'] = recall_i2t
        
        # Composition retrieval (image -> composition)
        similarities_t = similarities.T  # (N, N)
        for k in k_values:
            correct = 0
            for i in range(N):
                # Get top-k most similar compositions for image i
                top_k_indices = torch.topk(similarities_t[i], k=k, dim=0).indices
                # Check if the correct composition (index i) is in top-k
                if i in top_k_indices:
                    correct += 1
            
            recall_t2i = correct / N
            results[f'recall_t2i@{k}'] = recall_t2i
        
        return results


def load_checkpoint(checkpoint_path, device):
    """Load trained CCIP model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = argparse.Namespace(**checkpoint['args'])
    
    # Initialize models
    composition_encoder = CCIP(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        output_dim=args.output_dim
    ).to(device)
    
    image_encoder = ImageEncoder(output_dim=args.output_dim).to(device)
    
    # Load state dicts (handle DataParallel models)
    try:
        composition_encoder.load_state_dict(checkpoint['model_state_dict'])
        image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
    except RuntimeError:
        # Handle models saved with DataParallel (remove 'module.' prefix)
        model_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        image_state = {k.replace('module.', ''): v for k, v in checkpoint['image_encoder_state_dict'].items()}
        composition_encoder.load_state_dict(model_state)
        image_encoder.load_state_dict(image_state)
    
    return composition_encoder, image_encoder, args


def compute_embeddings(evaluator, dataloader):
    """Compute all embeddings for a dataset"""
    all_comp_embeddings = []
    all_img_embeddings = []
    all_indices = []
    
    for batch in tqdm(dataloader, desc="Computing embeddings"):
        composition_tokens = batch['composition_tokens']
        attention_mask = batch['attention_mask']
        images = batch['image']
        indices = batch['index']
        
        # Compute embeddings
        comp_embeddings = evaluator.encode_compositions(composition_tokens, attention_mask)
        img_embeddings = evaluator.encode_images(images)
        
        all_comp_embeddings.append(comp_embeddings.cpu())
        all_img_embeddings.append(img_embeddings.cpu())
        all_indices.extend(indices.tolist())
    
    # Concatenate all embeddings
    all_comp_embeddings = torch.cat(all_comp_embeddings, dim=0)
    all_img_embeddings = torch.cat(all_img_embeddings, dim=0)
    
    return all_comp_embeddings, all_img_embeddings, all_indices


def visualize_retrieval_results(dataset, tokenizer, evaluator, comp_embeddings, img_embeddings, 
                               indices, sample_idx=0, top_k=5, save_path=None):
    """Visualize retrieval results for a sample"""
    
    # Get the sample composition embedding
    comp_embedding = comp_embeddings[sample_idx]
    actual_index = indices[sample_idx]
    
    # Find similar images
    similar_indices, similarity_scores = evaluator.find_similar_images(
        comp_embedding, img_embeddings, top_k=top_k
    )
    
    # Get composition info
    comp_dict = dataset.get_composition_dict(actual_index)
    composition_str = ', '.join([f'{k}:{v:.2f}' for k, v in comp_dict.items()])
    
    # Create visualization
    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
    
    # Plot query composition (actual image)
    _, actual_img = dataset[actual_index]
    actual_img_pil = transforms.ToPILImage()(actual_img)
    axes[0].imshow(actual_img_pil)
    axes[0].set_title(f'Query\n{composition_str[:50]}...\n(Actual)', fontsize=8)
    axes[0].axis('off')
    
    # Plot retrieved images
    for i, (idx, score) in enumerate(zip(similar_indices, similarity_scores)):
        retrieved_index = indices[idx]
        _, retrieved_img = dataset[retrieved_index]
        retrieved_img_pil = transforms.ToPILImage()(retrieved_img)
        
        retrieved_comp_dict = dataset.get_composition_dict(retrieved_index)
        retrieved_comp_str = ', '.join([f'{k}:{v:.2f}' for k, v in retrieved_comp_dict.items()])
        
        axes[i + 1].imshow(retrieved_img_pil)
        rank_symbol = "✓" if idx == sample_idx else f"#{i+1}"
        axes[i + 1].set_title(f'{rank_symbol} Score: {score:.3f}\n{retrieved_comp_str[:50]}...', fontsize=8)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Evaluate CCIP model')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/dataset_comp_image_spectra.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--embedding_name', type=str, default='elem',
                       help='Embedding name for tokenizer')
    parser.add_argument('--test_indices', type=str, default='splits/split_test_indices.txt',
                       help='Path to test indices file')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--vis_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top similar items to show')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    composition_encoder, image_encoder, model_args = load_checkpoint(args.checkpoint_path, device)
    evaluator = CCIPEvaluator(composition_encoder, image_encoder, device)
    
    # Load data
    print("Loading dataset...")
    tokenizer = CompositionTokenizer(f'embedding/tokenizers/{args.embedding_name}-embedding.json')
    hdf5_dataset = HDF5CompositionImageDataset(args.data_path, tokenizer=tokenizer)
    
    # Load test indices
    with open(args.test_indices, 'r') as f:
        test_indices = [int(line.strip()) for line in f.readlines()]
    
    print(f"Test samples: {len(test_indices)}")
    
    # Create test dataset and dataloader
    test_dataset = CompositionDataset(
        hdf5_dataset, tokenizer, indices=test_indices, 
        max_length=model_args.max_length
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    # Compute embeddings
    print("Computing embeddings...")
    comp_embeddings, img_embeddings, dataset_indices = compute_embeddings(evaluator, test_loader)
    
    print(f"Computed embeddings: {comp_embeddings.shape}")
    
    # Evaluate retrieval performance
    print("Evaluating retrieval performance...")
    recall_results = evaluator.evaluate_retrieval(
        comp_embeddings.to(device), img_embeddings.to(device), 
        k_values=[1, 5, 10, 20]
    )
    
    # Print results
    print("\n" + "="*50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*50)
    
    for metric, score in recall_results.items():
        print(f"{metric:20s}: {score:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(recall_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Visualization
    if args.visualize:
        print(f"\nGenerating visualizations for {args.vis_samples} samples...")
        
        # Visualize a few samples
        for i in range(min(args.vis_samples, len(dataset_indices))):
            vis_path = os.path.join(args.output_dir, f'retrieval_sample_{i}.png')
            visualize_retrieval_results(
                hdf5_dataset, tokenizer, evaluator,
                comp_embeddings, img_embeddings, dataset_indices,
                sample_idx=i, top_k=args.top_k, save_path=vis_path
            )
    
    print("\n✅ Evaluation completed!")


if __name__ == '__main__':
    main()
