"""
True CCIP Evaluation and Inference Script

This script evaluates a trained True CCIP model and provides functions for:
1. Computing embeddings for compositions and images
2. Finding similar images for given compositions
3. Finding similar compositions for given images
4. Evaluating retrieval performance

True CCIP uses a different tokenization approach where compositions are 
tokenized into element sequences with fractional values, similar to CLIP text processing.
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
import logging

# Import True CCIP modules
from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer
from sd.true_ccip_data import TrueCCIPDataset, true_ccip_collate_fn
from train_ccip_simple import ImageEncoder  # Reuse the same image encoder


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('eval_true_ccip.log'),
            logging.StreamHandler()
        ]
    )


class TrueCCIPEvaluator:
    """Evaluator for trained True CCIP models"""
    
    def __init__(self, true_ccip_model, device):
        """
        Args:
            true_ccip_model: The TrueCCIPModel instance (contains both encoders)
            device: PyTorch device
        """
        self.model = true_ccip_model
        self.device = device
        
        # Set to eval mode
        self.model.eval()
    
    def encode_compositions(self, tokens, fractions, attention_masks, latent_scale_factor=1.0):
        """Encode compositions to embedding space"""
        with torch.no_grad():
            tokens = tokens.to(self.device)
            fractions = fractions.to(self.device)
            attention_masks = attention_masks.to(self.device)
            embeddings = self.model.composition_encoder(tokens, fractions, attention_masks)
            
            # Apply latent scale factor if specified
            if latent_scale_factor != 1.0:
                embeddings = embeddings * latent_scale_factor
                
        return embeddings
    
    def encode_images(self, images, latent_scale_factor=1.0):
        """Encode images to embedding space"""
        with torch.no_grad():
            images = images.to(self.device)
            embeddings = self.model.image_encoder(images)
            
            # Apply latent scale factor to image features only (for VAE latent space scaling)
            if latent_scale_factor != 1.0:
                embeddings = embeddings * latent_scale_factor
                
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
        
        # Filter k_values to not exceed dataset size
        k_values = [k for k in k_values if k <= N]
        if not k_values:
            logging.warning(f"No valid k values for dataset size {N}")
            return {}
        
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
            logging.info(f"Recall I2T@{k}: {recall_i2t:.4f}")
        
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
            logging.info(f"Recall T2I@{k}: {recall_t2i:.4f}")
        
        return results


def load_true_ccip_checkpoint(checkpoint_path, device):
    """Load trained True CCIP model from checkpoint"""
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint with weights_only=False for compatibility with older checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model arguments from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        if isinstance(args, dict):
            args = argparse.Namespace(**args)
    else:
        # Fallback to default arguments if not saved
        logging.warning("No args found in checkpoint, using defaults")
        args = argparse.Namespace(
            vocab_size=95,
            max_length=20,
            embedding_dim=128,
            n_layers=6,
            n_heads=8,
            output_dim=512,
            image_encoder_type="resnet",
            image_size=64
        )
    
    # Import the TrueCCIPModel from training script
    from train_true_ccip_distributed import TrueCCIPModel
    
    # Initialize model
    model = TrueCCIPModel(
        vocab_size=args.vocab_size,
        max_length=args.max_length,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        output_dim=args.output_dim,
        image_encoder_type=args.image_encoder_type,
        image_size=args.image_size
    ).to(device)
    
    # Load state dict (handle DataParallel and DDP models)
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from DataParallel/DDP)
    if any(key.startswith('module.') for key in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    
    logging.info(f"Model loaded successfully")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, args


def compute_embeddings(evaluator, dataloader, latent_scale_factor=1.0):
    """Compute all embeddings for a dataset"""
    all_comp_embeddings = []
    all_img_embeddings = []
    
    logging.info("Computing embeddings for evaluation dataset...")
    
    for batch in tqdm(dataloader, desc="Computing embeddings"):
        tokens, fractions, attention_masks, images = batch
        
        # Compute embeddings
        comp_embeddings = evaluator.encode_compositions(tokens, fractions, attention_masks, latent_scale_factor)
        img_embeddings = evaluator.encode_images(images, latent_scale_factor)
        
        all_comp_embeddings.append(comp_embeddings.cpu())
        all_img_embeddings.append(img_embeddings.cpu())
    
    # Concatenate all embeddings
    all_comp_embeddings = torch.cat(all_comp_embeddings, dim=0)
    all_img_embeddings = torch.cat(all_img_embeddings, dim=0)
    
    logging.info(f"Computed embeddings shape: {all_comp_embeddings.shape}")
    
    return all_comp_embeddings, all_img_embeddings


def visualize_retrieval_results(dataset, evaluator, comp_embeddings, img_embeddings, 
                               sample_idx=0, top_k=5, save_path=None):
    """Visualize retrieval results for a sample"""
    
    # Get the sample composition embedding
    comp_embedding = comp_embeddings[sample_idx]
    
    # Find similar images
    similar_indices, similarity_scores = evaluator.find_similar_images(
        comp_embedding, img_embeddings, top_k=top_k
    )
    
    # Get composition info
    comp_dict = dataset.get_composition_dict(sample_idx)
    composition_str = ', '.join([f'{k}:{v:.2f}' for k, v in comp_dict.items()])
    
    # Create visualization
    fig, axes = plt.subplots(1, top_k + 1, figsize=(3 * (top_k + 1), 3))
    
    # Plot query composition (actual image)
    _, _, _, actual_img = dataset[sample_idx]
    if actual_img.dim() == 3:
        actual_img_pil = transforms.ToPILImage()(actual_img)
    else:
        actual_img_pil = transforms.ToPILImage()(actual_img.unsqueeze(0))
    
    axes[0].imshow(actual_img_pil)
    axes[0].set_title(f'Query\n{composition_str[:50]}...\n(Actual)', fontsize=8)
    axes[0].axis('off')
    
    # Plot retrieved images
    for i, (idx, score) in enumerate(zip(similar_indices, similarity_scores)):
        _, _, _, retrieved_img = dataset[idx]
        if retrieved_img.dim() == 3:
            retrieved_img_pil = transforms.ToPILImage()(retrieved_img)
        else:
            retrieved_img_pil = transforms.ToPILImage()(retrieved_img.unsqueeze(0))
        
        retrieved_comp_dict = dataset.get_composition_dict(idx)
        retrieved_comp_str = ', '.join([f'{k}:{v:.2f}' for k, v in retrieved_comp_dict.items()])
        
        axes[i + 1].imshow(retrieved_img_pil)
        rank_symbol = "✓" if idx == sample_idx else f"#{i+1}"
        axes[i + 1].set_title(f'{rank_symbol} Score: {score:.3f}\n{retrieved_comp_str[:50]}...', fontsize=8)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f"Visualization saved to: {save_path}")
    
    plt.show()


def load_test_indices(test_indices_path):
    """Load test indices from file"""
    if not os.path.exists(test_indices_path):
        logging.warning(f"Test indices file not found: {test_indices_path}")
        return None
    
    with open(test_indices_path, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines()]
    
    logging.info(f"Loaded {len(indices)} test indices")
    return indices


def main():
    parser = argparse.ArgumentParser(description='Evaluate True CCIP model')
    
    # Model arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained True CCIP model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/dataset_comp_image_spectra.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--test_indices', type=str, default='splits/split_test_indices.txt',
                       help='Path to test indices file')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--max_length', type=int, default=20,
                       help='Maximum sequence length for compositions')
    
    # Visualization arguments
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--vis_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top similar items to show')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                       help='Directory to save evaluation results')
    
    # Data subset arguments
    parser.add_argument('--eval_subset', type=int, default=None,
                       help='Evaluate on a subset of data (for testing)')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Start index for evaluation subset')
    parser.add_argument('--latent_scale_factor', type=float, default=None,
                       help='Override latent scale factor (if not specified, uses value from checkpoint)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logging.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model, model_args = load_true_ccip_checkpoint(args.checkpoint_path, device)
    evaluator = TrueCCIPEvaluator(model, device)
    
    # Load test indices
    test_indices = load_test_indices(args.test_indices)
    
    # Create transform for images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((model_args.image_size, model_args.image_size), antialias=True),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # Load dataset
    logging.info("Loading dataset...")
    
    if test_indices is not None:
        # Use specific test indices
        dataset = TrueCCIPDataset(
            h5_path=args.data_path,
            transform=transform,
            max_length=args.max_length,
            start=0,
            end=None
        )
        
        # Create subset based on test indices
        if args.eval_subset:
            test_indices = test_indices[:args.eval_subset]
        
        # Use torch subset
        from torch.utils.data import Subset
        test_dataset = Subset(dataset, test_indices)
        
    else:
        # Use a subset of the full dataset
        end_idx = args.eval_subset if args.eval_subset else 1000
        dataset = TrueCCIPDataset(
            h5_path=args.data_path,
            transform=transform,
            max_length=args.max_length,
            start=args.start_idx,
            end=args.start_idx + end_idx
        )
        test_dataset = dataset
    
    logging.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=true_ccip_collate_fn
    )
    
    # Compute embeddings
    latent_scale_factor = args.latent_scale_factor if args.latent_scale_factor is not None else getattr(model_args, 'latent_scale_factor', 1.0)
    logging.info(f"Using latent scale factor: {latent_scale_factor}")
    
    comp_embeddings, img_embeddings = compute_embeddings(evaluator, test_loader, latent_scale_factor)
    
    # Evaluate retrieval performance
    logging.info("Evaluating retrieval performance...")
    recall_results = evaluator.evaluate_retrieval(
        comp_embeddings.to(device), 
        img_embeddings.to(device), 
        k_values=[1, 5, 10, 20]
    )
    
    # Print results
    print("\n" + "="*60)
    print("TRUE CCIP RETRIEVAL EVALUATION RESULTS")
    print("="*60)
    
    for metric, score in recall_results.items():
        print(f"{metric:20s}: {score:.4f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, 'true_ccip_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(recall_results, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")
    
    # Save model info
    model_info = {
        'checkpoint_path': args.checkpoint_path,
        'model_args': vars(model_args),
        'eval_args': vars(args),
        'test_dataset_size': len(test_dataset),
        'device': str(device)
    }
    
    info_file = os.path.join(args.output_dir, 'model_info.json')
    with open(info_file, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    # Visualization
    if args.visualize and hasattr(test_dataset, 'dataset'):
        logging.info(f"Generating visualizations for {args.vis_samples} samples...")
        
        # Get the base dataset for visualization
        base_dataset = test_dataset.dataset if hasattr(test_dataset, 'dataset') else test_dataset
        
        # Visualize a few samples
        for i in range(min(args.vis_samples, len(test_dataset))):
            vis_path = os.path.join(args.output_dir, f'true_ccip_retrieval_sample_{i}.png')
            
            # Get the actual dataset index
            actual_idx = test_indices[i] if test_indices else i
            
            try:
                visualize_retrieval_results(
                    base_dataset, evaluator,
                    comp_embeddings, img_embeddings,
                    sample_idx=i, top_k=args.top_k, save_path=vis_path
                )
            except Exception as e:
                logging.error(f"Error creating visualization {i}: {e}")
    
    logging.info("✅ True CCIP evaluation completed!")


if __name__ == '__main__':
    main()
