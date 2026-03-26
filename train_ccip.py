"""
CCIP Training Script - Composition-Constrained Image-Property Learning

This script trains a CLIP-like model where:
- Compositions are encoded using element fractions and embeddings
- Images are encoded using a CNN encoder
- The model learns to align composition and image representations in a shared embedding space

Similar to CLIP but for materials science: composition ↔ image alignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import os
import logging
from tqdm import tqdm
import wandb
from PIL import Image
from torchvision import transforms
import json
import random
from typing import Dict, List, Tuple, Optional

from sd.ccip import CCIP
from sd.data_helper import HDF5CompositionImageDataset
from embedding.element_tokenizer import CompositionTokenizer


class ImageEncoder(nn.Module):
    """CNN-based image encoder similar to CLIP's vision encoder"""
    def __init__(self, output_dim=512):
        super().__init__()
        
        # Simple CNN architecture - can be replaced with ResNet, ViT, etc.
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection = nn.Linear(512, output_dim)
        self.output_dim = output_dim
    
    def forward(self, images):
        # images shape: (batch_size, 3, H, W)
        features = self.conv_layers(images)  # (batch_size, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 512)
        
        # Project to shared embedding space
        output = self.projection(features)  # (batch_size, output_dim)
        
        # Normalize for contrastive learning
        output = F.normalize(output, p=2, dim=-1)
        
        return output


class CompositionDataset(Dataset):
    """Dataset for CCIP training that provides composition-image pairs"""
    
    def __init__(self, hdf5_dataset, tokenizer, indices=None, transform=None, max_length=20):
        self.dataset = hdf5_dataset
        self.tokenizer = tokenizer
        self.indices = indices if indices is not None else list(range(len(hdf5_dataset)))
        self.transform = transform
        self.max_length = max_length
        
        # Define image transforms
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        
        # Get composition and image from dataset
        composition_embedding, image_tensor, spectra = self.dataset[data_idx]
        
        # Get composition dictionary for tokenization
        composition_dict = self.dataset.get_composition_dict(data_idx)
        
        # Tokenize composition
        tokens = self.tokenizer.tokenize_composition(composition_dict)
        
        # Pad/truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))  # Pad with zeros
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (tokens != 0).float()
        
        # Transform image
        if len(image_tensor.shape) == 3:
            image = self.transform(image_tensor)
        else:
            # Handle different tensor formats
            image = self.transform(image_tensor.squeeze())
        
        return {
            'composition_tokens': tokens,
            'attention_mask': attention_mask,
            'image': image,
            'composition_dict': composition_dict,
            'index': data_idx
        }


def contrastive_loss(composition_features, image_features, temperature=0.07):
    """
    Compute contrastive loss similar to CLIP
    
    Args:
        composition_features: (batch_size, embed_dim) - normalized composition embeddings
        image_features: (batch_size, embed_dim) - normalized image embeddings
        temperature: Temperature parameter for softmax
    """
    batch_size = composition_features.shape[0]
    
    # Compute similarity matrix
    logits_per_composition = torch.matmul(composition_features, image_features.T) / temperature
    logits_per_image = logits_per_composition.T
    
    # Labels - diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=composition_features.device)
    
    # Contrastive loss - symmetric
    loss_composition = F.cross_entropy(logits_per_composition, labels)
    loss_image = F.cross_entropy(logits_per_image, labels)
    
    total_loss = (loss_composition + loss_image) / 2
    
    return total_loss, logits_per_composition, logits_per_image


def compute_accuracy(logits, labels):
    """Compute top-1 accuracy"""
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == labels).float().mean()
    return accuracy


def train_epoch(model, image_encoder, dataloader, optimizer, device, epoch, args):
    """Train for one epoch"""
    model.train()
    image_encoder.train()
    
    total_loss = 0
    total_comp_acc = 0
    total_img_acc = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        # Move data to device
        composition_tokens = batch['composition_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        images = batch['image'].to(device)
        
        batch_size = composition_tokens.shape[0]
        labels = torch.arange(batch_size, device=device)
        
        # Forward pass
        composition_features = model(composition_tokens, attention_mask)
        image_features = image_encoder(images)
        
        # Compute contrastive loss
        loss, logits_comp, logits_img = contrastive_loss(
            composition_features, image_features, args.temperature
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute metrics
        comp_acc = compute_accuracy(logits_comp, labels)
        img_acc = compute_accuracy(logits_img, labels)
        
        total_loss += loss.item()
        total_comp_acc += comp_acc.item()
        total_img_acc += img_acc.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'comp_acc': f'{comp_acc.item():.3f}',
            'img_acc': f'{img_acc.item():.3f}'
        })
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_comp_acc': comp_acc.item(),
                'train/batch_img_acc': img_acc.item(),
                'train/step': epoch * len(dataloader) + num_batches
            })
    
    avg_loss = total_loss / num_batches
    avg_comp_acc = total_comp_acc / num_batches
    avg_img_acc = total_img_acc / num_batches
    
    return avg_loss, avg_comp_acc, avg_img_acc


def validate(model, image_encoder, dataloader, device, epoch, args):
    """Validate the model"""
    model.eval()
    image_encoder.eval()
    
    total_loss = 0
    total_comp_acc = 0
    total_img_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Move data to device
            composition_tokens = batch['composition_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            images = batch['image'].to(device)
            
            batch_size = composition_tokens.shape[0]
            labels = torch.arange(batch_size, device=device)
            
            # Forward pass
            composition_features = model(composition_tokens, attention_mask)
            image_features = image_encoder(images)
            
            # Compute contrastive loss
            loss, logits_comp, logits_img = contrastive_loss(
                composition_features, image_features, args.temperature
            )
            
            # Compute metrics
            comp_acc = compute_accuracy(logits_comp, labels)
            img_acc = compute_accuracy(logits_img, labels)
            
            total_loss += loss.item()
            total_comp_acc += comp_acc.item()
            total_img_acc += img_acc.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_comp_acc = total_comp_acc / num_batches
    avg_img_acc = total_img_acc / num_batches
    
    return avg_loss, avg_comp_acc, avg_img_acc


def load_data_splits(train_indices_path, test_indices_path):
    """Load train/test indices"""
    def load_indices(path):
        with open(path, 'r') as f:
            return [int(line.strip()) for line in f.readlines()]
    
    train_indices = load_indices(train_indices_path)
    test_indices = load_indices(test_indices_path)
    
    return train_indices, test_indices


def save_checkpoint(model, image_encoder, optimizer, epoch, loss, args, filename):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'image_encoder_state_dict': image_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description='Train CCIP model')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to HDF5 dataset')
    parser.add_argument('--embedding_name', type=str, default='elem',
                       help='Embedding name for tokenizer')
    parser.add_argument('--train_indices', type=str, required=True,
                       help='Path to train indices file')
    parser.add_argument('--test_indices', type=str, required=True,
                       help='Path to test indices file')
    
    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=95,
                       help='Vocabulary size for composition tokenizer')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension for composition encoder')
    parser.add_argument('--output_dim', type=int, default=512,
                       help='Output dimension for both encoders')
    parser.add_argument('--max_length', type=int, default=20,
                       help='Maximum sequence length for compositions')
    parser.add_argument('--n_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                       help='Number of attention heads')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--temperature', type=float, default=0.07,
                       help='Temperature for contrastive loss')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='checkpoints/ccip',
                       help='Output directory for checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='ccip-training',
                       help='W&B project name')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Load data
    logging.info("Loading dataset...")
    tokenizer = CompositionTokenizer(f'embedding/tokenizers/{args.embedding_name}-embedding.json')
    hdf5_dataset = HDF5CompositionImageDataset(args.data_path, tokenizer=tokenizer)
    
    # Load data splits
    train_indices, test_indices = load_data_splits(args.train_indices, args.test_indices)
    logging.info(f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    
    # Create datasets
    train_dataset = CompositionDataset(
        hdf5_dataset, tokenizer, indices=train_indices, max_length=args.max_length
    )
    val_dataset = CompositionDataset(
        hdf5_dataset, tokenizer, indices=test_indices, max_length=args.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Initialize models
    logging.info("Initializing models...")
    composition_encoder = CCIP(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        max_length=args.max_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        output_dim=args.output_dim
    ).to(device)
    
    image_encoder = ImageEncoder(output_dim=args.output_dim).to(device)
    
    # Setup optimizer
    all_params = list(composition_encoder.parameters()) + list(image_encoder.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Training loop
    logging.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_comp_acc, train_img_acc = train_epoch(
            composition_encoder, image_encoder, train_loader, optimizer, device, epoch, args
        )
        
        # Validate
        val_loss, val_comp_acc, val_img_acc = validate(
            composition_encoder, image_encoder, val_loader, device, epoch, args
        )
        
        # Log epoch results
        logging.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Train Comp Acc: {train_comp_acc:.3f} | Train Img Acc: {train_img_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Comp Acc: {val_comp_acc:.3f} | Val Img Acc: {val_img_acc:.3f}"
        )
        
        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/comp_acc': train_comp_acc,
                'train/img_acc': train_img_acc,
                'val/loss': val_loss,
                'val/comp_acc': val_comp_acc,
                'val/img_acc': val_img_acc,
            })
        
        # Save checkpoint
        if epoch % args.save_every == 0 or val_loss < best_val_loss:
            checkpoint_path = os.path.join(args.output_dir, f'ccip_epoch_{epoch}.pth')
            save_checkpoint(composition_encoder, image_encoder, optimizer, epoch, val_loss, args, checkpoint_path)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(args.output_dir, 'ccip_best.pth')
                save_checkpoint(composition_encoder, image_encoder, optimizer, epoch, val_loss, args, best_checkpoint_path)
    
    logging.info("Training completed!")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
