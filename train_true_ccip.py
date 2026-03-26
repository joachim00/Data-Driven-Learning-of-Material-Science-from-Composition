#!/usr/bin/env python3
"""
Distributed True CCIP Training Script

This script trains a True CLIP-style CCIP model using contrastive learning
between chemical compositions and material images.

Features:
- YAML configuration support
- Projection heads for better modality alignment
- Optional positional encoding
- Learnable temperature option
- Warmup scheduler with cosine annealing
- Distributed training support (DDP)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm
import logging
import yaml
from datetime import datetime
import numpy as np
import math

# Import custom modules
from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer
from sd.true_ccip_data import TrueCCIPDataset, true_ccip_collate_fn
from sd.data_helper import indice_import
import torch.nn.functional as F

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """
    Projection head to map encoder outputs to shared embedding space.
    This helps align different modalities in contrastive learning.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ])
            current_dim = hidden_dim
        
        # Final layer without activation
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.projection(x)


class ImageEncoder(nn.Module):
    """Image encoder for contrastive learning with True CCIP."""
    
    def __init__(self, encoder_type="resnet", image_size=64, hidden_dim=512):
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        
        if encoder_type == "resnet":
            self.encoder = nn.Sequential(
                # Conv block 1
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                # Conv block 2
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                # Conv block 3
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                # Conv block 4
                nn.Conv2d(256, hidden_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                
                # Global average pooling
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    def forward(self, images):
        """Returns raw features (without normalization - done after projection)."""
        return self.encoder(images)


class TrueCCIPEmbeddingNoPos(nn.Module):
    """
    Embedding layer for CCIP without positional encoding.
    For compositions, order doesn't matter (set-like), so positional encoding is optional.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Element token embeddings
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Fraction projection
        self.fraction_projection = nn.Linear(1, embedding_dim)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, tokens: torch.LongTensor, fractions: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, seq_len = tokens.shape
        
        # Element token embeddings
        token_emb = self.token_embedding(tokens)
        
        # Fraction embeddings
        frac_emb = self.fraction_projection(fractions.unsqueeze(-1))
        
        # Combine embeddings (no positional encoding)
        embeddings = token_emb + frac_emb
        
        # Layer norm
        embeddings = self.layer_norm(embeddings)
        
        return embeddings


class CompositionEncoder(nn.Module):
    """
    Composition encoder with optional positional encoding.
    """
    
    def __init__(self, 
                 vocab_size: int = 98,
                 embedding_dim: int = 256,
                 max_length: int = 20,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 dropout: float = 0.1,
                 use_positional_encoding: bool = False):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_positional_encoding = use_positional_encoding
        
        # Embedding layer
        if use_positional_encoding:
            from sd.true_ccip import TrueCCIPEmbedding
            self.embedding = TrueCCIPEmbedding(vocab_size, embedding_dim, max_length)
        else:
            self.embedding = TrueCCIPEmbeddingNoPos(vocab_size, embedding_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(embedding_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, tokens: torch.LongTensor, fractions: torch.FloatTensor, 
                attention_mask: torch.BoolTensor) -> torch.FloatTensor:
        # Get embeddings
        x = self.embedding(tokens, fractions)
        
        # Convert mask for transformer (True = ignore)
        src_key_padding_mask = ~attention_mask
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Final norm
        x = self.final_norm(x)
        
        # Global pooling: mean over non-masked positions
        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return x


class TrueCCIPModelV2(nn.Module):
    """
    Improved True CCIP model with projection heads for better modality alignment.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        comp_config = config['model']['composition']
        img_config = config['model']['image']
        proj_config = config['model']['projection']
        
        # Composition encoder
        self.composition_encoder = CompositionEncoder(
            vocab_size=comp_config['vocab_size'] + 3,  # +3 for special tokens
            embedding_dim=comp_config['embedding_dim'],
            max_length=comp_config['max_length'],
            n_layers=comp_config['n_layers'],
            n_heads=comp_config['n_heads'],
            dropout=comp_config.get('dropout', 0.1),
            use_positional_encoding=comp_config.get('use_positional_encoding', False)
        )
        
        # Image encoder
        self.image_encoder = ImageEncoder(
            encoder_type=img_config['encoder_type'],
            image_size=img_config['image_size'],
            hidden_dim=img_config['hidden_dim']
        )
        
        # Projection heads
        use_projection = proj_config.get('use_projection_head', True)
        output_dim = proj_config['output_dim']
        hidden_dim = proj_config.get('hidden_dim', output_dim)
        num_layers = proj_config.get('num_layers', 2)
        
        if use_projection:
            self.comp_projection = ProjectionHead(
                input_dim=comp_config['embedding_dim'],
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers
            )
            self.img_projection = ProjectionHead(
                input_dim=img_config['hidden_dim'],
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers
            )
        else:
            # Simple linear projection
            self.comp_projection = nn.Linear(comp_config['embedding_dim'], output_dim)
            self.img_projection = nn.Linear(img_config['hidden_dim'], output_dim)
        
        # Learnable temperature (optional)
        self.learnable_temperature = config['training'].get('learnable_temperature', False)
        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(torch.zeros(1))
    
    def get_temperature(self, default_temp: float) -> float:
        if self.learnable_temperature:
            return torch.exp(self.log_temperature).clamp(min=0.01, max=100.0)
        return default_temp
    
    def forward(self, tokens, fractions, attention_mask, images):
        # Encode
        comp_features = self.composition_encoder(tokens, fractions, attention_mask)
        img_features = self.image_encoder(images)
        
        # Project to shared space
        comp_features = self.comp_projection(comp_features)
        img_features = self.img_projection(img_features)
        
        # L2 normalize
        comp_features = F.normalize(comp_features, p=2, dim=-1)
        img_features = F.normalize(img_features, p=2, dim=-1)
        
        return comp_features, img_features


def clip_loss(comp_features, img_features, temperature=0.07, label_smoothing=0.0):
    """
    CLIP-style contrastive loss with optional label smoothing.
    """
    batch_size = comp_features.size(0)
    
    # Compute similarity matrix
    logits = torch.matmul(comp_features, img_features.T) / temperature
    
    # Labels
    labels = torch.arange(batch_size, device=logits.device)
    
    # Apply label smoothing if specified
    if label_smoothing > 0:
        loss_comp_to_img = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        loss_img_to_comp = F.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)
    else:
        loss_comp_to_img = F.cross_entropy(logits, labels)
        loss_img_to_comp = F.cross_entropy(logits.T, labels)
    
    loss = (loss_comp_to_img + loss_img_to_comp) / 2
    
    # Compute accuracy for logging
    with torch.no_grad():
        pred_comp_to_img = logits.argmax(dim=1)
        pred_img_to_comp = logits.T.argmax(dim=1)
        acc_comp_to_img = (pred_comp_to_img == labels).float().mean()
        acc_img_to_comp = (pred_img_to_comp == labels).float().mean()
        accuracy = (acc_comp_to_img + acc_img_to_comp) / 2
    
    return loss, accuracy


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=0):
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, optimizer, scheduler, scaler, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    training_config = config['training']
    logging_config = config['logging']
    
    temperature = training_config['temperature']
    label_smoothing = training_config.get('label_smoothing', 0.0)
    
    if dist.get_rank() == 0:
        dataloader = tqdm(dataloader, desc=f"Epoch {epoch+1}/{training_config['num_epochs']}")
    
    for batch_idx, (tokens, fractions, attention_mask, images) in enumerate(dataloader):
        tokens = tokens.cuda(non_blocking=True)
        fractions = fractions.cuda(non_blocking=True)
        attention_mask = attention_mask.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.autocast(device_type='cuda', enabled=training_config['use_amp']):
            comp_features, img_features = model(tokens, fractions, attention_mask, images)
            
            # Get temperature (learnable or fixed)
            if hasattr(model, 'module'):
                temp = model.module.get_temperature(temperature)
            else:
                temp = model.get_temperature(temperature)
            
            loss, accuracy = clip_loss(comp_features, img_features, temp, label_smoothing)
        
        if training_config['use_amp']:
            scaler.scale(loss).backward()
            if training_config['grad_clip_norm'] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['grad_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if training_config['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['grad_clip_norm'])
            optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_acc += accuracy.item()
        num_batches += 1
        
        if batch_idx % logging_config['log_every_n_batches'] == 0 and dist.get_rank() == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(dataloader)}] "
                       f"Loss: {loss.item():.4f} Acc: {accuracy.item():.4f} LR: {current_lr:.2e}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
    
    if dist.is_initialized():
        loss_tensor = torch.tensor([avg_loss, avg_acc], device=torch.cuda.current_device())
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss, avg_acc = loss_tensor[0].item(), loss_tensor[1].item()
    
    return avg_loss, avg_acc


def evaluate(model, dataloader, config):
    """Evaluate the model."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    temperature = config['training']['temperature']
    
    with torch.no_grad():
        for tokens, fractions, attention_mask, images in dataloader:
            tokens = tokens.cuda(non_blocking=True)
            fractions = fractions.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
            
            comp_features, img_features = model(tokens, fractions, attention_mask, images)
            
            if hasattr(model, 'module'):
                temp = model.module.get_temperature(temperature)
            else:
                temp = model.get_temperature(temperature)
            
            loss, accuracy = clip_loss(comp_features, img_features, temp)
            
            total_loss += loss.item()
            total_acc += accuracy.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
    
    if dist.is_initialized():
        metrics_tensor = torch.tensor([avg_loss, avg_acc], device=torch.cuda.current_device())
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.AVG)
        avg_loss, avg_acc = metrics_tensor[0].item(), metrics_tensor[1].item()
    
    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, acc, config, is_best=False):
    """Save model checkpoint."""
    if dist.get_rank() != 0:
        return
    
    output_dir = config['logging']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc,
        'config': config,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    checkpoint_path = os.path.join(output_dir, f'true_ccip_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    if is_best:
        best_path = os.path.join(output_dir, 'true_ccip_best.pth')
        torch.save(checkpoint, best_path)
        logger.info(f"Best checkpoint saved: {best_path}")


def main():
    parser = argparse.ArgumentParser(description='Distributed True CCIP Training v2')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup distributed training
    local_rank = setup_distributed()
    device = torch.cuda.current_device()
    
    # Set seed
    seed = config['system'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create output directory
    if dist.get_rank() == 0:
        os.makedirs(config['logging']['output_dir'], exist_ok=True)
        # Save config
        config_save_path = os.path.join(config['logging']['output_dir'], 'config.yaml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Config saved to {config_save_path}")
    
    # Setup data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config['model']['image']['image_size'],) * 2, antialias=True),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    train_indices = indice_import(config['data']['train_indices'])
    test_indices = indice_import(config['data']['test_indices'])
    
    train_dataset = TrueCCIPDataset(
        h5_path=config['data']['data_path'],
        transform=transform,
        max_length=config['model']['composition']['max_length'],
        start=0, end=None
    )
    
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_indices)
    test_dataset = Subset(
        TrueCCIPDataset(
            h5_path=config['data']['data_path'],
            transform=transform,
            max_length=config['model']['composition']['max_length'],
            start=0, end=None
        ),
        test_indices
    )
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=train_sampler,
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        collate_fn=true_ccip_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        sampler=test_sampler,
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        collate_fn=true_ccip_collate_fn
    )
    
    # Create model
    model = TrueCCIPModelV2(config).cuda()
    
    # Wrap in DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=config['system'].get('find_unused_parameters', False),
        gradient_as_bucket_view=True
    )
    
    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable)")
        logger.info(f"Positional encoding: {config['model']['composition'].get('use_positional_encoding', False)}")
        logger.info(f"Projection head: {config['model']['projection'].get('use_projection_head', True)}")
    
    # Create optimizer
    training_config = config['training']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(training_config['learning_rate']),
        weight_decay=float(training_config['weight_decay'])
    )
    
    # Create scheduler
    num_epochs = training_config['num_epochs']
    warmup_epochs = training_config.get('warmup_epochs', 0)
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr=training_config.get('min_lr', 0)
    )
    
    scaler = torch.cuda.amp.GradScaler() if training_config['use_amp'] else None
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume and os.path.exists(args.resume):
        if dist.get_rank() == 0:
            logger.info(f"Resuming from: {args.resume}")
        
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('loss', float('inf'))
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Training loop
    logging_config = config['logging']
    
    if dist.get_rank() == 0:
        logger.info(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, config)
        
        if (epoch + 1) % logging_config['eval_every'] == 0:
            test_loss, test_acc = evaluate(model, test_loader, config)
            
            if dist.get_rank() == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] "
                           f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                           f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")
            
            is_best = test_loss < best_loss
            if is_best:
                best_loss = test_loss
            
            if (epoch + 1) % logging_config['save_every'] == 0 or is_best:
                save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, test_loss, test_acc, config, is_best)
    
    if dist.get_rank() == 0:
        save_checkpoint(model, optimizer, scheduler, scaler, num_epochs, best_loss, 0, config)
        logger.info("Training completed!")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
