import argparse
import torch
from torch.nn import DataParallel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import os
import logging
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

# Import your VAE components
from sd.data_helper import HDF5CompositionImageDataset, indice_import # Assuming this is your dataset
from sd.encoder import VAE_Encoder #
from sd.decoder import VAE_Decoder #
from embedding.element_tokenizer import CompositionTokenizer

def train_vae(batch_size, learning_rate, num_epochs, kl_weight, dataset_path, tokenizer_path, train_indices_path, checkpoint_dir, load_checkpoint_path=None, patience=5, min_delta=0.0001, verbose=False, latent_scaling_factor=0.18215):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{checkpoint_dir}/training_log_{timestamp}.txt"
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting VAE training...")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Latent scaling factor: {latent_scaling_factor}")
    
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info("Using CUDA (NVIDIA GPU)")
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
        logger.info("Using MPS (Apple Silicon)")
    else:
        DEVICE = "cpu"
        logger.info("Using CPU")
    
    # Early stopping variables
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    vae_encoder = VAE_Encoder().to(DEVICE)
    vae_decoder = VAE_Decoder().to(DEVICE)

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training.")
        vae_encoder = DataParallel(vae_encoder)
        vae_decoder = DataParallel(vae_decoder)

    vae_params = list(vae_encoder.parameters()) + list(vae_decoder.parameters())
    optimizer_vae = optim.AdamW(vae_params, lr=learning_rate)
    # GradScaler only works with CUDA, not with MPS
    scaler = GradScaler() if DEVICE == "cuda" else None
    reconstruction_loss_fn = nn.MSELoss()

    if load_checkpoint_path is not None and os.path.isfile(load_checkpoint_path):
        logger.info(f"Loading checkpoint from {load_checkpoint_path}")
        checkpoint = torch.load(load_checkpoint_path, map_location=DEVICE)
        vae_encoder.load_state_dict(checkpoint['encoder_state_dict'])
        vae_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer_vae.load_state_dict(checkpoint['optimizer_state_dict'])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    tokenizer = CompositionTokenizer(embedding_path=tokenizer_path, normalize=False)
    train_indices = indice_import(train_indices_path)
    vae_dataset = HDF5CompositionImageDataset(h5_path=dataset_path, transform=transform, tokenizer=tokenizer)
    vae_train_subset = Subset(vae_dataset, train_indices)

    from torch.utils.data import random_split
    train_size = int(0.9 * len(vae_train_subset))
    val_size = len(vae_train_subset) - train_size
    vae_train_subset, vae_test_subset = random_split(vae_train_subset, [train_size, val_size])

    vae_train_dataloader = DataLoader(vae_train_subset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    vae_test_dataloader = DataLoader(vae_test_subset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Train dataset size: {len(vae_train_subset)}")
    logger.info(f"Val dataset size: {len(vae_test_subset)}")
    logger.info(f"Train batches: {len(vae_train_dataloader)}")
    
    # Save initial checkpoint
    torch.save({
        'epoch': 0,
        'encoder_state_dict': vae_encoder.module.state_dict() if isinstance(vae_encoder, DataParallel) else vae_encoder.state_dict(),
        'decoder_state_dict': vae_decoder.module.state_dict() if isinstance(vae_decoder, DataParallel) else vae_decoder.state_dict(),
        'optimizer_state_dict': optimizer_vae.state_dict(),
        'train_loss': None,
        'val_loss': None,
    }, f"{checkpoint_dir}/vae_initial.pth")

    for epoch in range(num_epochs):
        vae_encoder.train()
        vae_decoder.train()
        total_vae_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0

        import time
        for batch_idx, (_, images) in enumerate(tqdm(vae_train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            if verbose:
                io_start = time.time()
            images = images.to(DEVICE, non_blocking=True)  # (Batch_Size, Channels, Height, Width)
            if verbose:
                io_end = time.time()
                if batch_idx % 100 == 0:
                    logger.info(f"[Batch {batch_idx}] Data loading time: {io_end - io_start:.4f} seconds")

            optimizer_vae.zero_grad()

            latent_h = images.size(2) // 8
            latent_w = images.size(3) // 8
            noise_for_reparam = torch.randn(images.size(0), 4, latent_h, latent_w, device=DEVICE)

            # Use mixed precision only for CUDA
            if DEVICE == "cuda":
                with autocast():
                    sampled_latents, mean, log_variance = vae_encoder(images, noise_for_reparam)
                    # Apply scaling factor to latents
                    sampled_latents = sampled_latents * latent_scaling_factor
                    # Scale back for decoder
                    reconstructed_images = vae_decoder(sampled_latents / latent_scaling_factor)
                    recon_loss = reconstruction_loss_fn(reconstructed_images, images)

                    kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=[1, 2, 3])
                    kl_div = torch.mean(kl_div)

                    vae_loss = recon_loss + kl_weight * kl_div

                scaler.scale(vae_loss).backward()
                scaler.step(optimizer_vae)
                scaler.update()
            else:
                # MPS and CPU: regular precision
                sampled_latents, mean, log_variance = vae_encoder(images, noise_for_reparam)
                # Apply scaling factor to latents
                sampled_latents = sampled_latents * latent_scaling_factor
                # Scale back for decoder
                reconstructed_images = vae_decoder(sampled_latents / latent_scaling_factor)
                recon_loss = reconstruction_loss_fn(reconstructed_images, images)

                kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=[1, 2, 3])
                kl_div = torch.mean(kl_div)

                vae_loss = recon_loss + kl_weight * kl_div
                
                vae_loss.backward()
                optimizer_vae.step()

            total_vae_loss += vae_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_div.item()

            if verbose and (batch_idx + 1) % 500 == 0:
                logger.info(f"VAE Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(vae_train_dataloader)}], "
                      f"Total Loss: {vae_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, KL Loss: {kl_div.item():.4f}")

        avg_vae_loss = total_vae_loss / len(vae_train_dataloader)
        avg_recon_loss = total_recon_loss / len(vae_train_dataloader)
        avg_kl_loss = total_kl_loss / len(vae_train_dataloader)
        logger.info(f"VAE Epoch [{epoch+1}/{num_epochs}] completed. Avg Total Loss: {avg_vae_loss:.4f}, "
              f"Avg Recon Loss: {avg_recon_loss:.4f}, Avg KL Loss: {avg_kl_loss:.4f}")

        logger.info("Validating VAE...")
        vae_encoder.eval()
        vae_decoder.eval()
        total_val_recon_loss = 0.0
        total_val_kl_loss = 0.0

        with torch.no_grad():
            for _, images in vae_test_dataloader:
                images = images.to(DEVICE)
                latent_h = images.size(2) // 8
                latent_w = images.size(3) // 8
                noise_for_reparam = torch.randn(images.size(0), 4, latent_h, latent_w, device=DEVICE)

                sampled_latents, mean, log_variance = vae_encoder(images, noise_for_reparam)
                # Apply scaling factor to latents
                sampled_latents = sampled_latents * latent_scaling_factor
                # Scale back for decoder
                reconstructed_images = vae_decoder(sampled_latents / latent_scaling_factor)

                recon_loss = reconstruction_loss_fn(reconstructed_images, images)
                kl_div = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=[1, 2, 3])
                kl_div = torch.mean(kl_div)

                total_val_recon_loss += recon_loss.item()
                total_val_kl_loss += kl_div.item()

        avg_val_recon_loss = total_val_recon_loss / len(vae_test_dataloader)
        avg_val_kl_loss = total_val_kl_loss / len(vae_test_dataloader)
        avg_val_total_loss = avg_val_recon_loss + kl_weight * avg_val_kl_loss

        logger.info(f"Validation - Recon Loss: {avg_val_recon_loss:.4f}, KL Loss: {avg_val_kl_loss:.4f}, Total Loss: {avg_val_total_loss:.4f}")

        # Save checkpoint for current epoch
        torch.save({
            'epoch': epoch + 1,
            'encoder_state_dict': vae_encoder.module.state_dict() if isinstance(vae_encoder, DataParallel) else vae_encoder.state_dict(),
            'decoder_state_dict': vae_decoder.module.state_dict() if isinstance(vae_decoder, DataParallel) else vae_decoder.state_dict(),
            'optimizer_state_dict': optimizer_vae.state_dict(),
            'train_loss': avg_vae_loss,
            'val_loss': avg_val_total_loss,
        }, f"{checkpoint_dir}/vae_epoch_{epoch+1}.pth")
        
        # Early stopping logic
        if avg_val_total_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_total_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'encoder_state_dict': vae_encoder.module.state_dict() if isinstance(vae_encoder, DataParallel) else vae_encoder.state_dict(),
                'decoder_state_dict': vae_decoder.module.state_dict() if isinstance(vae_decoder, DataParallel) else vae_decoder.state_dict(),
                'optimizer_state_dict': optimizer_vae.state_dict(),
                'train_loss': avg_vae_loss,
                'val_loss': avg_val_total_loss,
            }, f"{checkpoint_dir}/vae_best.pth")
            logger.info(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s). Best validation loss: {best_val_loss:.4f}")
            
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
                logger.info(f"Best validation loss was: {best_val_loss:.4f}")
                break

    logger.info("VAE training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--kl_weight", type=float, default=1e-6)
    parser.add_argument("--dataset_path", type=str, default="data/dataset_comp_image_spectra.h5")
    parser.add_argument("--tokenizer_path", type=str, default="embedding/tokenizers/megnet16-embedding.json")
    parser.add_argument("--train_indices_path", type=str, default="splits/split_train_indices.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/vae")
    parser.add_argument("--load_checkpoint_path", type=str, default=None, help="Path to a checkpoint file to resume training from")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs without improvement before early stopping")
    parser.add_argument("--min_delta", type=float, default=0.0001, help="Minimum change in validation loss to qualify as improvement")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output with detailed batch-level logging")
    parser.add_argument("--latent_scaling_factor", type=float, default=0.18215, help="Scaling factor for latent representations (default: 0.18215)")
    args = parser.parse_args()

    train_vae(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        kl_weight=args.kl_weight,
        dataset_path=args.dataset_path,
        tokenizer_path=args.tokenizer_path,
        train_indices_path=args.train_indices_path,
        checkpoint_dir=args.checkpoint_dir,
        load_checkpoint_path=args.load_checkpoint_path,
        patience=args.patience,
        min_delta=args.min_delta,
        verbose=args.verbose,
        latent_scaling_factor=args.latent_scaling_factor
    )