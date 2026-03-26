import os
import argparse
from matplotlib.pylab import rint
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import logging

# Ensure your custom modules are correctly importable
from sd.encoder import VAE_Encoder
from sd.diffusion import Diffusion # Assuming TimeEmbedding is used within or by Diffusion
from sd.ddpm import DDPMSampler
from sd.data_helper import HDF5CompositionImageDataset, indice_import
from embedding.element_tokenizer import CompositionTokenizer
from sd.ccip import CCIP
from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer
# --- Constants ---
VAE_DOWNSAMPLE_FACTOR = 8
NUM_VAE_LATENT_CHANNELS = 4

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def get_sinusoidal_time_embeddings(timesteps_tensor: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Generates sinusoidal time embeddings.
    Args:
        timesteps_tensor: 1D tensor of timesteps.
        embedding_dim: The dimension of the embeddings.
    Returns:
        A tensor of shape (len(timesteps_tensor), embedding_dim).
    """
    assert len(timesteps_tensor.shape) == 1, "Timesteps tensor must be 1D"
    
    half_dim = embedding_dim // 2
    emb_freq_den = torch.tensor(10000.0, device=timesteps_tensor.device, dtype=torch.float32)
    log_term = torch.log(emb_freq_den) / (half_dim - 1)
    exp_term = torch.arange(half_dim, device=timesteps_tensor.device, dtype=torch.float32) * -log_term
    emb = torch.exp(exp_term)
    emb = timesteps_tensor.float()[:, None] * emb[None, :]
    embeddings = torch.cat((emb.sin(), emb.cos()), dim=-1)
    
    if embedding_dim % 2 == 1:
        embeddings = torch.nn.functional.pad(embeddings, (0, 1, 0, 0))
    return embeddings

def setup_ddp(local_rank: int, rank: int, world_size: int):
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        logger.info(f"DDP: Rank {rank}/{world_size} initialized on cuda:{local_rank}.")
    else:
        logger.info("DDP: Running in single-process mode (no DDP initialization).")

def cleanup_ddp(rank: int, world_size: int):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()
        logger.info(f"DDP: Process group destroyed on Rank {rank}.")

# Modified initialize_models
def initialize_models(args, device: str, local_rank: int, world_size: int, 
                      initial_context_embedder_module: nn.Module = None):
    """Initializes VAE, main model, processes the initial_context_embedder_module, and tokenizer."""
    current_process_rank = dist.get_rank() if dist.is_initialized() else 0
    logger.info(f"Rank {current_process_rank}: Loading VAE checkpoint from {args.vae_checkpoint_path} onto {device}")
    vae_encoder = VAE_Encoder().to(device)
    try:
        vae_checkpoint = torch.load(args.vae_checkpoint_path, map_location=device, weights_only=True)
        if 'encoder_state_dict' not in vae_checkpoint:
            raise KeyError(f"Checkpoint does not contain 'encoder_state_dict'. Available keys: {vae_checkpoint.keys()}")
        vae_encoder.load_state_dict(vae_checkpoint['encoder_state_dict'])
    except FileNotFoundError:
        logger.error(f"VAE checkpoint not found at {args.vae_checkpoint_path}. Exiting.")
        raise
    except KeyError as e:
        logger.error(f"Error loading VAE state_dict: {e}. Exiting.")
        raise
    vae_encoder.eval()

    if not os.path.exists(args.tokenizer_path):
         logger.warning(f"Tokenizer path {args.tokenizer_path} does not exist. Ensure it's correct or training may fail.")

    if args.context_embedder_type == "TrueCCIP":
        tokenizer = CCIPCompositionTokenizer()
        logger.info(f"Rank {current_process_rank}: Using TrueCCIP tokenizer.")
    else:
        tokenizer = CompositionTokenizer(args.tokenizer_path)
        logger.info(f"Rank {current_process_rank}: Using standard CompositionTokenizer from {args.tokenizer_path}.")

    model = Diffusion(d_context=args.context_embed_dim).to(device)
    
    # Process the passed initial_context_embedder_module
    context_embedder = None
    if initial_context_embedder_module is not None:
        context_embedder = initial_context_embedder_module.to(device)
        logger.info(f"Rank {current_process_rank}: Context embedder of type {args.context_embedder_type} moved to {device}.")
        if isinstance(context_embedder, TrueCCIP):
            logger.info(f"Rank {current_process_rank}: Initial context embedder is a TrueCCIP model with output_dim {context_embedder.output_dim}.")
            context_embedder.eval()
            # Freeze TrueCCIP parameters
            for param in context_embedder.parameters():
                param.requires_grad = False
            logger.info(f"Rank {current_process_rank}: TrueCCIP parameters frozen (requires_grad=False).")
    else:
        # This case implies no context embedder is used, which would require model and training loop adaptation.
        # For now, we assume initial_context_embedder_module is always provided if context is expected.
        logger.warning(f"Rank {current_process_rank}: No initial_context_embedder_module provided. Contextual conditioning will be absent.")


    if world_size > 1:
        # Configure DDP with more robust settings to avoid gradient layout warnings
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank, 
            find_unused_parameters=args.ddp_find_unused_parameters,
            gradient_as_bucket_view=True,  # This can help with gradient layout issues
            static_graph=False  # Set to False for flexibility, True if the model graph is static
        )
        logger.info(f"Rank {current_process_rank}: Wrapped model with DDP.")
        if context_embedder is not None:
            # For TrueCCIP, we need to handle DDP differently since it's frozen
            actual_context_embedder = context_embedder
            if isinstance(actual_context_embedder, TrueCCIP):
                # Don't wrap frozen TrueCCIP with DDP since it has no trainable parameters
                # Just keep it as-is - each GPU will have its own copy but they won't sync
                logger.info(f"Rank {current_process_rank}: TrueCCIP context_embedder kept without DDP wrapping (frozen model).")
            else:
                context_embedder = DDP(
                    context_embedder, 
                    device_ids=[local_rank], 
                    output_device=local_rank, 
                    find_unused_parameters=args.ddp_find_unused_parameters,
                    gradient_as_bucket_view=True
                )
                logger.info(f"Rank {current_process_rank}: Wrapped context_embedder with DDP.")

    return vae_encoder, tokenizer, model, context_embedder


def initialize_optimizer_and_scheduler(args, model, context_embedder, num_train_steps_per_epoch: int):
    current_process_rank = dist.get_rank() if dist.is_initialized() else 0
    params_to_optimize = list(model.parameters())
    
    # Only add context_embedder parameters if it's not TrueCCIP (which should be frozen)
    if context_embedder is not None:
        # Check if it's wrapped with DDP first
        actual_context_embedder = context_embedder.module if hasattr(context_embedder, 'module') else context_embedder
        if not isinstance(actual_context_embedder, TrueCCIP):
            params_to_optimize.extend(list(context_embedder.parameters()))
            logger.info(f"Rank {current_process_rank}: Added context embedder parameters to optimizer.")
        else:
            logger.info(f"Rank {current_process_rank}: TrueCCIP context embedder parameters excluded from optimizer (frozen).")
    
    if not params_to_optimize:
        logger.error(f"Rank {current_process_rank}: No parameters to optimize. Check model and context_embedder instantiation.")
        raise ValueError("Optimizer received no parameters.")

    optimizer = optim.AdamW(params_to_optimize, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    lr_scheduler = None
    if args.lr_scheduler_type:
        if args.lr_scheduler_type.lower() == "cosine":
            total_train_steps = args.num_epochs * num_train_steps_per_epoch
            t_max = total_train_steps if args.lr_cosine_T_max_is_steps else args.num_epochs
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=args.lr_cosine_eta_min)
            logger.info(f"Rank {current_process_rank}: Using CosineAnnealingLR scheduler (T_max={t_max}, eta_min={args.lr_cosine_eta_min}).")
        elif args.lr_scheduler_type.lower() == "steplr":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
            logger.info(f"Rank {current_process_rank}: Using StepLR scheduler (step_size={args.lr_step_size}, gamma={args.lr_gamma}).")
    return optimizer, lr_scheduler

def prepare_dataloader(args, tokenizer, current_rank: int, world_size: int):
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size), antialias=True),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    logger.info(f"Rank {current_rank}: Initializing dataset...")
    
    # Print the path to indices
    print(f"Rank {current_rank}: Train indices path: {args.train_indices_path}")
    
    try:
        if not os.path.exists(args.h5_data_path):
            raise FileNotFoundError(f"HDF5 data file not found at {args.h5_data_path}")
        if not os.path.exists(args.train_indices_path):
            raise FileNotFoundError(f"Train indices file not found at {args.train_indices_path}")
        train_indices = indice_import(args.train_indices_path)
        
        # Also print the actual loaded indices for debugging
        print(f"Rank {current_rank}: Loaded {len(train_indices)} train indices from {args.train_indices_path}")
        
        # Validate indices
        if len(train_indices) == 0:
            raise ValueError("No training indices loaded - file might be empty or malformed")
        
        # Debug: Check range of indices
        min_idx, max_idx = min(train_indices), max(train_indices)
        logger.info(f"Rank {current_rank}: Index range - min: {min_idx}, max: {max_idx}")
        
    except FileNotFoundError as e:
        logger.error(f"Rank {current_rank}: {e}. Exiting.")
        raise
    except Exception as e:
        logger.error(f"Rank {current_rank}: Error loading indices: {e}")
        raise

    try:
        dataset = HDF5CompositionImageDataset(
            h5_path=args.h5_data_path,
            transform=image_transforms,
            comp_format='dict',
            tokenizer=tokenizer
        )
        logger.info(f'train_indices sample: {train_indices[:10]}')  # Debugging line to check indices
        
        # Validate dataset size vs indices
        dataset_size = len(dataset)
        if max_idx >= dataset_size:
            raise ValueError(f"Index {max_idx} exceeds dataset size {dataset_size}")
            
        logger.info(f"Rank {current_rank}: Dataset size: {dataset_size}")
        
        train_dataset = Subset(dataset, train_indices)
        
    except Exception as e:
        logger.error(f"Rank {current_rank}: Error creating dataset: {e}")
        raise

    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=current_rank, shuffle=True, seed=args.seed, drop_last=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,  # Set to 0 to disable multiprocessing for debugging
        pin_memory=args.pin_memory,
        persistent_workers=False,  # Disable persistent workers when num_workers=0
        drop_last=True
    )

    if current_rank == 0:
        logger.info("Dataset initialization completed successfully.")

    logger.info(f"Rank {current_rank}: Dataset initialized. Num train samples: {len(train_dataset)}")
    if current_rank == 0:
        effective_batch_size = args.batch_size * world_size
        logger.info(f"Global effective batch size: {effective_batch_size}. Per-GPU batch size: {args.batch_size}.")
        if not os.path.exists(args.checkpoint_dir):
            try:
                os.makedirs(args.checkpoint_dir)
                logger.info(f"Created checkpoint directory: {args.checkpoint_dir}")
            except OSError as e:
                logger.error(f"Could not create checkpoint directory {args.checkpoint_dir}: {e}")
    return train_dataloader, train_sampler


def train_epoch_step(epoch_num, args, model, context_embedder, vae_encoder, dataloader, sampler,
                     optimizer, step_lr_scheduler, loss_fn, ddpm_sampler, device, 
                     current_rank, world_size, is_main_process, scaler=None):
    model.train()
    if context_embedder: # Ensure context_embedder is also in train mode if it exists
        # Check if it's wrapped with DDP first
        actual_context_embedder = context_embedder.module if hasattr(context_embedder, 'module') else context_embedder
        if isinstance(actual_context_embedder, TrueCCIP):
            context_embedder.eval()
            logger.info(f"Rank {current_rank}: TrueCCIP context embedder in eval mode.")
        else:
            context_embedder.train()
            logger.info(f"Rank {current_rank}: Context embedder in train mode.")

    if sampler and hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(epoch_num)

    total_loss_epoch = 0.0
    num_batches_processed = 0
    progress_bar_desc = f"Epoch {epoch_num+1}/{args.num_epochs}"
    if world_size > 1:
        progress_bar_desc += f" (Rank {current_rank})"
    
    data_iterator = dataloader
    if is_main_process:
        data_iterator = tqdm(dataloader, desc=progress_bar_desc, unit="batch", leave=False)

    # Memory monitoring
    if torch.cuda.is_available() and is_main_process:
        logger.info(f"GPU memory before epoch {epoch_num+1}: {torch.cuda.memory_allocated(device)/1024**3:.2f}GB allocated, "
                   f"{torch.cuda.memory_reserved(device)/1024**3:.2f}GB reserved")

    for batch_idx, (comp_vectors, images) in enumerate(data_iterator):
        try:
            # Zero gradients at the start of each batch
            optimizer.zero_grad()
            
            images = images.to(device, non_blocking=True)
            
            # Handle different tokenizer outputs
            if isinstance(comp_vectors, list) and len(comp_vectors) == 3:
                # TrueCCIP tokenizer returns (tokens, fractions, attention_mask)
                tokens, fractions, attention_mask = comp_vectors
                tokens = tokens.to(device, non_blocking=True)
                fractions = fractions.to(device, non_blocking=True).float()
                attention_mask = attention_mask.to(device, non_blocking=True)
                comp_vectors = (tokens, fractions, attention_mask)
                
                # Debug: Check input data for TrueCCIP
                if is_main_process and batch_idx % args.log_every_n_batches == 0:
                    logger.info(f"Batch {batch_idx}: images shape: {images.shape}")
                    logger.info(f"TrueCCIP tokens shape: {tokens.shape}, fractions shape: {fractions.shape}, attention_mask shape: {attention_mask.shape}")
                    logger.info(f"Images device: {images.device}, tokens device: {tokens.device}")
                    
                # Check for invalid input values
                if torch.isnan(images).any() or torch.isinf(images).any():
                    logger.error(f"Rank {current_rank}: Invalid image values detected in batch {batch_idx}")
                    continue
                    
                if torch.isnan(fractions).any() or torch.isinf(fractions).any():
                    logger.error(f"Rank {current_rank}: Invalid fractions detected in batch {batch_idx}")
                    continue
            else:
                # Standard tokenizer returns single tensor
                comp_vectors = comp_vectors.to(device, non_blocking=True).float()
                
                # Debug: Check input data for standard tokenizer
                if is_main_process and batch_idx % args.log_every_n_batches == 0:
                    logger.info(f"Batch {batch_idx}: images shape: {images.shape}, comp_vectors shape: {comp_vectors.shape}")
                    logger.info(f"Images device: {images.device}, comp_vectors device: {comp_vectors.device}")
                    
                # Check for invalid input values
                if torch.isnan(images).any() or torch.isinf(images).any():
                    logger.error(f"Rank {current_rank}: Invalid image values detected in batch {batch_idx}")
                    continue
                    
                if torch.isnan(comp_vectors).any() or torch.isinf(comp_vectors).any():
                    logger.error(f"Rank {current_rank}: Invalid composition vectors detected in batch {batch_idx}")
                    continue

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                latent_height = args.image_size // VAE_DOWNSAMPLE_FACTOR
                latent_width = args.image_size // VAE_DOWNSAMPLE_FACTOR
                noise_for_vae = torch.randn(
                    images.size(0), NUM_VAE_LATENT_CHANNELS,
                    latent_height, latent_width, device=device
                )
                
                try:
                    latents, _, _ = vae_encoder(images, noise_for_vae)
                    
                    # Debug: Check VAE output
                    if is_main_process and batch_idx % args.log_every_n_batches == 0:
                        logger.info(f"VAE latents - shape: {latents.shape}, device: {latents.device}")
                        logger.info(f"Images input - shape: {images.shape}, device: {images.device}, "
                                   f"min: {images.min():.4f}, max: {images.max():.4f}")
                        
                except Exception as e:
                    logger.error(f"Rank {current_rank}: Error in VAE encoding: {e}")
                    logger.error(f"Images shape: {images.shape}, noise_for_vae shape: {noise_for_vae.shape}")
                    raise
                
                if args.latent_scale_factor is not None:
                    print(f"Rank {current_rank}: Applying latent scale factor: {args.latent_scale_factor}")
                    latents = latents * args.latent_scale_factor
                    if is_main_process and batch_idx % args.log_every_n_batches == 0:
                        logger.info(f"Applied latent scale factor: {args.latent_scale_factor}")
                else:
                    if is_main_process and batch_idx == 0:
                        logger.warning("latent scale factor is None - using unscaled latents")

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                context_embeddings = None
                if context_embedder is not None: # Check if context_embedder exists
                    try:
                        # Check if it's wrapped with DDP first
                        actual_context_embedder = context_embedder.module if hasattr(context_embedder, 'module') else context_embedder
                        
                        if isinstance(actual_context_embedder, TrueCCIP):
                            # For TrueCCIP, use no_grad to prevent gradient computation
                            with torch.no_grad():
                                # TrueCCIP expects (tokens, fractions, attention_mask) as separate arguments
                                if isinstance(comp_vectors, tuple) and len(comp_vectors) == 3:
                                    tokens, fractions, attention_mask = comp_vectors
                                    raw_context_embeddings = context_embedder(tokens, fractions, attention_mask)
                                else:
                                    raw_context_embeddings = context_embedder(comp_vectors)
                                context_embeddings = raw_context_embeddings.unsqueeze(1)
                                if is_main_process and batch_idx % args.log_every_n_batches == 0:
                                    logger.info(f"TrueCCIP context embeddings computed with no_grad - shape: {context_embeddings.shape}, device: {context_embeddings.device}")
                        else:
                            # For other context embedders, compute normally (with gradients)
                            raw_context_embeddings = context_embedder(comp_vectors)
                            context_embeddings = raw_context_embeddings.unsqueeze(1)
                            if is_main_process and batch_idx % args.log_every_n_batches == 0:
                                logger.info(f"Context embeddings shape: {context_embeddings.shape}, device: {context_embeddings.device}")
                            
                    except Exception as e:
                        logger.error(f"Rank {current_rank}: Error in context embedder: {e}")
                        raise
                else: 
                    if is_main_process and batch_idx == 0:
                        logger.warning("Context embedder is None - model must handle context_embeddings=None")

                # Debug: Validate input tensor ranges and shapes
                if is_main_process and batch_idx % args.log_every_n_batches == 0:
                    logger.info(f"Latents - shape: {latents.shape}, device: {latents.device}, "
                               f"min: {latents.min():.4f}, max: {latents.max():.4f}, "
                               f"mean: {latents.mean():.4f}, std: {latents.std():.4f}")

                t_int = torch.randint(0, args.timesteps, (images.size(0),), device=device, dtype=torch.long)
                
                try:
                    noisy_latents, noise_target = ddpm_sampler.add_noise(latents, t_int)
                    
                    # Debug: Check noisy latents
                    if is_main_process and batch_idx % args.log_every_n_batches == 0:
                        logger.info(f"Noisy latents - shape: {noisy_latents.shape}, "
                                   f"min: {noisy_latents.min():.4f}, max: {noisy_latents.max():.4f}")
                        logger.info(f"Noise target - shape: {noise_target.shape}, "
                                   f"min: {noise_target.min():.4f}, max: {noise_target.max():.4f}")
                                   
                except Exception as e:
                    logger.error(f"Rank {current_rank}: Error in noise addition: {e}")
                    logger.error(f"Latents shape: {latents.shape}, t_int shape: {t_int.shape}")
                    raise

                time_embeddings = get_sinusoidal_time_embeddings(t_int, args.time_embedding_size).to(device)
                
                # Debug: Check time embeddings
                if is_main_process and batch_idx % args.log_every_n_batches == 0:
                    logger.info(f"Time embeddings - shape: {time_embeddings.shape}, device: {time_embeddings.device}")
                
                # The Diffusion model's forward pass
                try:
                    predicted_noise = model(noisy_latents, context_embeddings, time_embeddings) 
                    
                    # Debug: Check model output
                    if is_main_process and batch_idx % args.log_every_n_batches == 0:
                        logger.info(f"Predicted noise - shape: {predicted_noise.shape}, "
                                   f"min: {predicted_noise.min():.4f}, max: {predicted_noise.max():.4f}")
                        
                    # Validate output shape matches target
                    if predicted_noise.shape != noise_target.shape:
                        logger.error(f"Shape mismatch: predicted_noise {predicted_noise.shape} vs noise_target {noise_target.shape}")
                        raise ValueError(f"Model output shape {predicted_noise.shape} doesn't match target shape {noise_target.shape}")
                        
                except Exception as e:
                    logger.error(f"Rank {current_rank}: Error in model forward pass: {e}")
                    logger.error(f"Input shapes - noisy_latents: {noisy_latents.shape}, "
                               f"context_embeddings: {context_embeddings.shape if context_embeddings is not None else None}, "
                               f"time_embeddings: {time_embeddings.shape}")
                    raise
                    
                loss = loss_fn(predicted_noise, noise_target)
                
                # Debug: Check loss value
                if is_main_process and batch_idx % args.log_every_n_batches == 0:
                    logger.info(f"Loss: {loss.item():.6f}")
                    
                # Check for invalid loss values
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Rank {current_rank}: Invalid loss detected: {loss.item()}")
                    logger.error(f"Predicted noise stats - min: {predicted_noise.min()}, max: {predicted_noise.max()}, "
                               f"mean: {predicted_noise.mean()}, std: {predicted_noise.std()}")
                    logger.error(f"Noise target stats - min: {noise_target.min()}, max: {noise_target.max()}, "
                               f"mean: {noise_target.mean()}, std: {noise_target.std()}")
                    raise ValueError(f"Invalid loss value: {loss.item()}")

            params_for_clipping = list(model.parameters())
            # Only add context_embedder parameters for clipping if it's not TrueCCIP (which is frozen)
            if context_embedder:
                actual_context_embedder = context_embedder.module if hasattr(context_embedder, 'module') else context_embedder
                if not isinstance(actual_context_embedder, TrueCCIP):
                    params_for_clipping.extend(list(context_embedder.parameters()))

            if args.use_amp:
                if scaler is None: raise ValueError("AMP enabled but GradScaler not provided.")
                scaler.scale(loss).backward()
                if args.grad_clip_norm is not None and params_for_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params_for_clipping, args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm is not None and params_for_clipping:
                    torch.nn.utils.clip_grad_norm_(params_for_clipping, args.grad_clip_norm)
                optimizer.step()
            
            if step_lr_scheduler:
                 step_lr_scheduler.step()

            total_loss_epoch += loss.item()

            if is_main_process and (batch_idx + 1) % args.log_every_n_batches == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"  Epoch [{epoch_num+1}/{args.num_epochs}] Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6e}")
                
        except Exception as e:
            logger.error(f"Rank {current_rank}: Error in batch {batch_idx}: {e}")
            if is_main_process:
                logger.error(f"Skipping batch {batch_idx} due to error: {e}")
                
            # Reset scaler state if AMP is enabled
            if args.use_amp and scaler is not None:
                scaler.update()
            
            # Reset optimizer state
            optimizer.zero_grad()
            
            # Clear CUDA cache to help with memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            continue
    
    avg_loss_epoch_local = total_loss_epoch / len(dataloader) if len(dataloader) > 0 else 0.0
    return avg_loss_epoch_local

# Modified run_training
def run_training(args, local_rank: int, current_rank: int, world_size: int, is_main_process: bool, 
                 initial_context_embedder_module: nn.Module = None): # Added initial_context_embedder_module
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed + current_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + current_rank)

    logger.info(f"Rank {current_rank}: Starting training on device {device}. Args: {args}")

    # Pass initial_context_embedder_module to initialize_models
    vae_encoder, tokenizer, model, context_embedder = initialize_models(
        args, device, local_rank, world_size, initial_context_embedder_module
    )

    train_dataloader, train_sampler = prepare_dataloader(args, tokenizer, current_rank, world_size)
    num_train_steps_per_epoch = len(train_dataloader)

    optimizer, lr_scheduler = initialize_optimizer_and_scheduler(args, model, context_embedder, num_train_steps_per_epoch)
    
    loss_fn = nn.MSELoss()
    ddpm_sampler_generator = torch.Generator(device=device).manual_seed(args.seed + current_rank)
    ddpm_sampler = DDPMSampler(ddpm_sampler_generator, num_training_steps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)
    
    scaler = None
    if args.use_amp:
        if not torch.cuda.is_available():
            logger.warning(f"Rank {current_rank}: AMP requested but CUDA is not available. AMP will be disabled.")
            args.use_amp = False
        else:
            scaler = torch.cuda.amp.GradScaler()
            logger.info(f"Rank {current_rank}: Automatic Mixed Precision (AMP) enabled.")

    start_epoch = 0
    if args.resume_from_checkpoint:
        if os.path.isfile(args.resume_from_checkpoint):
            logger.info(f"Rank {current_rank}: Resuming from checkpoint: {args.resume_from_checkpoint}")
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            
            model_to_load = model.module if world_size > 1 else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            
            if context_embedder is not None and 'context_embedder_state_dict' in checkpoint:
                # Check if context embedder is wrapped with DDP
                actual_context_embedder = context_embedder.module if hasattr(context_embedder, 'module') else context_embedder
                # For TrueCCIP, we don't wrap with DDP, so always use the context_embedder directly
                if isinstance(actual_context_embedder, TrueCCIP):
                    context_embedder.load_state_dict(checkpoint['context_embedder_state_dict'])
                else:
                    context_embedder_to_load = context_embedder.module if world_size > 1 else context_embedder
                    context_embedder_to_load.load_state_dict(checkpoint['context_embedder_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            if args.use_amp and scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logger.info(f"Rank {current_rank}: Resumed from epoch {start_epoch}. Previous loss: {checkpoint.get('avg_global_loss', 'N/A')}")
        else:
            logger.warning(f"Rank {current_rank}: Resume checkpoint not found at {args.resume_from_checkpoint}. Starting from scratch.")

    step_wise_lr_scheduler = None
    epoch_wise_lr_scheduler = None
    if lr_scheduler:
        if args.lr_scheduler_type == "cosine" and args.lr_cosine_T_max_is_steps:
            step_wise_lr_scheduler = lr_scheduler
        else:
            epoch_wise_lr_scheduler = lr_scheduler

    logger.info(f"Rank {current_rank}: Starting training from epoch {start_epoch + 1} up to {args.num_epochs} epochs...")
    for epoch in range(start_epoch, args.num_epochs):
        avg_loss_epoch_local = train_epoch_step(
            epoch, args, model, context_embedder, vae_encoder, train_dataloader, train_sampler,
            optimizer, step_wise_lr_scheduler, loss_fn, ddpm_sampler, device,
            current_rank, world_size, is_main_process, scaler
        )
        
        if epoch_wise_lr_scheduler:
            epoch_wise_lr_scheduler.step()

        avg_loss_epoch_global = avg_loss_epoch_local
        if world_size > 1:
            avg_loss_tensor = torch.tensor(avg_loss_epoch_local, device=device)
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss_epoch_global = avg_loss_tensor.item()

        if is_main_process:
            current_lr_display = optimizer.param_groups[0]['lr']
            logger.info(f"Epoch [{epoch+1}/{args.num_epochs}] completed. Global Avg Loss: {avg_loss_epoch_global:.4f}, Current LR: {current_lr_display:.6e}")
            
            if (epoch + 1) % args.save_every_n_epochs == 0 or (epoch + 1) == args.num_epochs:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"ddpm_epoch_{epoch+1}.pth")
                
                save_obj = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': args,
                    'avg_global_loss': avg_loss_epoch_global
                }
                if context_embedder is not None:
                    # Check if context embedder is wrapped with DDP
                    actual_context_embedder = context_embedder.module if hasattr(context_embedder, 'module') else context_embedder
                    # For TrueCCIP, we don't wrap with DDP, so always use the context_embedder directly
                    if isinstance(actual_context_embedder, TrueCCIP):
                        save_obj['context_embedder_state_dict'] = context_embedder.state_dict()
                    else:
                        save_obj['context_embedder_state_dict'] = context_embedder.module.state_dict() if world_size > 1 else context_embedder.state_dict()
                if lr_scheduler:
                    save_obj['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
                if args.use_amp and scaler:
                    save_obj['scaler_state_dict'] = scaler.state_dict()

                try:
                    torch.save(save_obj, checkpoint_path)
                    logger.info(f"Checkpoint saved to {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")

    if is_main_process:
        logger.info(f"Training finished for Rank {current_rank}.")

# Modified get_args
def get_args():
    parser = argparse.ArgumentParser(description="PyTorch DDP Training for DDPM (Improved)")
    
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument("--h5_data_path", type=str, default='data/dataset_comp_image_spectra.h5')
    path_group.add_argument("--train_indices_path", type=str, default='splits/split_train_indices.txt')
    path_group.add_argument("--checkpoint_dir", type=str, default="checkpoints_ddpm_improved")
    path_group.add_argument("--vae_checkpoint_path", type=str, default="checkpoints/vae_epoch_2.pth")
    path_group.add_argument("--ccip_checkpoint_path", type=str, default="checkpoints/ccip_epoch_2.pth")
    path_group.add_argument("--tokenizer_path", type=str, default="embedding/tokenizers/megnet16-embedding.json")
    path_group.add_argument("--resume_from_checkpoint", type=str, default=None)

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--image_size", type=int, default=64)
    # Remove the hard-coded default and let it be determined from tokenizer
    model_group.add_argument("--comp_vector_dim", type=int, default=None, help="Dimension of input composition vector. If None, will be inferred from tokenizer.")
    model_group.add_argument("--context_embed_dim", type=int, default=64, help="Output dimension of context embedder / input to U-Net for context.")
    model_group.add_argument("--time_embedding_size", type=int, default=320)
    model_group.add_argument("--latent_scale_factor", type=float, default=0.18215)
    # New argument for context embedder type
    model_group.add_argument("--context_embedder_type", type=str, default="linear", choices=["linear", "mlp", "only_lin", "none", "TrueCCIP"], 
                             help="Type of context embedder. 'linear' uses nn.Linear. 'mlp' uses a multi-layer perceptron. 'none' uses no context embedder (model must support this).")


    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument("--learning_rate", type=float, default=1e-4)
    train_group.add_argument("--weight_decay", type=float, default=1e-2)
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--num_epochs", type=int, default=100)
    train_group.add_argument("--save_every_n_epochs", type=int, default=10)
    train_group.add_argument("--log_every_n_batches", type=int, default=100)
    train_group.add_argument("--grad_clip_norm", type=float, default=None)
    train_group.add_argument("--use_amp", action='store_true')
    train_group.add_argument("--no_amp", action='store_false', dest='use_amp')
    parser.set_defaults(use_amp=False)

    ddpm_group = parser.add_argument_group('DDPM Parameters')
    ddpm_group.add_argument("--timesteps", type=int, default=1000)
    ddpm_group.add_argument("--beta_start", type=float, default=0.00085)
    ddpm_group.add_argument("--beta_end", type=float, default=0.0120)

    data_ddp_group = parser.add_argument_group('DataLoader and DDP')
    data_ddp_group.add_argument("--num_workers", type=int, default=0)
    data_ddp_group.add_argument("--pin_memory", action='store_true')
    data_ddp_group.add_argument("--no_pin_memory", action='store_false', dest='pin_memory')
    parser.set_defaults(pin_memory=True)
    data_ddp_group.add_argument("--ddp_find_unused_parameters", action='store_true')
    data_ddp_group.add_argument("--no_ddp_find_unused_parameters", action='store_false', dest='ddp_find_unused_parameters')
    parser.set_defaults(ddp_find_unused_parameters=False)
    
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument("--seed", type=int, default=42)

    lr_sched_group = parser.add_argument_group('Learning Rate Scheduler')
    lr_sched_group.add_argument("--lr_scheduler_type", type=str, default=None, choices=["cosine", "steplr"])
    lr_sched_group.add_argument("--lr_cosine_T_max_is_steps", action='store_true')
    lr_sched_group.add_argument("--no_lr_cosine_T_max_is_steps", action='store_false', dest='lr_cosine_T_max_is_steps')
    parser.set_defaults(lr_cosine_T_max_is_steps=False)
    lr_sched_group.add_argument("--lr_cosine_eta_min", type=float, default=0.0)
    lr_sched_group.add_argument("--lr_step_size", type=int, default=30)
    lr_sched_group.add_argument("--lr_gamma", type=float, default=0.1)

    return parser.parse_args()

# Modified main
def main():
    args = get_args()
    
    # Print the indices path early in main
    print(f"Train indices path: {args.train_indices_path}")
    
    # Determine comp_vector_dim from tokenizer if not provided
    if args.comp_vector_dim is None:
        try:
            import json
            with open(args.tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
            # Assuming the JSON structure has embeddings for elements like 'H'
            if 'H' in tokenizer_data:
                args.comp_vector_dim = len(tokenizer_data['H'])
                logger.info(f"Inferred comp_vector_dim={args.comp_vector_dim} from tokenizer file.")
            else:
                # Fallback: try to get dimension from any element
                first_element = next(iter(tokenizer_data.values()))
                args.comp_vector_dim = len(first_element) if isinstance(first_element, list) else 16
                logger.warning(f"Could not find 'H' in tokenizer. Using inferred dimension: {args.comp_vector_dim}")
        except Exception as e:
            logger.error(f"Could not read tokenizer file {args.tokenizer_path}: {e}")
            args.comp_vector_dim = 16  # Fallback default
            logger.warning(f"Using fallback comp_vector_dim={args.comp_vector_dim}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    if rank == 0: logging.getLogger().setLevel(logging.INFO)
    else: logging.getLogger().setLevel(logging.WARNING)
    logger.info(f"Process Rank {rank}, Local Rank {local_rank}, World Size {world_size}")

    is_main_process = (rank == 0)
    setup_ddp(local_rank, rank, world_size)

    # Instantiate initial context embedder module based on args
    initial_context_embedder_module = None
    if args.context_embedder_type == "linear":
        # Option 1: Simple linear + activation
        initial_context_embedder_module = nn.Sequential(
            nn.Linear(args.comp_vector_dim, args.context_embed_dim),
            nn.ReLU()
        )
        logger.info(f"Rank {rank}: Created initial nn.Linear context embedder with ReLU activation (CPU).")
    elif args.context_embedder_type == "mlp":
        # Option 2: Multi-layer perceptron
        initial_context_embedder_module = nn.Sequential(
            nn.Linear(args.comp_vector_dim, args.context_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.context_embed_dim * 2, args.context_embed_dim),
            nn.ReLU()
        )
    elif args.context_embedder_type == 'only_lin':
        initial_context_embedder_module = nn.Sequential(
            nn.Linear(args.comp_vector_dim, args.context_embed_dim)
        )
        logger.info(f"Rank {rank}: Created initial MLP context embedder (CPU).")
    elif args.context_embedder_type == "none":
        logger.info(f"Rank {rank}: No context embedder will be used as per --context_embedder_type none.")
        # The Diffusion model (U-Net) and training loop must be able to handle this.
        # Specifically, model's forward pass should accept context_embeddings=None.
    elif args.context_embedder_type == "TrueCCIP":

        initial_context_embedder_module = TrueCCIP(
            vocab_size=98,  # 95 elements + 3 special tokens
            embedding_dim=128,  # Internal embedding dimension
            max_length=20,  # Max composition sequence length
            n_layers=6,
            n_heads=8,
            output_dim=args.context_embed_dim,  # Use the argument value
            dropout=0.1
        )
        logger.info(f"Rank {rank}: Created TrueCCIP context embedder with output_dim={args.context_embed_dim} (CPU).")
        
        # Load the pretrained weights
        context_embedder_checkpoint = torch.load(args.ccip_checkpoint_path, weights_only=False, map_location='cpu')
        if 'model_state_dict' in context_embedder_checkpoint:
            state_dict = context_embedder_checkpoint['model_state_dict']
            logger.info(f"Found model_state_dict with {len(state_dict)} keys")
        else:
            state_dict = context_embedder_checkpoint
            logger.info(f"Using checkpoint directly with {len(state_dict)} keys")

        if any(key.startswith('composition_encoder.') for key in state_dict.keys()):
            logger.info("Found composition_encoder prefix, removing it...")
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('composition_encoder.'):
                    new_key = key[len('composition_encoder.'):]
                    clean_state_dict[new_key] = value
                    logger.info(f"Mapped: {key} -> {new_key}")
            state_dict = clean_state_dict
            logger.info(f"Cleaned state dict has {len(state_dict)} keys")
        
        # Load weights with strict=False to handle dimension mismatches in projection layer
        missing_keys, unexpected_keys = initial_context_embedder_module.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys when loading TrueCCIP: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys when loading TrueCCIP: {unexpected_keys}")
        
        logger.info(f"Rank {rank}: TrueCCIP context embedder loaded with output_dim={args.context_embed_dim}.")
    else:
        # This case should ideally not be reached if choices in argparse are set correctly.
        logger.error(f"Rank {rank}: Unknown context_embedder_type '{args.context_embedder_type}'. Exiting.")
        if world_size > 1 and dist.is_initialized(): 
            dist.destroy_process_group()
        return 1  # Exit with an error code

    # Pass the initial_context_embedder_module to run_training
    try:
        run_training(args, local_rank, rank, world_size, is_main_process, initial_context_embedder_module)
    except Exception as e:
        logger.error(f"Exception in run_training on Rank {rank}: {e}", exc_info=True)
        if world_size > 1 and dist.is_initialized(): 
            dist.destroy_process_group()
        raise
    finally:
        cleanup_ddp(rank, world_size)


if __name__ == "__main__":
    main()