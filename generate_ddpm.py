#!/usr/bin/env python3
"""
DDPM + Context Embedder Image Generation Script

This script generates images using a trained DDPM model with context embeddings.
Follows the structure and patterns used in train_ddpm.py for consistency.

Usage:
    # TrueCCIP mode:
    python generate_ddpm.py --ddpm_checkpoint path/to/ddpm.pth --context_embedder_type TrueCCIP --ccip_checkpoint_path path/to/ccip.pth

    # Linear context embedder mode:
    python generate_ddpm.py --ddpm_checkpoint path/to/ddpm.pth --context_embedder_type linear
"""
import os
import argparse
import time
import json
import sys
import gc
import logging
import numpy as np
import h5py
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

# Import our models and utilities
from sd.ddpm import DDPMSampler
from sd.diffusion import Diffusion
from sd.decoder import VAE_Decoder
from sd.true_ccip import TrueCCIP, CCIPCompositionTokenizer
from sd.data_helper import HDF5CompositionImageDataset, indice_import
from embedding.element_tokenizer import CompositionTokenizer
from sd.ccip import CCIP

# --- Constants ---
VAE_DOWNSAMPLE_FACTOR = 8
NUM_VAE_LATENT_CHANNELS = 4

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- DataLoader helpers (module-level for pickling safety) ---
class CompositionTokensDataset(Dataset):
    """Wraps pre-tokenized compositions for DataLoader usage."""
    def __init__(self, tokenized_list):
        self.data = tokenized_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return as-is; collate_fn will stack properly
        return self.data[idx]

def collate_tokenized(batch):
    """Collate function to handle TrueCCIP tuples or standard tensors.

    batch: list of tokenized entries.
      - TrueCCIP: each entry is a tuple (tokens, fractions, attention_mask)
      - Standard: each entry is a tensor or array
    Returns a tuple of tensors for TrueCCIP or a stacked tensor for standard.
    """
    first = batch[0]
    if isinstance(first, tuple) and len(first) == 3:
        tokens = torch.stack([b[0] if isinstance(b[0], torch.Tensor) else torch.as_tensor(b[0]) for b in batch], dim=0)
        fractions = torch.stack([b[1] if isinstance(b[1], torch.Tensor) else torch.as_tensor(b[1]) for b in batch], dim=0)
        attention_mask = torch.stack([b[2] if isinstance(b[2], torch.Tensor) else torch.as_tensor(b[2]) for b in batch], dim=0)
        return (tokens, fractions, attention_mask)
    # Standard path
    return torch.stack([b if isinstance(b, torch.Tensor) else torch.as_tensor(b) for b in batch], dim=0)


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

def clear_cuda_memory():
    """Clear CUDA memory and run garbage collection."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def log_memory_usage(stage: str):
    """Log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        logger.info(f"Memory at {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types and PyTorch tensors."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        return super(NumpyEncoder, self).default(obj)

class TimingLogger:
    """Utility class for timing operations and logging results."""
    
    def __init__(self, log_file: str = None):
        self.timings = {}
        self.log_file = log_file
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timings[operation] = {'start': time.time()}
        
    def end_timer(self, operation: str):
        """End timing an operation and log the result."""
        if operation in self.timings:
            end_time = time.time()
            duration = end_time - self.timings[operation]['start']
            self.timings[operation]['duration'] = duration
            logger.info(f"{operation}: {duration:.4f} seconds")
            return duration
        return None
        
    def save_timing_report(self):
        """Save timing report to file."""
        if self.log_file:
            report = {
                'timestamp': datetime.now().isoformat(),
                'timings': {op: data.get('duration', 0) for op, data in self.timings.items()}
            }
            with open(self.log_file, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            logger.info(f"Timing report saved to {self.log_file}")

def setup_logging(output_dir: str, verbose: bool = False):
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = os.path.join(output_dir, f"generation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Get the root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set the logging level
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    return log_file

def create_context_embedder_module(args):
    """Create a context embedder module based on the specified type."""
    context_embedder_module = None
    
    if args.context_embedder_type == "linear":
        context_embedder_module = nn.Sequential(
            nn.Linear(args.comp_vector_dim, args.context_embed_dim),
            nn.ReLU()
        )
        logger.info("Created linear context embedder with ReLU activation.")
    elif args.context_embedder_type == "mlp":
        context_embedder_module = nn.Sequential(
            nn.Linear(args.comp_vector_dim, args.context_embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.context_embed_dim * 2, args.context_embed_dim),
            nn.ReLU()
        )
        logger.info("Created MLP context embedder.")
    elif args.context_embedder_type == "only_lin":
        context_embedder_module = nn.Sequential(
            nn.Linear(args.comp_vector_dim, args.context_embed_dim)
        )
        logger.info("Created linear-only context embedder.")
    elif args.context_embedder_type == "TrueCCIP":
        context_embedder_module = TrueCCIP(
            vocab_size=98,  # 95 elements + 3 special tokens
            embedding_dim=128,  # Internal embedding dimension
            max_length=20,  # Max composition sequence length
            n_layers=6,
            n_heads=8,
            output_dim=args.context_embed_dim,
            dropout=0.1
        )
        logger.info(f"Created TrueCCIP context embedder with output_dim={args.context_embed_dim}.")
        
        # Load the pretrained weights
        if args.ccip_checkpoint_path:
            context_embedder_checkpoint = torch.load(args.ccip_checkpoint_path, weights_only=False, map_location='cpu')
            if 'model_state_dict' in context_embedder_checkpoint:
                state_dict = context_embedder_checkpoint['model_state_dict']
            else:
                state_dict = context_embedder_checkpoint
            
            # Handle prefix removal if needed
            if any(key.startswith('composition_encoder.') for key in state_dict.keys()):
                logger.info("Found composition_encoder prefix, removing it...")
                clean_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('composition_encoder.'):
                        new_key = key[len('composition_encoder.'):]
                        clean_state_dict[new_key] = value
                state_dict = clean_state_dict
            
            # Load weights with strict=False to handle dimension mismatches in projection layer
            missing_keys, unexpected_keys = context_embedder_module.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys when loading TrueCCIP: {missing_keys}")
            if unexpected_keys:  
                logger.warning(f"Unexpected keys when loading TrueCCIP: {unexpected_keys}")
            
            logger.info(f"TrueCCIP context embedder loaded from {args.ccip_checkpoint_path}.")
    elif args.context_embedder_type == "none":
        logger.info("No context embedder will be used.")
    else:
        raise ValueError(f"Unknown context_embedder_type: {args.context_embedder_type}")
    
    return context_embedder_module

def initialize_models(args, device: str, context_embedder_module: nn.Module = None):
    """Initialize all required models and return them."""
    logger.info("Loading models...")
    
    # Clear memory before loading
    clear_cuda_memory()
    log_memory_usage("before_loading")
    
    # Enable memory efficient settings and optimizations
    enable_memory_efficient_attention()
    enable_fast_generation_optimizations()
    
    # Load DDPM model
    ddpm_model = Diffusion(d_context=args.context_embed_dim).to(device)
    ddpm_checkpoint = torch.load(args.ddpm_checkpoint, map_location=device, weights_only=False)
    ddpm_model.load_state_dict(ddpm_checkpoint['model_state_dict'])
    ddpm_model.eval()
    logger.info(f"DDPM model loaded from {args.ddpm_checkpoint}")
    
    # Apply optimizations to DDPM model
    if args.use_half_precision and device.type == "cuda":
        ddpm_model = ddpm_model.half()
        logger.info("DDPM model converted to half precision (FP16)")
    
    if args.compile_models:
        try:
            ddpm_model = torch.compile(ddpm_model, mode="reduce-overhead")
            logger.info("DDPM model compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Could not compile DDPM model: {e}")
    
    # Clear memory after loading DDPM
    del ddpm_checkpoint
    clear_cuda_memory()
    
    # Process context embedder
    context_embedder = None
    if context_embedder_module is not None:
        context_embedder = context_embedder_module.to(device)
        context_embedder.eval()
        logger.info(f"Context embedder of type {args.context_embedder_type} loaded.")
        
        # Apply optimizations to context embedder
        if args.use_half_precision and device.type == "cuda":
            # Skip half precision for TrueCCIP due to internal dtype compatibility issues
            if args.context_embedder_type == "TrueCCIP":
                logger.warning("Skipping TrueCCIP half precision conversion due to internal dtype issues")
            else:
                context_embedder = context_embedder.half()
                logger.info("Context embedder converted to half precision (FP16)")
        
        if args.compile_models:
            try:
                # Skip compilation for TrueCCIP with half precision due to dtype compatibility issues
                if args.context_embedder_type == "TrueCCIP" and args.use_half_precision and device.type == "cuda":
                    logger.warning("Skipping TrueCCIP compilation with half precision due to potential dtype issues")
                else:
                    context_embedder = torch.compile(context_embedder, mode="reduce-overhead")
                    logger.info("Context embedder compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile context embedder: {e}")
        
        if isinstance(context_embedder, TrueCCIP):
            logger.info(f"TrueCCIP context embedder with output_dim {context_embedder.output_dim}.")
            # Freeze TrueCCIP parameters  
            for param in context_embedder.parameters():
                param.requires_grad = False
            logger.info("TrueCCIP parameters frozen (requires_grad=False).")
    else:
        logger.warning("No context embedder provided. Contextual conditioning will be absent.")
    
    # Load VAE decoder
    vae_decoder = VAE_Decoder().to(device)
    if args.vae_checkpoint_path:
        vae_checkpoint = torch.load(args.vae_checkpoint_path, map_location=device, weights_only=False)
        vae_decoder.load_state_dict(vae_checkpoint['decoder_state_dict'])
        del vae_checkpoint
        clear_cuda_memory()
    vae_decoder.eval()
    
    # Apply optimizations to VAE decoder
    if args.use_half_precision and device.type == "cuda":
        vae_decoder = vae_decoder.half()
        logger.info("VAE decoder converted to half precision (FP16)")
    
    if args.compile_models:
        try:
            vae_decoder = torch.compile(vae_decoder, mode="reduce-overhead")
            logger.info("VAE decoder compiled with torch.compile")
        except Exception as e:
            logger.warning(f"Could not compile VAE decoder: {e}")
    
    logger.info(f"VAE decoder loaded from {args.vae_checkpoint_path}")
    
    # Load tokenizer
    if args.context_embedder_type == "TrueCCIP":
        tokenizer = CCIPCompositionTokenizer()
        logger.info("Using TrueCCIP tokenizer.")
    else:
        tokenizer = CompositionTokenizer(args.tokenizer_path)
        logger.info(f"Using standard CompositionTokenizer from {args.tokenizer_path}.")
    
    log_memory_usage("after_all_models")
    return ddpm_model, context_embedder, vae_decoder, tokenizer

def generate_latents_with_ddpm(ddpm_model, ddpm_sampler, context_embeddings, 
                              batch_size, latent_height, latent_width, device, args):
    """Generate latents using DDPM sampling process."""
    # Start with random noise
    latents = torch.randn(
        batch_size, NUM_VAE_LATENT_CHANNELS, latent_height, latent_width,
        device=device, dtype=torch.float32
    )
    
    #if args.latent_scale_factor is not None:
    #    latents = latents * args.latent_scale_factor
    
    # DDPM sampling loop
    ddpm_sampler.set_inference_timesteps(args.timesteps)
    
    for i, t in enumerate(ddpm_sampler.timesteps):
        # Convert timestep to proper types
        t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        t_tensor = torch.full((batch_size,), t_scalar, device=device, dtype=torch.long)
        
        # Get time embeddings
        time_embeddings = get_sinusoidal_time_embeddings(t_tensor, args.time_embedding_size).to(device)
        
        # Predict noise
        predicted_noise = ddpm_model(latents, context_embeddings, time_embeddings)
        
        # Apply DDPM sampling step (correct parameter order: timestep, latents, model_output)
        latents = ddpm_sampler.step(t_scalar, latents, predicted_noise)
        
        if i % 50 == 0:
            logger.info(f"DDPM step {i}/{len(ddpm_sampler.timesteps)}")
    
    return latents

def generate_batch_images(comp_vectors_batch, ddpm_model, context_embedder, vae_decoder,
                         ddpm_sampler, device, args):
    """Generate a batch of images efficiently following training script patterns."""
    # Normalize input types: tuple (TrueCCIP), list (to be stacked), or tensor
    if isinstance(comp_vectors_batch, tuple) and len(comp_vectors_batch) == 3:
        tokens, fractions, attention_mask = comp_vectors_batch
        tokens = tokens.to(device, non_blocking=True)
        fractions = fractions.to(device, non_blocking=True).float()
        attention_mask = attention_mask.to(device, non_blocking=True)
        comp_vectors_batch = (tokens, fractions, attention_mask)
        batch_size = tokens.size(0)
        logger.info(f"TrueCCIP batch (tuple) - tokens: {tokens.shape}, fractions: {fractions.shape}")
    elif isinstance(comp_vectors_batch, list) and len(comp_vectors_batch) > 0:
        first = comp_vectors_batch[0]
        if isinstance(first, tuple) and len(first) == 3:
            # TrueCCIP list of tuples -> stack components
            batch_tokens = []
            batch_fractions = []
            batch_attention_masks = []
            for tkns, fracs, att in comp_vectors_batch:
                batch_tokens.append(tkns)
                batch_fractions.append(fracs)
                batch_attention_masks.append(att)
            tokens = torch.stack(batch_tokens).to(device, non_blocking=True)
            fractions = torch.stack(batch_fractions).to(device, non_blocking=True).float()
            attention_mask = torch.stack(batch_attention_masks).to(device, non_blocking=True)
            comp_vectors_batch = (tokens, fractions, attention_mask)
            batch_size = tokens.size(0)
            logger.info(f"TrueCCIP batch - tokens: {tokens.shape}, fractions: {fractions.shape}")
        else:
            # Standard list of tensors -> stack
            comp_vectors_batch = torch.stack(comp_vectors_batch).to(device, non_blocking=True).float()
            batch_size = comp_vectors_batch.size(0)
            logger.info(f"Standard batch - comp_vectors: {comp_vectors_batch.shape}")
    elif isinstance(comp_vectors_batch, torch.Tensor):
        comp_vectors_batch = comp_vectors_batch.to(device, non_blocking=True).float()
        batch_size = comp_vectors_batch.size(0)
        logger.info(f"Standard batch - comp_vectors: {comp_vectors_batch.shape}")
    else:
        raise TypeError(f"Unsupported batch type: {type(comp_vectors_batch)}")
    
    # Apply half precision if enabled
    if args.use_half_precision and device.type == "cuda":
        if isinstance(comp_vectors_batch, tuple):
            # Handle TrueCCIP tuple: (tokens, fractions, attention_mask)
            # Skip half precision conversion for TrueCCIP inputs since model stays in float32
            if args.context_embedder_type == "TrueCCIP":
                logger.info("Keeping TrueCCIP inputs in float32 to match model precision")
            else:
                tokens, fractions, attention_mask = comp_vectors_batch
                logger.info(f"Converting fractions from {fractions.dtype} to half precision")
                fractions = fractions.half()  # Convert fractions to half precision
                # Also convert attention_mask to half precision if it's float
                if attention_mask.dtype == torch.float32:
                    logger.info(f"Converting attention_mask from {attention_mask.dtype} to half precision")
                    attention_mask = attention_mask.half()
                comp_vectors_batch = (tokens, fractions, attention_mask)
        else:
            logger.info(f"Converting comp_vectors from {comp_vectors_batch.dtype} to half precision")
            comp_vectors_batch = comp_vectors_batch.half()
    
    logger.info(f"Starting generation for batch of {batch_size} images...")
    
    with torch.no_grad():
        # Generate context embeddings
        context_embeddings = None
        if context_embedder is not None:
            # Check if this is TrueCCIP using the args.context_embedder_type
            if args.context_embedder_type == "TrueCCIP":
                # For TrueCCIP, handle tuple input
                if isinstance(comp_vectors_batch, tuple) and len(comp_vectors_batch) == 3:
                    tokens, fractions, attention_mask = comp_vectors_batch
                    logger.info(f"Calling TrueCCIP with dtypes: tokens={tokens.dtype}, fractions={fractions.dtype}, attention_mask={attention_mask.dtype}")
                    raw_context_embeddings = context_embedder(tokens, fractions, attention_mask)
                else:
                    logger.error(f"TrueCCIP expects tuple input but got: {type(comp_vectors_batch)}")
                    raise ValueError("TrueCCIP requires (tokens, fractions, attention_mask) tuple input")
                context_embeddings = raw_context_embeddings.unsqueeze(1)
                logger.info(f"TrueCCIP context embeddings: {context_embeddings.shape}")
            else:
                # For other context embedders
                raw_context_embeddings = context_embedder(comp_vectors_batch)
                context_embeddings = raw_context_embeddings.unsqueeze(1)
                logger.info(f"Context embeddings: {context_embeddings.shape}")
        
        # Generate random latents
        latent_height = args.image_size // VAE_DOWNSAMPLE_FACTOR
        latent_width = args.image_size // VAE_DOWNSAMPLE_FACTOR
        
        # Use autocast for mixed precision if half precision is enabled
        if args.use_half_precision and device.type == "cuda":
            logger.info("Using half precision with autocast")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                latents = generate_latents_with_ddpm(
                    ddpm_model, ddpm_sampler, context_embeddings,
                    batch_size, latent_height, latent_width, device, args
                )
        else:
            logger.info("Using standard precision")
            latents = generate_latents_with_ddpm(
                ddpm_model, ddpm_sampler, context_embeddings,
                batch_size, latent_height, latent_width, device, args
            )

        latents = latents / args.latent_scale_factor

        # Ensure latents dtype matches decoder weight dtype when using half precision to avoid conv2d bias mismatch
        try:
            target_dtype = None
            # Grab first conv weight to infer dtype
            for m in vae_decoder.modules():
                if isinstance(m, torch.nn.Conv2d):
                    target_dtype = m.weight.dtype
                    break
            if target_dtype is not None and latents.dtype != target_dtype:
                logger.info(f"Casting latents from {latents.dtype} to {target_dtype} to match VAE decoder")
                latents = latents.to(target_dtype)
            images = vae_decoder(latents)
        except RuntimeError as e:
            if "Input type (float) and bias type" in str(e):
                logger.warning("Decoder dtype mismatch encountered. Retrying decode in float32 without compilation.")
                try:
                    latents = latents.float()
                    vae_decoder.float()
                    images = vae_decoder(latents)
                except Exception as e2:
                    logger.error(f"Retry in float32 failed: {e2}")
                    raise
            else:
                raise
        
        # Convert to proper format
        images = (images + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0.0, 1.0)
        
        # Debug: log image statistics
        logger.info(f"Final images stats: min={images.min():.4f}, max={images.max():.4f}, mean={images.mean():.4f}")
    
    logger.info(f"Batch generation completed. Output shape: {images.shape}")
    return images

def parse_composition_string(comp_str: str) -> dict:
    """Parse composition string like 'Fe:0.5,O:0.3,Al:0.2' into dict."""
    composition = {}
    try:
        for pair in comp_str.split(','):
            element, fraction = pair.strip().split(':')
            composition[element.strip()] = float(fraction.strip())
    except Exception as e:
        logger.error(f"Failed to parse composition '{comp_str}': {e}")
        raise
    return composition

def load_compositions_from_json(json_path: str):
    """Load compositions from a JSON file.

    Supported schemas:
      - List[dict] where each dict is element->fraction
      - Dict[name -> dict]
      - List[{"name": str, "composition": dict|str}]
      - List[str] where each string is "Fe:0.5,O:0.5"
      - {"compositions": <any of the above>}

    Returns:
      (compositions: List[dict], names: List[str])
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Unwrap common container key
    if isinstance(data, dict) and 'compositions' in data and isinstance(data['compositions'], (list, dict)):
        data = data['compositions']

    def to_comp_dict(obj):
        if isinstance(obj, str):
            return parse_composition_string(obj)
        if isinstance(obj, dict):
            # ensure numeric floats
            return {str(k): float(v) for k, v in obj.items()}
        raise ValueError(f"Unsupported composition value type: {type(obj)}")

    # Case: mapping name -> composition
    if isinstance(data, dict):
        compositions, names = [], []
        for name, comp in data.items():
            compositions.append(to_comp_dict(comp))
            names.append(str(name))
        return compositions, names

    # Case: list of entries
    if isinstance(data, list):
        compositions, names = [], []
        for i, item in enumerate(data):
            # Entry has explicit name and composition
            if isinstance(item, dict) and 'composition' in item:
                comp = item['composition']
                name = item.get('name') or item.get('id') or item.get('label') or f"json_{i}"
                compositions.append(to_comp_dict(comp))
                names.append(str(name))
            # Entry is a composition dict directly
            elif isinstance(item, dict):
                compositions.append(to_comp_dict(item))
                names.append(f"json_{i}")
            # Entry is a composition string
            elif isinstance(item, str):
                comp_dict = parse_composition_string(item)
                safe = item.replace(':', '').replace(',', '_').replace(' ', '')
                names.append(f"json_{i}_{safe[:40]}")
                compositions.append(comp_dict)
            else:
                raise ValueError(f"Unsupported item type in list at index {i}: {type(item)}")
        return compositions, names

    raise ValueError("Unsupported JSON schema for compositions. Provide a list or a dict, optionally under the 'compositions' key.")

def validate_and_normalize_composition(comp: dict, tolerance: float = 1e-3, normalize: bool = True) -> dict:
    """Validate a composition and optionally normalize.

    Checks:
      - Non-empty dict
      - All values finite and >= 0
      - Sum close to 1 within tolerance (or normalize if allowed)

    Returns a (possibly normalized) copy of the composition.
    Raises ValueError if validation fails and normalization cannot fix it.
    """
    if not isinstance(comp, dict) or not comp:
        raise ValueError("Composition must be a non-empty dict of element->fraction")

    clean = {}
    total = 0.0
    for k, v in comp.items():
        try:
            val = float(v)
        except Exception:
            raise ValueError(f"Invalid fraction for element '{k}': {v}")
        if not np.isfinite(val):
            raise ValueError(f"Non-finite fraction for element '{k}': {val}")
        if val < 0:
            raise ValueError(f"Negative fraction for element '{k}': {val}")
        clean[str(k)] = val
        total += val

    if total <= 0:
        raise ValueError("Sum of fractions must be > 0")

    if abs(total - 1.0) > tolerance:
        if normalize:
            norm = {k: v / total for k, v in clean.items()}
            logger.debug(f"Normalized composition from sum={total:.6f} to 1.0")
            return norm
        else:
            raise ValueError(f"Composition fractions sum to {total:.6f}, outside tolerance {tolerance}")

    return clean

def create_hdf5_file(output_path: str, max_images: int, image_size: int):
    """Create an HDF5 file for storing generated images and metadata."""
    logger.info(f"Creating HDF5 file: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Create datasets for images and metadata
        f.create_dataset('images', 
                        shape=(max_images, 3, image_size, image_size),
                        dtype=np.float32,
                        compression='gzip',
                        compression_opts=1)
        
        # Create variable-length string type for composition names
        string_type = h5py.special_dtype(vlen=str)
        f.create_dataset('image_names',
                        shape=(max_images,),
                        dtype=string_type)
        
        f.create_dataset('compositions',
                        shape=(max_images,),
                        dtype=string_type)
        
        f.create_dataset('timestamps',
                        shape=(max_images,),
                        dtype=string_type)
        
        # Add metadata attributes
        f.attrs['creation_time'] = datetime.now().isoformat()
        f.attrs['image_size'] = image_size
        f.attrs['num_images'] = 0  # Will be updated as images are added
        
    logger.info("HDF5 file created successfully")

def save_images_to_hdf5(h5_path: str, images: list, names: list, compositions: list, start_idx: int):
    """Save a batch of images to HDF5 file."""
    batch_size = len(images)
    logger.info(f"Saving {batch_size} images to HDF5...")
    
    with h5py.File(h5_path, 'a') as f:
        end_idx = start_idx + batch_size
        
        # Convert images to numpy arrays
        image_arrays = []
        for image in images:
            if isinstance(image, torch.Tensor):
                # Tensor is in [C, H, W] format, values in [0, 1]
                image_array = image.detach().cpu().numpy().astype(np.float32)
            else:
                # PIL Image
                image_array = np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0
            image_arrays.append(image_array)
        
        # Write to HDF5
        timestamp = datetime.now().isoformat()
        f['images'][start_idx:end_idx] = np.stack(image_arrays)
        f['image_names'][start_idx:end_idx] = names
        f['compositions'][start_idx:end_idx] = [json.dumps(comp, cls=NumpyEncoder) for comp in compositions]
        f['timestamps'][start_idx:end_idx] = [timestamp] * batch_size
        
        # Update metadata
        f.attrs['num_images'] = end_idx
    
    logger.info(f"Saved {batch_size} images to HDF5")

def save_images_to_files(images: list, names: list, compositions: list, output_dir: str, args):
    """Save images as individual files (PNG format)."""
    for i, (image, name, comp_dict) in enumerate(zip(images, names, compositions)):
        # Convert to PIL Image
        if isinstance(image, torch.Tensor):
            pil_image = transforms.ToPILImage()(image.cpu())
        else:
            pil_image = image
        
        # Save image
        output_path = os.path.join(output_dir, f"{name}.png")
        pil_image.save(output_path)
        logger.debug(f"Saved {output_path}")

def get_args():
    """Parse command line arguments following training script structure."""
    parser = argparse.ArgumentParser(description="Generate images using DDPM + Context Embedder")
    
    # Path arguments group
    path_group = parser.add_argument_group('Paths')
    path_group.add_argument("--ddpm_checkpoint", type=str, required=True,
                           help="Path to DDPM model checkpoint")
    path_group.add_argument("--vae_checkpoint_path", type=str, 
                           default="checkpoints/vae_epoch_9.pth",
                           help="Path to VAE decoder checkpoint")
    path_group.add_argument("--ccip_checkpoint_path", type=str, default=None,
                           help="Path to CCIP/TrueCCIP model checkpoint")
    path_group.add_argument("--tokenizer_path", type=str, 
                           default=None,
                           help="Path to element embedding JSON file for tokenizer")
    path_group.add_argument("--h5_data_path", type=str,
                           default="data/dataset_comp_image_spectra.h5",
                           help="Path to HDF5 dataset")
    path_group.add_argument("--indices_file", type=str, default=None,
                           help="Path to indices file (if not provided, uses all compositions in dataset)")
    path_group.add_argument("--compositions_json", type=str, default=None,
                           help="Path to JSON file with compositions to generate. Supports list[dict], dict[name->dict], list[{name, composition}], list[str], or an object with 'compositions' field.")
    
    # Model configuration group  
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--image_size", type=int, default=64,
                            help="Generated image size")
    model_group.add_argument("--comp_vector_dim", type=int, default=None,
                            help="Dimension of input composition vector. If None, inferred from tokenizer.")
    model_group.add_argument("--context_embed_dim", type=int, default=64,
                            help="Output dimension of context embedder / input to U-Net for context.")
    model_group.add_argument("--time_embedding_size", type=int, default=320,
                            help="Time embedding size")
    model_group.add_argument("--latent_scale_factor", type=float, default=0.18215,
                            help="VAE latent scale factor")
    model_group.add_argument("--context_embedder_type", type=str, default="linear", 
                            choices=["linear", "mlp", "only_lin", "none", "TrueCCIP"],
                            help="Type of context embedder")
    
    # Generation parameters group
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument("--num_samples", type=int, default=None,
                          help="Number of samples to generate (if not specified, generates all available samples)")
    gen_group.add_argument("--batch_size", type=int, default=4,
                          help="Batch size for generation")
    gen_group.add_argument("--specific_indices", type=str, default=None,
                          help="Comma-separated list of specific indices to generate")
    gen_group.add_argument("--custom_compositions", type=str, default=None,
                          help="Semicolon-separated custom compositions (e.g., 'Fe:0.5,O:0.5;Al:0.3,O:0.7')")
    gen_group.add_argument("--json_range", type=str, default=None,
                          help="Range slice start:end (end exclusive) applied to --compositions_json before generation. Supports negative indices and open bounds, e.g. 0:100, 100:, :500, -500:." )
    gen_group.add_argument("--json_slice_only", action="store_true",
                          help="Load JSON, apply --json_range if provided, report selected count, then exit without model loading.")
    gen_group.add_argument("--seed", type=int, default=42,
                          help="Random seed")
    
    # DDPM parameters group
    ddpm_group = parser.add_argument_group('DDPM Parameters')
    ddpm_group.add_argument("--timesteps", type=int, default=1000,
                           help="Number of denoising timesteps")
    ddpm_group.add_argument("--beta_start", type=float, default=0.00085,
                           help="DDPM beta start")
    ddpm_group.add_argument("--beta_end", type=float, default=0.0120,
                           help="DDPM beta end")
    
    # Output and logging group
    output_group = parser.add_argument_group('Output and Logging')
    output_group.add_argument("--output_dir", type=str, default="generated_samples",
                             help="Output directory for generated images")
    output_group.add_argument("--save_to_hdf5", action="store_true",
                             help="Save images to HDF5 file instead of individual PNG files")
    output_group.add_argument("--hdf5_path", type=str, default=None,
                             help="Path to HDF5 file for saving images (auto-generated if not specified)")
    output_group.add_argument("--save_both_formats", action="store_true",
                             help="Save images in both PNG and HDF5 formats")
    output_group.add_argument("--verbose", action="store_true",
                             help="Enable verbose logging")
    output_group.add_argument("--timing_report", type=str, default=None,
                             help="Path to save timing report JSON file")
    output_group.add_argument("--composition_tolerance", type=float, default=1e-3,
                             help="Allowed absolute tolerance for composition sum to equal 1.0 before normalization")
    output_group.add_argument("--no_normalize", action="store_true",
                             help="If set, do not normalize compositions; error if sum deviates beyond tolerance")
    
    # Performance group
    perf_group = parser.add_argument_group('Performance Settings')
    perf_group.add_argument("--device", type=str, default="auto",
                           help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    perf_group.add_argument("--use_half_precision", action="store_true",
                           help="Use half precision (FP16) for faster generation")
    perf_group.add_argument("--compile_models", action="store_true",
                           help="Use torch.compile for faster model execution (PyTorch 2.0+)")
    perf_group.add_argument("--fast_sampling", action="store_true",
                           help="Use faster sampling with fewer timesteps")
    perf_group.add_argument("--num_workers", type=int, default=0,
                           help="Number of DataLoader workers (0 = no multiprocessing)")
    perf_group.add_argument("--pin_memory", action="store_true",
                           help="Pin CPU memory for faster GPU transfers (CUDA only useful)")

    return parser.parse_args()

def main():
    args = get_args()
    
    # Determine comp_vector_dim from tokenizer if not provided (following training script pattern)
    if args.comp_vector_dim is None:
        try:
            with open(args.tokenizer_path, 'r') as f:
                tokenizer_data = json.load(f)
            # Get dimension from the first embedding vector
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
    
    # Setup device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Validate arguments (following training script pattern)
    if args.context_embedder_type == "TrueCCIP" and args.ccip_checkpoint_path is None:
        logger.error("--ccip_checkpoint_path is required when --context_embedder_type is TrueCCIP")
        return 1
    
    # Setup output directory and logging
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir, args.verbose)
    
    # Determine actual number of samples to generate
    json_compositions = None
    json_comp_names = None
    if args.compositions_json:
        try:
            json_compositions, json_comp_names = load_compositions_from_json(args.compositions_json)
            total_before = len(json_compositions)
            if args.json_range:
                if ':' not in args.json_range:
                    logger.error("--json_range must contain ':' (start:end)")
                    return 1
                start_str, end_str = args.json_range.split(':', 1)
                try:
                    start = int(start_str) if start_str.strip() != '' else None
                    end = int(end_str) if end_str.strip() != '' else None
                except ValueError:
                    logger.error(f"Invalid integers in --json_range '{args.json_range}'")
                    return 1
                json_compositions = json_compositions[slice(start, end)]
                json_comp_names = json_comp_names[slice(start, end)]
                logger.info(f"Applied json_range {args.json_range}: selected {len(json_compositions)} / {total_before}")
            actual_num_samples = len(json_compositions)
            logger.info(f"Using {actual_num_samples} compositions from JSON file: {args.compositions_json}")
            if args.json_slice_only:
                logger.info("--json_slice_only set; exiting after slice test.")
                return 0
        except Exception as e:
            logger.error(f"Failed to load compositions JSON '{args.compositions_json}': {e}")
            return 1
    elif args.custom_compositions:
        comp_strings = args.custom_compositions.split(';')
        actual_num_samples = len(comp_strings)
        logger.info(f"Using {actual_num_samples} custom compositions")
    elif args.specific_indices:
        indices = [int(idx) for idx in args.specific_indices.split(',')]
        actual_num_samples = len(indices)
        logger.info(f"Using {actual_num_samples} specific indices")
    else:
        # Determine available samples from dataset or indices file
        if args.indices_file is not None:
            # Use indices file
            all_indices = indice_import(args.indices_file)
            if args.num_samples is None:
                actual_num_samples = len(all_indices)
                logger.info(f"--num_samples not specified, will generate all {actual_num_samples} available samples from indices file")
            else:
                actual_num_samples = min(args.num_samples, len(all_indices))
                logger.info(f"Will generate {actual_num_samples} samples (requested: {args.num_samples}, available: {len(all_indices)})")
        else:
            # No indices file provided - need to check dataset size later
            logger.info("No indices file provided, will use all compositions from dataset")
            actual_num_samples = None  # Will be determined after loading dataset
    
    # Setup HDF5 file if needed (defer creation if actual_num_samples is unknown)
    h5_path = None
    create_h5_later = False
    if args.save_to_hdf5 or args.save_both_formats:
        if args.hdf5_path:
            h5_path = args.hdf5_path
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            h5_path = os.path.join(args.output_dir, f"generated_images_{timestamp}.h5")
        
        if actual_num_samples is not None:
            create_hdf5_file(h5_path, actual_num_samples, args.image_size)
            logger.info(f"HDF5 output: {h5_path}")
        else:
            create_h5_later = True
            logger.info(f"HDF5 output: {h5_path} (will be created after determining dataset size)")
    
    if not args.save_to_hdf5 or args.save_both_formats:
        logger.info(f"PNG output directory: {args.output_dir}")
    
    # Initialize timing logger
    timing_file = args.timing_report or os.path.join(args.output_dir, "timing_report.json")
    timer = TimingLogger(timing_file)
    
    # Apply fast sampling optimization
    if args.fast_sampling:
        args.timesteps = min(args.timesteps, 50)  # Very fast sampling
        logger.info(f"Fast sampling enabled: Using {args.timesteps} timesteps")
    
    # Set memory allocation strategy for better memory management
    if device.type == "cuda":
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Memory efficiency warnings for GPU devices
    if device.type in ["cuda", "mps"] and args.batch_size > 32:
        logger.warning(f"Large batch size detected ({args.batch_size}). Consider reducing for memory efficiency.")
    
    logger.info(f"Starting DDPM generation on {device}")
    logger.info(f"Context embedder type: {args.context_embedder_type}")
    
    # Log optimization settings
    if args.use_half_precision:
        logger.info("Half precision (FP16) enabled for faster generation")
    if args.compile_models:
        logger.info("Model compilation enabled for faster execution")
    if args.fast_sampling:
        logger.info("Fast sampling enabled with reduced timesteps")
    if hasattr(args, 'num_workers') and args.num_workers:
        logger.info(f"Data loading workers: {args.num_workers}")
    if hasattr(args, 'pin_memory') and args.pin_memory:
        logger.info("Pinned memory enabled for faster GPU transfers")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    elif device.type == "mps":
        torch.mps.manual_seed(args.seed)
    
    try:
        # Create context embedder module (following training script pattern)
        context_embedder_module = create_context_embedder_module(args)
        
        # Load models (following training script pattern)
        timer.start_timer("model_loading")
        ddpm_model, context_embedder, vae_decoder, tokenizer = initialize_models(
            args, device, context_embedder_module
        )
        timer.end_timer("model_loading")
        
        # Setup DDPM sampler
        generator = torch.Generator(device=device).manual_seed(args.seed)
        ddpm_sampler = DDPMSampler(generator, num_training_steps=args.timesteps,
                                  beta_start=args.beta_start, beta_end=args.beta_end)
        
        # Prepare compositions
        compositions = []
        composition_names = []
        
        if args.compositions_json:
            # Use compositions from JSON file loaded earlier
            compositions = json_compositions
            # Use provided names (already generated fallbacks if missing)
            composition_names = list(json_comp_names)
            logger.info(f"Using {len(compositions)} compositions from JSON for generation")
        elif args.custom_compositions:
            # Use custom compositions
            comp_strings = args.custom_compositions.split(';')
            for i, comp_str in enumerate(comp_strings):
                comp_dict = parse_composition_string(comp_str.strip())
                compositions.append(comp_dict)
                composition_names.append(f"custom_{i}_{comp_str.replace(':', '').replace(',', '_')}")
            logger.info(f"Using {len(compositions)} custom compositions")
            
        else:
            # Load from dataset (following training script pattern)
            image_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((args.image_size, args.image_size), antialias=True),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            
            dataset = HDF5CompositionImageDataset(
                h5_path=args.h5_data_path,
                transform=image_transforms,
                comp_format='dict',
                tokenizer=tokenizer
            )
            
            if args.specific_indices:
                indices = [int(idx) for idx in args.specific_indices.split(',')]
            elif args.indices_file is not None:
                # Use indices file
                all_indices = indice_import(args.indices_file)
                if args.num_samples is None:
                    indices = all_indices
                    logger.info(f"Using all {len(indices)} available indices from file")
                else:
                    indices = all_indices[:args.num_samples]
                    logger.info(f"Using first {len(indices)} indices from file")
            else:
                # No indices file - use all samples from dataset
                dataset_size = len(dataset)
                if args.num_samples is None:
                    indices = list(range(dataset_size))
                    logger.info(f"Using all {len(indices)} samples from dataset")
                else:
                    indices = list(range(min(args.num_samples, dataset_size)))
                    logger.info(f"Using first {len(indices)} samples from dataset")
                
                # Update actual_num_samples if it was None
                if actual_num_samples is None:
                    actual_num_samples = len(indices)
                    logger.info(f"Determined actual_num_samples = {actual_num_samples}")
                    
                    # Create HDF5 file now if needed
                    if create_h5_later:
                        create_hdf5_file(h5_path, actual_num_samples, args.image_size)
                        logger.info(f"Created HDF5 file: {h5_path}")
            
            for idx in indices:
                comp_dict = dataset.get_composition_dict(idx)
                compositions.append(comp_dict)
                composition_names.append(f"sample_{idx}")
            
            logger.info(f"Loaded {len(compositions)} compositions from dataset")
        
        # Validate and optionally normalize compositions, then tokenize
        timer.start_timer("tokenization")
        tokenized_compositions = []
        for comp_dict in compositions:
            try:
                validated = validate_and_normalize_composition(
                    comp_dict,
                    tolerance=getattr(args, 'composition_tolerance', 1e-3),
                    normalize=not getattr(args, 'no_normalize', False),
                )
            except Exception as e:
                logger.error(f"Invalid composition {comp_dict}: {e}")
                raise
            # Handle different tokenizer types (following training script pattern)
            tokenized_result = tokenizer.tokenize(validated)
            tokenized_compositions.append(tokenized_result)
        timer.end_timer("tokenization")
        
        # Batch generation
        if tokenized_compositions:
            logger.info(f"Generating {len(tokenized_compositions)} images...")
            timer.start_timer("batch_generation")

            total_images_generated = 0
            total_batches = (len(tokenized_compositions) + args.batch_size - 1) // args.batch_size

            if args.num_workers and args.num_workers > 0:
                logger.info("Using DataLoader for batch processing")
                dataloader = DataLoader(
                    CompositionTokensDataset(tokenized_compositions),
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(args.pin_memory and torch.cuda.is_available()),
                    collate_fn=collate_tokenized,
                    drop_last=False,
                )
                logger.info(f"Created DataLoader with {args.num_workers} workers, pin_memory={args.pin_memory and torch.cuda.is_available()}")

                consumed = 0
                for batch_idx, batch in enumerate(dataloader, start=1):
                    if isinstance(batch, tuple):
                        current_bs = batch[0].size(0)
                    else:
                        current_bs = batch.size(0)
                    batch_names = composition_names[consumed:consumed + current_bs]

                    logger.info(f"Generating batch {batch_idx}/{total_batches} ({current_bs} images)")
                    try:
                        images = generate_batch_images(
                            batch, ddpm_model, context_embedder,
                            vae_decoder, ddpm_sampler, device, args
                        )
                        logger.info(f"Batch {batch_idx} completed successfully")

                        image_list = [images[i] for i in range(images.size(0))]
                        batch_comp_dicts = compositions[consumed:consumed + current_bs]

                        if args.save_to_hdf5 or args.save_both_formats:
                            save_images_to_hdf5(h5_path, image_list, batch_names, batch_comp_dicts, total_images_generated)
                        if not args.save_to_hdf5 or args.save_both_formats:
                            save_images_to_files(image_list, batch_names, batch_comp_dicts, args.output_dir, args)

                        total_images_generated += len(image_list)
                        consumed += current_bs
                        logger.info(f"Saved {len(image_list)} images. Total generated: {total_images_generated}")
                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {e}")
                        raise
            else:
                # Simple batching fallback
                for batch_start in range(0, len(tokenized_compositions), args.batch_size):
                    batch_end = min(batch_start + args.batch_size, len(tokenized_compositions))
                    batch_compositions = tokenized_compositions[batch_start:batch_end]
                    batch_names = composition_names[batch_start:batch_end]

                    batch_num = batch_start // args.batch_size + 1
                    logger.info(f"Generating batch {batch_num}/{total_batches} ({batch_end - batch_start} images)")

                    try:
                        images = generate_batch_images(
                            batch_compositions, ddpm_model, context_embedder,
                            vae_decoder, ddpm_sampler, device, args
                        )
                        logger.info(f"Batch {batch_num} completed successfully")

                        image_list = [images[i] for i in range(images.size(0))]
                        batch_comp_dicts = []
                        for i in range(len(batch_names)):
                            batch_comp_dicts.append(compositions[batch_start + i])

                        if args.save_to_hdf5 or args.save_both_formats:
                            save_images_to_hdf5(h5_path, image_list, batch_names, batch_comp_dicts, total_images_generated)
                        if not args.save_to_hdf5 or args.save_both_formats:
                            save_images_to_files(image_list, batch_names, batch_comp_dicts, args.output_dir, args)

                        total_images_generated += len(image_list)
                        logger.info(f"Saved {len(image_list)} images. Total generated: {total_images_generated}")
                    except Exception as e:
                        logger.error(f"Error in batch {batch_num}: {e}")
                        raise

            timer.end_timer("batch_generation")
            logger.info(f"Successfully generated {total_images_generated} images")
        
        # Save timing report
        timer.save_timing_report()
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise
    
    finally:
        logger.info("Script completed")

def enable_memory_efficient_attention():
    """Enable memory efficient attention if available."""
    try:
        # Try to enable memory efficient attention
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("Flash attention enabled for memory efficiency")
    except:
        logger.info("Flash attention not available, using standard attention")

def enable_fast_generation_optimizations():
    """Enable various optimizations for faster generation."""
    # Enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Enable optimized attention if available
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TensorFloat-32 (TF32) enabled for faster computation")
    except:
        logger.info("TF32 not available")
    
    # Disable gradient computation globally for inference
    torch.set_grad_enabled(False)
    logger.info("Global gradient computation disabled for inference")

def get_available_memory():
    """Get available GPU memory in bytes."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        return total_memory - allocated_memory
    return 0

def estimate_memory_per_sample(image_size: int, use_half_precision: bool = False):
    """Estimate memory needed per sample for generation."""
    # Latent space is 1/8 of image size
    latent_size = image_size // 8
    
    # Memory components:
    # 1. Latent tensor: 4 channels * latent_size^2 * 4 bytes (float32)
    # 2. Decoded image: 3 channels * image_size^2 * 4 bytes (float32)
    # 3. Context embeddings: ~1000 bytes (rough estimate)
    # 4. Model activations during forward pass: ~5x the base memory
    
    base_memory = (4 * latent_size * latent_size + 3 * image_size * image_size) * 4
    if use_half_precision:
        base_memory //= 2
    
    # Add overhead for model activations
    total_memory = base_memory * 6  # 6x multiplier for safety
    
    return total_memory


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)
