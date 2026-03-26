import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import logging

from sd.encoder import VAE_Encoder
from sd.decoder import VAE_Decoder
from sd.diffusion import Diffusion, TimeEmbedding
from sd.ddpm import DDPMSampler
from sd.data_helper import HDF5CompositionImageDataset, indice_import
from torchvision import transforms
from embedding.element_tokenizer import CompositionTokenizer

import argparse
from PIL import ImageDraw, ImageFont


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

def get_time_embedding(timestep, embedding_dim=320):
    """
    Wrapper function for single timestep embedding generation.
    Args:
        timestep: Single timestep value (int or float)
        embedding_dim: Dimension of the embedding (default: 320)
    Returns:
        A tensor of shape (1, embedding_dim)
    """
    if isinstance(timestep, (int, float)):
        timestep_tensor = torch.tensor([timestep], dtype=torch.float32)
    else:
        timestep_tensor = timestep.view(-1)  # Ensure 1D
    
    return get_sinusoidal_time_embeddings(timestep_tensor, embedding_dim)

def annotate_image(image, text):
    from PIL import ImageFont
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("fonts/Minimal4.ttf", 15)  # or path to any small TTF font
    # Wrap text for small images
    max_chars = 15
    lines = []
    for l in text.split("\n"):
        while len(l) > max_chars:
            lines.append(l[:max_chars])
            l = l[max_chars:]
        lines.append(l)
    wrapped = "\n".join(lines)
    draw.text((2, 2), wrapped, fill=(255, 255, 255), font=font)
    return image

def generate_batch(comp_vectors, model, context_embedder, vae_decoder, ddpm_sampler, 
                   image_size, device, latent_scale_factor, time_embedding_size=320):
    """Generate a batch of images from composition vectors."""
    # Handle both tuple (CCIP) and tensor (regular) formats
    if isinstance(comp_vectors, tuple):
        # CCIP format: get batch size from first element (tokens)
        batch_size = comp_vectors[0].size(0)
    else:
        # Regular tensor format
        batch_size = comp_vectors.size(0)
    
    logging.info(f"Starting batch generation for {batch_size} images")
    
    # Get context embeddings
    logging.info("🧠 Computing context embeddings...")
    if isinstance(comp_vectors, tuple):
        # CCIP format: unpack tuple and pass as separate arguments
        tokens, fractions, attention_mask = comp_vectors
        context = context_embedder(tokens, fractions, attention_mask).unsqueeze(1)
    else:
        # Regular tensor format
        context = context_embedder(comp_vectors).unsqueeze(1)
    logging.info(f"Context embeddings shape: {context.shape}")
    
    # Initialize latent noise
    latent_height = image_size // 8
    latent_width = image_size // 8
    logging.info(f"Latent dimensions: {latent_height}x{latent_width}")
    
    noise = torch.randn((batch_size, 4, latent_height, latent_width), 
                       generator=ddpm_sampler.generator, device=device)
    latents = noise
    logging.info(f"Initial noise shape: {latents.shape}")
    
    # Denoising loop
    total_steps = len(ddpm_sampler.timesteps)
    logging.info(f"Starting denoising process with {total_steps} timesteps")
    
    for i, timestep in enumerate(ddpm_sampler.timesteps):
        if i % 50 == 0 or i == total_steps - 1:  # Log every 50 steps and the last step
            logging.info(f"Denoising step {i+1}/{total_steps} (timestep {timestep})")
        
        time_embedding = get_time_embedding(timestep, time_embedding_size).to(device)
        model_output = model(latents, context, time_embedding)
        latents = ddpm_sampler.step(timestep, latents, model_output)
    
    logging.info("Decoding latents to images...")
    decoded = vae_decoder(latents/latent_scale_factor)
    images = (decoded.clamp(-1, 1) + 1) / 2.0
    
    logging.info(f"Generated {batch_size} images with shape {images.shape}")
    return images


def custom_collate_fn(batch):
    """Custom collate function to handle CCIP tokenizer output."""
    comp_data, images = zip(*batch)
    
    # Check if comp_data contains tuples (from CCIP tokenizer)
    if isinstance(comp_data[0], tuple) and len(comp_data[0]) == 3:
        # CCIP tokenizer returns (tokens, fractions, attention_mask)
        # We need to stack each component separately
        tokens_batch = torch.stack([item[0] for item in comp_data])
        fractions_batch = torch.stack([item[1] for item in comp_data])
        attention_mask_batch = torch.stack([item[2] for item in comp_data])
        comp_vectors = (tokens_batch, fractions_batch, attention_mask_batch)
    else:
        # Regular tensor data
        comp_vectors = torch.stack(comp_data) if isinstance(comp_data[0], torch.Tensor) else comp_data
    
    images = torch.stack(images)
    return comp_vectors, images

def generate_from_dataset(dataset, selected_indices, batch_size, model, context_embedder, vae_decoder, 
                         ddpm_sampler, image_size, device, output_dir, annotate_fn, latent_scale_factor, time_embedding_size=320):
    subset = Subset(dataset, selected_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    model.eval()
    vae_decoder.eval()
    total_batches = len(dataloader)
    with torch.no_grad():
        for batch_idx, (comp_vectors, _) in enumerate(dataloader):
            logging.info(f"Processing batch {batch_idx + 1}/{total_batches} in dataset generation...")
            # Handle both tuple (CCIP) and tensor (regular) formats
            if isinstance(comp_vectors, tuple):
                # CCIP format: move each component to device
                tokens, fractions, attention_mask = comp_vectors
                comp_vectors = (tokens.to(device), fractions.to(device), attention_mask.to(device))
            else:
                # Regular tensor format
                comp_vectors = comp_vectors.to(device).float()
            images = generate_batch(comp_vectors, model, context_embedder, vae_decoder, ddpm_sampler, 
                                  image_size, device, latent_scale_factor, time_embedding_size)

            for i, (img, idx) in enumerate(zip(images, selected_indices[batch_idx*batch_size:batch_idx*batch_size+len(images)])):
                pil_img = transforms.ToPILImage()(img.cpu())
                # Robust composition string extraction
                comp_dict = dataset.get_composition_dict(idx)
                composition_str = ', '.join([f'{k}:{v:.2f}' for k, v in comp_dict.items()])
                if annotate_fn is not None:
                    pil_img = annotate_fn(pil_img, f"#{idx}\n{composition_str}")
                pil_img.save(os.path.join(output_dir, f"sample_{idx}.png"))


def generate_images(
    dataset,
    indices,
    model,
    context_embedder,
    vae_decoder,
    ddpm_sampler,
    annotate_fn,
    batch_size=8,
    image_size=64,
    output_dir="generated_samples",
    device="cpu",
    latent_scale_factor=1.0,
    time_embedding_size=320
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generate_from_dataset(
        dataset, indices, batch_size, model, context_embedder,
        vae_decoder, ddpm_sampler, image_size, device, output_dir, annotate_fn, latent_scale_factor, time_embedding_size
    )

if __name__ == "__main__":
    # Legacy usage mode: assemble all components and call generate_images
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/ddpm/cluster_ddpm/ddpm_epoch_10.pth")
    parser.add_argument("--decoder_checkpoint_path", type=str, default="checkpoints/vae_epoch_9.pth")
    parser.add_argument("--output_dir", type=str, default="generated_samples")
    parser.add_argument("--specific_indices", type=str, default=None, help="Comma-separated list of specific indices to generate")
    parser.add_argument("--h5_data_path", type=str, default='data/dataset_comp_image_spectra.h5')
    parser.add_argument("--train_indices_path", type=str, default='splits/split_test_indices.txt')
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--comp_vector_dim", type=int, default=16)
    parser.add_argument("--context_embed_dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=50)
    parser.add_argument("--beta_start", type=float, default=0.00085)
    parser.add_argument("--beta_end", type=float, default=0.0120)
    parser.add_argument("--tokenizer_path", type=str, default="embedding/tokenizers/megnet16-embedding.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent_scale_factor", type=float, default=1.0, help="Scale factor for latent space (default: 1.0)")
    parser.add_argument("--time_embedding_size", type=int, default=320, help="Size of time embeddings (default: 320)")
    args = parser.parse_args()

    local_rank = args.local_rank
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"



    tokenizer = CompositionTokenizer(args.tokenizer_path)
    model = Diffusion().to(device)
    context_embedder = nn.Linear(args.comp_vector_dim, args.context_embed_dim).to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    context_embedder.load_state_dict(checkpoint['context_embedder_state_dict'])
    vae_decoder = VAE_Decoder().to(device)
    decoder_checkpoint = torch.load(args.decoder_checkpoint_path, map_location=torch.device(device), weights_only=False)
    vae_decoder.load_state_dict(decoder_checkpoint['decoder_state_dict'])
    vae_decoder.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    ddpm_sampler = DDPMSampler(generator, num_training_steps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)
    ddpm_sampler.set_inference_timesteps(args.timesteps)
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = HDF5CompositionImageDataset(
        h5_path=args.h5_data_path,
        transform=image_transforms,
        comp_format='dict',
        tokenizer=tokenizer
    )
    indices = indice_import(args.train_indices_path)
    selected_indices = list(map(int, args.specific_indices.split(","))) if args.specific_indices else indices[:32]

    generate_images(
        dataset=dataset,
        indices=selected_indices,
        model=model,
        context_embedder=context_embedder,
        vae_decoder=vae_decoder,
        ddpm_sampler=ddpm_sampler,
        annotate_fn=annotate_image,
        batch_size=args.batch_size,
        image_size=args.image_size,
        output_dir=args.output_dir,
        device=device,
        latent_scale_factor=args.latent_scale_factor,
        time_embedding_size=args.time_embedding_size
    )
    print(f"Image generation completed. Samples saved to {args.output_dir}")