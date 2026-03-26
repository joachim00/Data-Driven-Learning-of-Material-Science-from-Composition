"""
DDPM (Denoising Diffusion Probabilistic Models) sampler implementation.

This module implements the noise scheduling and sampling procedures for DDPM
as described in Ho et al. (2020) "Denoising Diffusion Probabilistic Models".

Reference: https://arxiv.org/pdf/2006.11239.pdf
"""

import torch
import numpy as np


class DDPMSampler:
    """
    DDPM noise scheduler for diffusion model training and inference.
    
    Implements the forward diffusion process (adding noise) and reverse
    sampling process (denoising) using a linear beta schedule.
    
    Args:
        generator: PyTorch random number generator for reproducibility.
        num_training_steps: Number of diffusion steps (default: 1000).
        beta_start: Starting value for noise schedule (default: 0.00085).
        beta_end: Ending value for noise schedule (default: 0.0120).
    
    Attributes:
        betas: Noise schedule values for each timestep.
        alphas: 1 - betas for each timestep.
        alphas_cumprod: Cumulative product of alphas.
        timesteps: Array of timestep indices for sampling.
    
    Example:
        >>> generator = torch.Generator().manual_seed(42)
        >>> sampler = DDPMSampler(generator, num_training_steps=1000)
        >>> sampler.set_inference_timesteps(50)
        >>> for t in sampler.timesteps:
        ...     noise_pred = model(latents, t)
        ...     latents = sampler.step(t, latents, noise_pred)
    """

    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120
    ):
        # Linear schedule in sqrt space, then squared (scaled linear schedule)
        self.betas = torch.linspace(
            beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype=torch.float32
        ) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_train_timesteps = num_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps: int = 50) -> None:
        """
        Configure the sampler for inference with a reduced number of steps.
        
        Args:
            num_inference_steps: Number of denoising steps during inference.
                Fewer steps means faster generation but potentially lower quality.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """Calculate the previous timestep in the sampling schedule."""
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        return prev_t
    
    def _get_variance(self, timestep: int) -> torch.Tensor:
        """
        Compute variance for the reverse diffusion step.
        
        Implements formula (7) from the DDPM paper for computing the
        posterior variance used in sampling.
        """
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # Posterior variance: equation (7) from DDPM paper
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        variance = torch.clamp(variance, min=1e-20)

        return variance
    
    def set_strength(self, strength: float = 1.0) -> None:
        """
        Configure noise strength for image-to-image generation.
        
        Args:
            strength: Amount of noise to add (0 to 1).
                1.0 means full denoising from pure noise.
                0.0 means minimal changes to input image.
        """
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(
        self,
        timestep: int,
        latents: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform one reverse diffusion step.
        
        Given the current noisy latents and the model's noise prediction,
        compute the denoised latents for the previous timestep.
        
        Args:
            timestep: Current timestep index.
            latents: Current noisy latent representation.
            model_output: Predicted noise from the diffusion model.
        
        Returns:
            Denoised latents for the previous timestep.
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        # Compute alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # Predict original sample x_0: equation (15)
        pred_original_sample = (latents - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5

        # Compute coefficients for equation (7)
        pred_original_sample_coeff = (alpha_prod_t_prev ** 0.5 * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t

        # Compute predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        # Add noise for t > 0
        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype
            )
            variance = (self._get_variance(t) ** 0.5) * noise
        
        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample
    
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Add noise to samples for the forward diffusion process.
        
        Implements q(x_t | x_0) from equation (4) of the DDPM paper.
        
        Args:
            original_samples: Clean samples to add noise to.
            timesteps: Timestep indices determining noise level.
        
        Returns:
            Tuple of (noisy_samples, noise) where noise is the added Gaussian noise.
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype
        )
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples, noise