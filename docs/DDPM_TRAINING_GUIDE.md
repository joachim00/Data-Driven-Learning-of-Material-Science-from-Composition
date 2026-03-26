# DDPM Training with Context Conditioning

Train the diffusion U-Net with composition conditioning using `train_ddpm.py`. Supports linear/MLP/only_lin/no context and TrueCCIP.

## Prerequisites
- Dataset: `data/dataset_comp_image_spectra.h5`
- Train indices: `splits/split_train_indices.txt`
- Tokenizer: `embedding/tokenizers/megnet16-embedding.json` (for linear/MLP modes)
- VAE encoder checkpoint for latent generation: `--vae_checkpoint_path checkpoints/vae/vae_best.pth`
- Optional TrueCCIP weights: `checkpoints/true_ccip/true_ccip_best.pth`

## Command examples
Linear/MLP conditioning:
```bash
torchrun --nproc_per_node <gpus> train_ddpm.py \
  --h5_data_path data/dataset_comp_image_spectra.h5 \
  --train_indices_path splits/split_train_indices.txt \
  --vae_checkpoint_path checkpoints/vae/vae_best.pth \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --context_embedder_type mlp \
  --context_embed_dim 64 \
  --checkpoint_dir checkpoints/ddpm_mlp \
  --batch_size 4 --num_epochs 100 --timesteps 1000
```

TrueCCIP conditioning:
```bash
torchrun --nproc_per_node <gpus> train_ddpm.py \
  --h5_data_path data/dataset_comp_image_spectra.h5 \
  --train_indices_path splits/split_train_indices.txt \
  --vae_checkpoint_path checkpoints/vae/vae_best.pth \
  --context_embedder_type TrueCCIP \
  --ccip_checkpoint_path checkpoints/true_ccip/true_ccip_best.pth \
  --context_embed_dim 64 \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --checkpoint_dir checkpoints/ddpm_true_ccip \
  --batch_size 4 --num_epochs 100 --timesteps 1000
```

## Key arguments
- Model: `--image_size`, `--context_embed_dim`, `--time_embedding_size`, `--latent_scale_factor`.
- Context: `--context_embedder_type {linear,mlp,only_lin,none,TrueCCIP}`, `--ccip_checkpoint_path` (when TrueCCIP), `--comp_vector_dim` (auto-inferred if omitted for linear/MLP).
- Training: `--batch_size`, `--learning_rate`, `--weight_decay`, `--num_epochs`, `--grad_clip_norm`, `--use_amp`.
- Scheduler: `--lr_scheduler_type {cosine,steplr}` plus scheduler-specific flags.
- DDP/Data: `--num_workers`, `--pin_memory`, `--ddp_find_unused_parameters`.

## Outputs
- Checkpoints in `--checkpoint_dir` (per-save interval).
- Logs to stdout; consider redirecting when running long jobs.

## Tips
- Ensure `context_embed_dim` matches the checkpoint you will use for generation.
- For TrueCCIP, parameters are frozen; only the diffusion model trains.
- Start with small batch sizes; scale up after confirming stability.
