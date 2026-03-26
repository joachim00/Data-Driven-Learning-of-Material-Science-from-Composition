# Composition-Conditioned Diffusion for Microstructure Generation

Diffusion-based pipeline for synthesizing 64×64 microstructure images conditioned on chemical compositions. The stack combines:
- **VAE** for latent compression (8× downsample, 4 channels)
- **Context encoders**: standard linear/MLP or **True CCIP** (token-level transformer over elements)
- **DDPM U-Net** with cross-attention on context

## Repository Layout
```
sd/                    Core models (Diffusion U-Net, DDPM scheduler, VAE, context embedders, data helpers)
embedding/             Composition tokenizers and vocab
configs/               YAML configs (True CCIP training)
scripts/               Data utilities, evaluation, visualization helpers
bash_scripts/          Convenience shell entrypoints for training/generation
checkpoints/           Expected location for saved weights
data/                  HDF5 dataset (dataset_comp_image_spectra.h5)
splits/                Train/test index files
generate_ddpm.py       Image generation entrypoint
train_vae.py           VAE training
train_true_ccip.py     True CCIP contrastive training (DDP, YAML-driven)
train_ddpm.py          DDPM training with configurable context embedder
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
```
Requires Python ≥3.10 and PyTorch ≥2.0.

## Data and Checkpoints
- Dataset: `data/dataset_comp_image_spectra.h5` (images + compositions). Index files live in `splits/` (e.g., `split_train_indices.txt`, `split_test_indices.txt`).
- Tokenizer embeddings: e.g., `embedding/tokenizers/megnet16-embedding.json`.
- Checkpoints are expected under `checkpoints/` by default (script flags allow overrides).

## Training Pipeline

### 1) Train VAE
```bash
python train_vae.py \
  --dataset_path data/dataset_comp_image_spectra.h5 \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --train_indices_path splits/split_train_indices.txt \
  --checkpoint_dir checkpoints/vae \
  --batch_size 256 --learning_rate 1e-4 --num_epochs 20
```
Key flags: `--latent_scaling_factor` (default 0.18215), `--kl_weight`, `--patience` for early stopping.

### 2) Train True CCIP (Distributed)
```bash
torchrun --nproc_per_node <gpus> train_true_ccip.py \
  --config configs/true_ccip_config.yaml
```
Config controls data paths, transformer width/depth, projection heads, AMP, and logging (`logging.output_dir` e.g., `checkpoints/true_ccip`). Use `--resume` to continue.

### 3) Train DDPM with Context Conditioning
Supports linear/MLP/only_lin/no context or **TrueCCIP** conditioning.
```bash
torchrun --nproc_per_node <gpus> train_ddpm.py \
  --h5_data_path data/dataset_comp_image_spectra.h5 \
  --train_indices_path splits/split_train_indices.txt \
  --vae_checkpoint_path checkpoints/vae/vae_best.pth \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --context_embedder_type TrueCCIP \
  --ccip_checkpoint_path checkpoints/true_ccip/true_ccip_best.pth \
  --checkpoint_dir checkpoints/ddpm_true_ccip \
  --batch_size 4 --num_epochs 100 --timesteps 1000
```
Notes:
- `--comp_vector_dim` is inferred from the tokenizer if omitted.
- `--latent_scale_factor` allows scaling VAE latents (defaults to none here; training warns if unset).
- Scheduler options: `--lr_scheduler_type {cosine,steplr}` with associated parameters.

## Image Generation
`generate_ddpm.py` mirrors the training interfaces and supports multiple composition sources (JSON, custom strings, indices, or full dataset). Examples:

**True CCIP-conditioned generation**
```bash
python generate_ddpm.py \
  --ddpm_checkpoint checkpoints/ddpm_true_ccip/ddpm_epoch_12.pth \
  --vae_checkpoint_path checkpoints/vae/vae_best.pth \
  --ccip_checkpoint_path checkpoints/true_ccip/true_ccip_best.pth \
  --context_embedder_type TrueCCIP \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --h5_data_path data/dataset_comp_image_spectra.h5 \
  --indices_file splits/split_test_indices.txt \
  --num_samples 100 --batch_size 4 \
  --output_dir generated_samples --save_to_hdf5
```

**Custom compositions (linear/MLP embedding)**
```bash
python generate_ddpm.py \
  --ddpm_checkpoint checkpoints/ddpm_linear/ddpm_best.pth \
  --vae_checkpoint_path checkpoints/vae/vae_best.pth \
  --context_embedder_type mlp \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --custom_compositions "Fe:0.5,O:0.5;Al:0.3,O:0.7" \
  --batch_size 2 --output_dir generated_samples/custom
```

Useful flags:
- Composition inputs: `--compositions_json`, `--custom_compositions`, `--specific_indices`, `--indices_file`.
- Performance: `--use_half_precision`, `--compile_models`, `--fast_sampling` (caps timesteps to 50).
- Outputs: `--save_to_hdf5`, `--save_both_formats`, `--hdf5_path`, `--timing_report`.

## Additional Resources
- True CCIP training guide: `docs/TRUE_CCIP_TRAINING_GUIDE.md`
- DDPM generation details: `docs/DDPM_TRUE_CCIP_GENERATION_GUIDE.md`
- Cluster training tips: `docs/CLUSTER_TRAINING_GUIDE.md`
- Tokenization analysis: `docs/README_tokenizer_analysis.md`

## License
MIT License (see `LICENSE`).
