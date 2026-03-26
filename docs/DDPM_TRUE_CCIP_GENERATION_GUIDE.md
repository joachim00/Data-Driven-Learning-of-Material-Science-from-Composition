# Image Generation (DDPM + Context)

Use `generate_ddpm.py` to sample images conditioned on compositions. Supports TrueCCIP, linear/MLP, or no context.

## Inputs
- DDPM checkpoint trained with the target `context_embed_dim`.
- VAE decoder checkpoint (e.g., `checkpoints/vae/vae_best.pth`).
- Context embedder:
  - TrueCCIP: `--context_embedder_type TrueCCIP --ccip_checkpoint_path checkpoints/true_ccip/true_ccip_best.pth`
  - Linear/MLP: `--context_embedder_type {linear|mlp|only_lin} --tokenizer_path embedding/tokenizers/megnet16-embedding.json`
- Compositions from JSON, custom strings, indices file, or dataset (`data/dataset_comp_image_spectra.h5` + `splits`).

## Common commands
TrueCCIP on dataset indices:
```bash
python generate_ddpm.py \
  --ddpm_checkpoint checkpoints/ddpm_true_ccip/ddpm_epoch_12.pth \
  --vae_checkpoint_path checkpoints/vae/vae_best.pth \
  --context_embedder_type TrueCCIP \
  --ccip_checkpoint_path checkpoints/true_ccip/true_ccip_best.pth \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --h5_data_path data/dataset_comp_image_spectra.h5 \
  --indices_file splits/split_test_indices.txt \
  --num_samples 50 --batch_size 4 \
  --output_dir generated_samples
```

Custom compositions with MLP embedder:
```bash
python generate_ddpm.py \
  --ddpm_checkpoint checkpoints/ddpm_linear/ddpm_best.pth \
  --vae_checkpoint_path checkpoints/vae/vae_best.pth \
  --context_embedder_type mlp \
  --tokenizer_path embedding/tokenizers/megnet16-embedding.json \
  --custom_compositions "Fe:0.5,O:0.5;Al:0.3,O:0.7" \
  --batch_size 2 --output_dir generated_samples/custom
```

## Important flags
- Inputs: `--compositions_json`, `--custom_compositions`, `--specific_indices`, `--indices_file`, `--num_samples`.
- Performance: `--use_half_precision`, `--compile_models`, `--fast_sampling` (caps timesteps), `--batch_size`, `--num_workers`, `--pin_memory`.
- Quality/speed: `--timesteps`, `--image_size`, `--latent_scale_factor`.
- Outputs: `--save_to_hdf5`, `--save_both_formats`, `--hdf5_path`, `--output_dir`, `--timing_report`.

## Outputs
- PNGs (and optional HDF5) in `output_dir`.
- Log file and optional timing report JSON with per-stage timings.

## Tips
- Ensure tokenizer embedding dim matches DDPM `context_embed_dim` for linear/MLP modes.
- For TrueCCIP, keep inputs in float32; FP16 is handled where safe.
- Start with small batches if using many timesteps or limited GPU memory.
