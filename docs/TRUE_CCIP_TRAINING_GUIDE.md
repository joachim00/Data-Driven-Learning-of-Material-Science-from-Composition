# True CCIP Training (DDP)

Train the True CCIP composition encoder with contrastive learning on compositions and images using `train_true_ccip.py`.

## Prerequisites
- Dataset: `data/dataset_comp_image_spectra.h5`
- Splits: `splits/split_train_indices.txt`, `splits/split_test_indices.txt`
- Config: `configs/true_ccip_config.yaml` (vocab size, transformer depth/heads, projection heads, logging/output paths)

## Command
```bash
torchrun --nproc_per_node <gpus> train_true_ccip.py \
  --config configs/true_ccip_config.yaml \
  [--resume checkpoints/true_ccip/true_ccip_best.pth]
```

## What the script does
- Loads data via `TrueCCIPDataset` with max composition length from config.
- Builds composition transformer + image CNN + projection heads (see `model.*` in the config).
- Uses CLIP-style loss with optional learnable temperature (`training.learnable_temperature`).
- Runs DDP; saves checkpoints to `logging.output_dir` (best + per-epoch).

## Key config fields
- `data.data_path`, `data.train_indices`, `data.test_indices`
- `model.composition.{vocab_size,max_length,embedding_dim,n_layers,n_heads,use_positional_encoding}`
- `model.image.{encoder_type,image_size,hidden_dim}`
- `model.projection.{output_dim,use_projection_head,num_layers}`
- `training.{batch_size,learning_rate,num_epochs,temperature,grad_clip_norm,use_amp,warmup_epochs,min_lr}`
- `logging.{output_dir,save_every,eval_every,log_every_n_batches}`

## Tips
- Start with smaller `batch_size` if using large GPUs; AMP is enabled by config.
- Ensure `output_dir` exists or is creatable on all ranks.
- Resume with `--resume` to continue training and scheduler state.
