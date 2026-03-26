import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sd.true_ccip import CCIPCompositionTokenizer
from typing import Dict, Tuple, Optional


class TrueCCIPDataset(Dataset):
    """
    Dataset for True CLIP-style CCIP training.
    Returns tokenized compositions instead of weighted vectors.
    """
    
    def __init__(self, h5_path: str, transform=None, max_length: int = 20, start: int = 0, end: Optional[int] = None):
        """
        Args:
            h5_path: Path to the HDF5 file
            transform: Optional transform for images
            max_length: Maximum sequence length for compositions
            start: Start index of dataset slice
            end: End index of dataset slice
        """
        self.h5_path = h5_path
        self.transform = transform
        self.start = start
        self.max_length = max_length
        
        # Initialize true CCIP tokenizer
        self.tokenizer = CCIPCompositionTokenizer(max_length=max_length)
        
        with h5py.File(h5_path, 'r') as f:
            self.end = end if end is not None else f['images'].shape[0]
        self.N = self.end - self.start
    
    def __len__(self):
        return self.N
    
    def get_composition_dict(self, idx: int) -> Dict[str, float]:
        """Get the composition dictionary for a specific index."""
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            comp_vector = f['atfrac'][:, idx]
            comp_dict = {
                str(e, 'utf-8').split('.PM')[0]: v 
                for v, e in zip(comp_vector, f['atfrac_keys']) 
                if v > 0
            }
        return comp_dict
    
    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor, torch.Tensor]:
        """
        Get a single item from the dataset.
        
        Returns:
            tokens: [max_length] element token IDs
            fractions: [max_length] atomic fractions
            attention_mask: [max_length] attention mask
            image: [C, H, W] transformed image
        """
        idx = self.start + idx
        
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            # Get composition
            comp_vector = f['atfrac'][:, idx]
            comp_dict = {
                str(e, 'utf-8').split('.PM')[0]: v 
                for v, e in zip(comp_vector, f['atfrac_keys']) 
                if v > 0
            }
            
            # Get image
            image = f['images'][idx]
        
        # Tokenize composition
        tokens, fractions, attention_mask = self.tokenizer.tokenize(comp_dict)
        
        # Transform image
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        
        return tokens, fractions, attention_mask, image


def true_ccip_collate_fn(batch):
    """
    Custom collate function for True CCIP dataset.
    
    Args:
        batch: List of (tokens, fractions, attention_mask, image) tuples
        
    Returns:
        batch_tokens: [batch_size, max_length]
        batch_fractions: [batch_size, max_length]
        batch_attention_masks: [batch_size, max_length]
        batch_images: [batch_size, C, H, W]
    """
    tokens, fractions, attention_masks, images = zip(*batch)
    
    return (
        torch.stack(tokens),
        torch.stack(fractions),
        torch.stack(attention_masks),
        torch.stack(images)
    )


# Test the dataset
if __name__ == "__main__":
    from torchvision import transforms
    
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((64, 64), antialias=True),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    print("Testing True CCIP Dataset")
    print("=" * 40)
    
    try:
        # Test dataset
        dataset = TrueCCIPDataset(
            h5_path='data/dataset_comp_image_spectra.h5',
            transform=transform,
            max_length=20,
            start=0,
            end=10  # Test with first 10 samples
        )
        
        print(f"Dataset size: {len(dataset)}")
        print(f"Tokenizer vocab size: {dataset.tokenizer.vocab_size}")
        
        # Test a single sample
        tokens, fractions, mask, image = dataset[0]
        print(f"\nSample 0:")
        print(f"  Tokens shape: {tokens.shape}")
        print(f"  Fractions shape: {fractions.shape}")
        print(f"  Attention mask shape: {mask.shape}")
        print(f"  Image shape: {image.shape}")
        print(f"  Real tokens: {mask.sum().item()}")
        
        # Decode composition
        comp_dict = dataset.tokenizer.decode(tokens, fractions, mask)
        print(f"  Composition: {comp_dict}")
        
        # Test dataloader with collate function
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False, 
            collate_fn=true_ccip_collate_fn
        )
        
        batch_tokens, batch_fractions, batch_masks, batch_images = next(iter(dataloader))
        print(f"\nBatch test:")
        print(f"  Batch tokens shape: {batch_tokens.shape}")
        print(f"  Batch fractions shape: {batch_fractions.shape}")
        print(f"  Batch masks shape: {batch_masks.shape}")
        print(f"  Batch images shape: {batch_images.shape}")
        
        print("\nTrue CCIP Dataset test successful!")
        
    except FileNotFoundError:
        print("Dataset file not found. Expected: data/dataset_comp_image_spectra.h5")
        print("   This is normal if running outside the project directory.")
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("   This is expected if the HDF5 file structure differs.")
