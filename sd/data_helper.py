import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class HDF5CompositionImageDataset(Dataset):
    def __init__(self, h5_path, transform=None, comp_format='dict', tokenizer=None, start=0, end=None):
        """
        Args:
            h5_path (str): Path to the HDF5 file.
            transform (callable, optional): Optional transform to be applied on a sample.
            comp_format (str): Format of the composition vector ('dict' or 'tensor').
            tokenizer (object, optional): Tokenizer for converting composition dicts to vectors.
            start (int, optional): Start index of dataset slice.
            end (int, optional): End index of dataset slice.
        """
        assert comp_format in ['dict', 'tensor'], "comp_format must be either 'dict' or 'tensor'"
        self.h5_path = h5_path
        self.transform = transform
        self.start = start
        self.comp_format = comp_format
        if tokenizer is not None:
            self.tokenizer = tokenizer
        with h5py.File(h5_path, 'r') as f:
            self.end = end if end is not None else f['images'].shape[0]
        self.N = self.end - self.start

    def __len__(self):
        return self.N
    
    def get_composition_dict(self, idx):
        """
        Get the composition dictionary for a specific index.
        """
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            comp_vector = f['atfrac'][:, idx]
            comp_dict = {str(e, 'utf-8').split('.PM')[0]: float(v) for v, e in zip(comp_vector, f['atfrac_keys']) if v > 0}
        return comp_dict

    def __getitem__(self, idx):
        idx = self.start + idx
        with h5py.File(self.h5_path, 'r', swmr=True) as f:
            if self.comp_format == 'dict':
                comp_vector = f['atfrac'][:, idx]
                comp_dict = {str(e, 'utf-8').split('.PM')[0]: float(v) for v, e in zip(comp_vector, f['atfrac_keys']) if v > 0}
                comp_vector = comp_dict
                if hasattr(self, 'tokenizer'):
                    comp_vector = self.tokenizer.tokenize(comp_vector)
            elif self.comp_format == 'tensor':
                comp_vector = torch.tensor(f['atfrac'][:, idx], dtype=torch.float32)

            image = f['images'][idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

        return comp_vector, image
    
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts (H, W, C) ndarray to (C, H, W) tensor and scales to [0, 1]
        transforms.Normalize([0.5]*3, [0.5]*3)  # Scales [0, 1] → [-1, 1]
    ])
    index = 100
    dataset = HDF5CompositionImageDataset('data/dataset_comp_image_spectra.h5', transform=transform)

    comp_vector, image = dataset.__getitem__(index)
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    print(f"Image array shape: {image.shape}")
    print(type(image), image.shape)
    print(np.min(image))     # np. 0.0
    print(np.max(image))
    print(comp_vector)
    plt.imshow(image)
    plt.show()

def indice_import(path):
    indices = []
    with open(path, "r") as f:
        for line in f:
            idx = int(line.strip())
            indices.append(idx)
    return indices