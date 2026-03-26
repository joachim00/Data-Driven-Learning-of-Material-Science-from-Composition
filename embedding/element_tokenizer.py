import torch
from torch import nn
from torch.nn import functional as F
import json
import numpy as np

class CompositionTokenizer:
    def __init__(self, embedding_path, normalize=True):
        """
        Initialize the tokenizer with a path to an embedding JSON file.

        Args:
            embedding_path (str): Path to the megnet16-embedding.json file
            normalize (bool): Whether to normalize vectors to unit norm
        """
        with open(embedding_path, "r") as f:
            self.element_embeddings = json.load(f)
        self.embedding_dim = len(next(iter(self.element_embeddings.values())))
        self.normalize = normalize

    def tokenize(self, composition_dict):
        """
        Convert a composition dictionary to a vector using elemental embeddings.

        Args:
            composition_dict (dict): e.g., {'C': 0.6, 'O': 0.4}

        Returns:
            torch.Tensor: shape (embedding_dim,)
        """
        composition_vector = np.zeros(self.embedding_dim, dtype=np.float32)

        for element, fraction in composition_dict.items():
            if element not in self.element_embeddings:
                raise ValueError(f"Element {element} not found in embedding.")
            element_vec = np.array(self.element_embeddings[element], dtype=np.float32)
            composition_vector += fraction * element_vec

        if self.normalize:
            norm = np.linalg.norm(composition_vector)
            if norm > 0:
                composition_vector /= norm

        return torch.tensor(composition_vector)

    def tokenize_batch(self, composition_dicts):
        """
        Convert a list of composition dictionaries to a batch of vectors.

        Args:
            composition_dicts (list[dict]): List of composition dicts, e.g., [{'C': 0.5, 'O': 0.5}, {'Si': 0.7, 'O': 0.3}]

        Returns:
            torch.Tensor: shape (batch_size, embedding_dim)
        """
        vectors = [self.tokenize(cd).unsqueeze(0) for cd in composition_dicts]
        return torch.cat(vectors, dim=0)


if __name__ == "__main__":
    # Example usage
    tokenizer = CompositionTokenizer("embedding/tokenizers/megnet16-embedding.json", normalize=False)
    composition_dict = {'C': 0.5, 'O': 0.5}
    vector = tokenizer.tokenize(composition_dict)
    print(vector)

    # Batch example
    batch_dicts = [{'C': 0.5, 'O': 0.5}, {'Si': 0.7, 'O': 0.3}]
    batch_vectors = tokenizer.tokenize_batch(batch_dicts)
    print(batch_vectors)