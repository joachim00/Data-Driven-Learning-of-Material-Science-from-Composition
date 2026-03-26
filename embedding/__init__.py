"""
embedding — Element tokenization and composition encoding.

Provides tools for converting chemical compositions to numerical representations
using various pre-trained element embeddings.

Modules:
    - element_tokenizer: CompositionTokenizer for encoding compositions

Available embeddings (in tokenizers/):
    - elem: Standard element embeddings
    - megnet16: MEGNet element features
    - matscholar: MatScholar embeddings
    - word2vec: Word2Vec-trained element embeddings
    - cgcnn: CGCNN element features
    - onehot: One-hot encoded elements
"""

from .element_tokenizer import CompositionTokenizer

__all__ = ['CompositionTokenizer']
