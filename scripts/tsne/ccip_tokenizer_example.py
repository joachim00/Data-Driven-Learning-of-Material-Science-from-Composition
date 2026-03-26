import torch
import numpy as np
from typing import Dict, List, Tuple

class CCIPTokenizer:
    """
    CLIP-style tokenizer for compositions.
    Treats each element as a separate token, like words in CLIP.
    """
    
    def __init__(self, vocab_size=95, max_length=20, fraction_bins=10):
        """
        Args:
            vocab_size: Number of elements (95 for periodic table)
            max_length: Maximum sequence length
            fraction_bins: Number of bins for discretizing fractions
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.fraction_bins = fraction_bins
        
        # Create element to ID mapping (simplified - would use actual periodic table)
        elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                   'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                   'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                   # ... extend to 95 elements
                   ]
        self.element_to_id = {elem: i for i, elem in enumerate(elements)}
        self.id_to_element = {i: elem for elem, i in self.element_to_id.items()}
        
        # Special tokens
        self.PAD_TOKEN = 0
        self.START_TOKEN = 1
        self.END_TOKEN = 2
    
    def fraction_to_token(self, fraction: float) -> int:
        """Convert continuous fraction to discrete token."""
        # Discretize fractions into bins (0.1, 0.2, 0.3, etc.)
        bin_idx = min(int(fraction * self.fraction_bins), self.fraction_bins - 1)
        return bin_idx
    
    def tokenize_sequence_style(self, composition: Dict[str, float]) -> torch.LongTensor:
        """
        Convert composition to sequence of element tokens.
        Each element appears as separate token.
        """
        tokens = [self.START_TOKEN]
        
        # Sort elements by fraction (descending) for consistency
        sorted_elements = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        
        for element, fraction in sorted_elements:
            if element in self.element_to_id:
                # Add element token
                element_id = self.element_to_id[element]
                tokens.append(element_id)
                
                # Optionally add fraction token
                # fraction_token = self.fraction_to_token(fraction)
                # tokens.append(self.vocab_size + fraction_token)
        
        tokens.append(self.END_TOKEN)
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.PAD_TOKEN)
        
        return torch.LongTensor(tokens[:self.max_length])
    
    def tokenize_repeated_style(self, composition: Dict[str, float]) -> torch.LongTensor:
        """
        Convert composition to sequence where elements repeat based on fraction.
        More intuitive - higher fraction = more repetitions.
        """
        tokens = [self.START_TOKEN]
        total_tokens = self.max_length - 2  # Reserve space for START/END
        
        # Calculate number of tokens per element
        element_counts = {}
        for element, fraction in composition.items():
            if element in self.element_to_id:
                count = max(1, int(fraction * total_tokens))
                element_counts[element] = count
        
        # Normalize to fit in max_length
        total_count = sum(element_counts.values())
        if total_count > total_tokens:
            scale = total_tokens / total_count
            element_counts = {elem: max(1, int(count * scale)) 
                            for elem, count in element_counts.items()}
        
        # Create token sequence
        for element, count in element_counts.items():
            element_id = self.element_to_id[element]
            tokens.extend([element_id] * count)
        
        tokens.append(self.END_TOKEN)
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.PAD_TOKEN)
        
        return torch.LongTensor(tokens[:self.max_length])
    
    def tokenize_with_fractions(self, composition: Dict[str, float]) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Hybrid approach: element tokens + continuous fractions.
        Returns both token sequence and fraction values.
        """
        tokens = [self.START_TOKEN]
        fractions = [0.0]  # Start token has no fraction
        
        sorted_elements = sorted(composition.items(), key=lambda x: x[1], reverse=True)
        
        for element, fraction in sorted_elements:
            if element in self.element_to_id:
                element_id = self.element_to_id[element]
                tokens.append(element_id)
                fractions.append(fraction)
        
        tokens.append(self.END_TOKEN)
        fractions.append(0.0)  # End token has no fraction
        
        # Pad both sequences
        while len(tokens) < self.max_length:
            tokens.append(self.PAD_TOKEN)
            fractions.append(0.0)
        
        return (torch.LongTensor(tokens[:self.max_length]), 
                torch.FloatTensor(fractions[:self.max_length]))


# Example usage
if __name__ == "__main__":
    tokenizer = CCIPTokenizer()
    composition = {"Al": 0.33, "Fe": 0.33, "Ni": 0.34}
    
    print("Original composition:", composition)
    
    # Method 1: Sequence style
    tokens_seq = tokenizer.tokenize_sequence_style(composition)
    print("Sequence tokens:", tokens_seq[:10])
    
    # Method 2: Repeated style  
    tokens_rep = tokenizer.tokenize_repeated_style(composition)
    print("Repeated tokens:", tokens_rep[:10])
    
    # Method 3: With fractions
    tokens_frac, fractions = tokenizer.tokenize_with_fractions(composition)
    print("Tokens with fractions:", tokens_frac[:10])
    print("Fraction values:", fractions[:10])
