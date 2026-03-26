"""
Tokenizer Embedding t-SNE Visualization with Spectral Colors and Ternary Systems

This script:
1. Loads composition tokenizer embeddings directly (no CCIP model needed)
2. Generates t-SNE visualization of token-based composition representations
3. Colors points by material spectra and marks ternary composition regions
4. Compares different tokenizer approaches or train/test data splits
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py
from tqdm import tqdm
import argparse
import os
from collections import Counter

from sd.data_helper import HDF5CompositionImageDataset
from embedding.element_tokenizer import CompositionTokenizer


def extract_dominant_colors_from_spectra(spectra, energy_eV=None, color_weights=(1.5, 1.2, 0.5)):
    """
    Extract RGB colors from material spectra using energy-based mapping
    
    Args:
        spectra: Array of shape (N, spectral_bands) - spectral intensities
        energy_eV: Array of shape (spectral_bands,) - energy values in eV
        color_weights: Tuple of (red_weight, green_weight, blue_weight) for balancing
        
    Returns:
        colors: Array of shape (N, 3) with RGB values [0, 1]
    """
    if spectra.ndim == 1:
        spectra = spectra.reshape(1, -1)
    
    # Normalize spectra to [0, 1] for each sample
    spectra_norm = (spectra - spectra.min(axis=1, keepdims=True)) / (
        spectra.max(axis=1, keepdims=True) - spectra.min(axis=1, keepdims=True) + 1e-8
    )
    
    n_samples, n_spectral_bands = spectra_norm.shape
    
    if energy_eV is not None and len(energy_eV) == n_spectral_bands:
        # Energy-based RGB mapping (physically meaningful)
        energy_min, energy_max = energy_eV.min(), energy_eV.max()
        energy_range = energy_max - energy_min
        
        # Overlapping ranges for balanced color mixing
        red_range = (energy_min, energy_min + energy_range * 0.5)
        green_range = (energy_min + energy_range * 0.25, energy_min + energy_range * 0.85)
        blue_range = (energy_min + energy_range * 0.6, energy_max)
        
        print(f"Energy-based RGB mapping (balanced):")
        print(f"  Red:   {red_range[0]:.2f} - {red_range[1]:.2f} eV (covers {((red_range[1]-red_range[0])/energy_range)*100:.1f}% of range)")
        print(f"  Green: {green_range[0]:.2f} - {green_range[1]:.2f} eV (covers {((green_range[1]-green_range[0])/energy_range)*100:.1f}% of range)")
        print(f"  Blue:  {blue_range[0]:.2f} - {blue_range[1]:.2f} eV (covers {((blue_range[1]-blue_range[0])/energy_range)*100:.1f}% of range)")
        
        # Find indices for each energy range
        red_mask = (energy_eV >= red_range[0]) & (energy_eV < red_range[1])
        green_mask = (energy_eV >= green_range[0]) & (energy_eV < green_range[1])
        blue_mask = (energy_eV >= blue_range[0]) & (energy_eV <= blue_range[1])
        
        # Calculate average intensity in each energy range
        red_intensity = np.zeros(n_samples)
        green_intensity = np.zeros(n_samples)
        blue_intensity = np.zeros(n_samples)
        
        if np.any(red_mask):
            red_intensity = spectra_norm[:, red_mask].mean(axis=1)
        if np.any(green_mask):
            green_intensity = spectra_norm[:, green_mask].mean(axis=1)
        if np.any(blue_mask):
            blue_intensity = spectra_norm[:, blue_mask].mean(axis=1)
        
        # Apply weights to balance the color channels
        red_weight, green_weight, blue_weight = color_weights
        
        red_intensity = red_intensity * red_weight
        green_intensity = green_intensity * green_weight
        blue_intensity = blue_intensity * blue_weight
        
        rgb_colors = np.column_stack([red_intensity, green_intensity, blue_intensity])
        
        # Normalize each sample's RGB to prevent oversaturation
        max_vals = rgb_colors.max(axis=1, keepdims=True)
        rgb_colors = np.where(max_vals > 1.0, rgb_colors / max_vals, rgb_colors)
        
        print(f"Successful energy-based RGB mapping with {np.sum(red_mask)} red, {np.sum(green_mask)} green, {np.sum(blue_mask)} blue bands")
        print(f"Applied color balancing: R×{red_weight}, G×{green_weight}, B×{blue_weight}")
            
    else:
        # Band-based mapping (when no energy info available)
        print("No energy information provided, using band-based RGB mapping")
        
        if n_spectral_bands >= 10:
            # Use low, mid, high spectral regions
            low_region = spectra_norm[:, :n_spectral_bands//3].mean(axis=1)
            mid_region = spectra_norm[:, n_spectral_bands//3:2*n_spectral_bands//3].mean(axis=1)
            high_region = spectra_norm[:, 2*n_spectral_bands//3:].mean(axis=1)
            
            rgb_colors = np.column_stack([low_region, mid_region, high_region])
        elif n_spectral_bands >= 3:
            # Use evenly distributed bands
            band_indices = np.linspace(0, n_spectral_bands-1, 3, dtype=int)
            rgb_colors = spectra_norm[:, band_indices]
        else:
            # Replicate available bands
            rgb_colors = np.repeat(spectra_norm, 3, axis=1)[:, :3]
    
    # Ensure values are in [0, 1]
    rgb_colors = np.clip(rgb_colors, 0, 1)
    
    return rgb_colors


def identify_ternary_regions(compositions):
    """
    Identify ternary composition regions and assign colors/markers
    
    Args:
        compositions: List of composition dictionaries
        
    Returns:
        ternary_labels: List of ternary system labels
        ternary_colors: Dict mapping ternary systems to colors
        ternary_markers: Dict mapping ternary systems to markers
    """
    ternary_systems = []
    
    for comp in compositions:
        # Get top 3 elements by atomic fraction
        sorted_elements = sorted(comp.items(), key=lambda x: x[1], reverse=True)
        top_3_elements = sorted([elem for elem, frac in sorted_elements[:3]])
        
        # Create ternary system label
        if len(top_3_elements) >= 3:
            ternary_label = "-".join(top_3_elements)
        elif len(top_3_elements) == 2:
            ternary_label = "-".join(top_3_elements) + "-X"
        else:
            ternary_label = "Single-element"
        
        ternary_systems.append(ternary_label)
    
    # Get unique ternary systems
    unique_systems = list(set(ternary_systems))
    print(f"\nFound {len(unique_systems)} unique ternary systems:")
    
    # Count compositions per system
    system_counts = Counter(ternary_systems)
    for system, count in system_counts.most_common(10):
        print(f"  {system}: {count} compositions")
    
    # Assign colors and markers to ternary systems
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_systems), 20)))
    if len(unique_systems) > 20:
        # Use additional colormap for more systems
        extra_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_systems) - 20))
        colors = np.vstack([colors, extra_colors])
    
    # Assign markers
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'h', 'p', '*', '+', 'x', '8', 'H', 'd', '|', '_']
    marker_cycle = markers * ((len(unique_systems) // len(markers)) + 1)
    
    ternary_colors = {system: colors[i] for i, system in enumerate(unique_systems)}
    ternary_markers = {system: marker_cycle[i] for i, system in enumerate(unique_systems)}
    
    return ternary_systems, ternary_colors, ternary_markers


def load_spectra_colors(data_path, indices, color_weights=(1.5, 1.2, 0.5)):
    """
    Load spectra and energy data, then extract RGB colors
    
    Args:
        data_path: Path to HDF5 file
        indices: List of sample indices
        color_weights: Tuple of (red_weight, green_weight, blue_weight)
        
    Returns:
        colors: RGB colors from spectra (N, 3)
        energy_info: Tuple of (energy_min, energy_max) or None
    """
    print("Loading spectra and energy data...")
    
    energy_info = None
    
    with h5py.File(data_path, 'r') as f:
        # Check available keys
        available_keys = list(f.keys())
        print(f"Available datasets: {available_keys}")
        
        if 'spectra' in f.keys():
            # Load spectra for specific indices
            all_spectra = f['spectra'][:]
            spectra = all_spectra[indices]
            
            print(f"Loaded spectra shape: {spectra.shape}")
            
            # Load energy data if available
            energy_eV = None
            if 'energy_eV' in f.keys():
                energy_eV = f['energy_eV'][:]
                energy_info = (energy_eV.min(), energy_eV.max())
                print(f"Loaded energy data shape: {energy_eV.shape}")
                print(f"Energy range: {energy_eV.min():.2f} - {energy_eV.max():.2f} eV")
            
            # Extract RGB colors from spectra using energy information
            colors = extract_dominant_colors_from_spectra(spectra, energy_eV, color_weights)
            
            print(f"Generated RGB colors shape: {colors.shape}")
            print(f"Color value ranges - R: {colors[:, 0].min():.3f}-{colors[:, 0].max():.3f}, "
                  f"G: {colors[:, 1].min():.3f}-{colors[:, 1].max():.3f}, "
                  f"B: {colors[:, 2].min():.3f}-{colors[:, 2].max():.3f}")
            
        else:
            print("Error: 'spectra' dataset not found in HDF5 file")
            print(f"Available datasets: {available_keys}")
            print("Using random colors as fallback")
            colors = np.random.rand(len(indices), 3)
    
    return colors, energy_info


def compute_tokenizer_embeddings(compositions, tokenizer, max_length=512, pooling='mean'):
    """
    Compute embeddings using only the tokenizer (no CCIP model)
    
    Args:
        compositions: List of composition dictionaries
        tokenizer: CompositionTokenizer instance
        max_length: Maximum sequence length (not used for element tokenizer)
        pooling: Pooling strategy (not applicable for element tokenizer)
        
    Returns:
        embeddings: Tokenizer embeddings (N, embedding_dim)
    """
    print(f"Computing tokenizer embeddings using element-based composition vectors...")
    
    all_embeddings = []
    
    for comp in tqdm(compositions, desc="Processing compositions"):
        # Use the tokenizer's tokenize method to get composition vector
        comp_embedding = tokenizer.tokenize(comp)
        
        # Convert to numpy if it's a tensor
        if hasattr(comp_embedding, 'numpy'):
            comp_embedding = comp_embedding.numpy()
        
        all_embeddings.append(comp_embedding)
    
    embeddings = np.stack(all_embeddings)
    print(f"Generated tokenizer embeddings shape: {embeddings.shape}")
    
    return embeddings


def create_tokenizer_tsne_visualization(embeddings, colors, compositions, tokenizer_name, 
                                      save_path=None, perplexity=30, energy_info=None, dataset_type="", max_systems=15):
    """
    Create t-SNE visualization of tokenizer embeddings with ternary system annotations
    """
    print(f"Computing t-SNE for {embeddings.shape[0]} tokenizer embeddings...")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, embeddings.shape[0] - 1),
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    
    tsne_results = tsne.fit_transform(embeddings_scaled)
    
    # Identify ternary regions
    ternary_labels, ternary_colors, ternary_markers = identify_ternary_regions(compositions)
    
    # Create plot with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Spectral colors
    scatter1 = ax1.scatter(
        tsne_results[:, 0], 
        tsne_results[:, 1],
        c=colors,
        s=25,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.1
    )
    
    ax1.set_title(f'{tokenizer_name} Tokenizer - Spectral Colors\n({dataset_type} dataset)', 
                  fontsize=12, weight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ternary system regions
    unique_systems = list(set(ternary_labels))
    
    for i, system in enumerate(unique_systems[:max_systems]):  # Use max_systems parameter
        mask = np.array([label == system for label in ternary_labels])
        if np.sum(mask) > 0:
            ax2.scatter(
                tsne_results[mask, 0],
                tsne_results[mask, 1],
                c=[ternary_colors[system]],
                marker=ternary_markers[system],
                s=30,
                alpha=0.7,
                label=f'{system} ({np.sum(mask)})',
                edgecolors='black',
                linewidth=0.1
            )
    
    ax2.set_title(f'{tokenizer_name} Tokenizer - Ternary Systems\n({dataset_type} dataset)', 
                  fontsize=12, weight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.grid(True, alpha=0.3)
    
    # Add legend for ternary systems (outside plot)
    if len(unique_systems) <= max_systems:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax2.text(1.05, 1.0, f'Showing top {max_systems} of {len(unique_systems)} ternary systems', 
                transform=ax2.transAxes, fontsize=9, verticalalignment='top')
        ax2.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left', fontsize=7)
    
    # Add annotations
    if energy_info:
        energy_min, energy_max = energy_info
        fig.text(0.02, 0.02, 
                f'Left: Spectral colors from {energy_min:.2f}-{energy_max:.2f} eV\n'
                f'Right: Ternary composition systems\n'
                f'Tokenizer: {tokenizer_name} | Similar clustering indicates good composition representation',
                fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return tsne_results, ternary_labels


def create_combined_train_test_tokenizer_visualization(train_results, test_results, tokenizer_name, 
                                                    save_dir="tokenizer_tsne_results", energy_info=None, max_systems=15):
    """
    Create visualization combining both train and test data with different markers for tokenizer analysis
    """
    print(f"\n{'='*60}")
    print("CREATING COMBINED TRAIN/TEST TOKENIZER VISUALIZATION")
    print(f"{'='*60}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine embeddings and data
    all_embeddings = np.vstack([train_results['embeddings'], test_results['embeddings']])
    all_colors = np.vstack([train_results['colors'], test_results['colors']])
    all_compositions = train_results['compositions'] + test_results['compositions']
    
    # Create dataset labels
    n_train = len(train_results['compositions'])
    n_test = len(test_results['compositions'])
    dataset_labels = ['train'] * n_train + ['test'] * n_test
    
    print(f"Combined data: {n_train} train + {n_test} test = {len(all_compositions)} total samples")
    print(f"Combined embeddings shape: {all_embeddings.shape}")
    
    # Compute t-SNE for combined embeddings
    print("Computing combined tokenizer embeddings t-SNE...")
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(all_embeddings)
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(50, all_embeddings.shape[0] - 1),
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    
    tsne_results = tsne.fit_transform(embeddings_scaled)
    
    # Identify ternary regions
    ternary_labels, ternary_colors, ternary_markers = identify_ternary_regions(all_compositions)
    
    # Create comprehensive plot with 2x3 layout
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: All data with spectral colors, train/test markers
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Train data (circles)
    train_mask = np.array(dataset_labels) == 'train'
    ax1.scatter(tsne_results[train_mask, 0], tsne_results[train_mask, 1],
               c=all_colors[train_mask], marker='o', s=25, alpha=0.7,
               edgecolors='black', linewidth=0.2, label=f'Train ({n_train})')
    
    # Test data (squares)
    test_mask = np.array(dataset_labels) == 'test'
    ax1.scatter(tsne_results[test_mask, 0], tsne_results[test_mask, 1],
               c=all_colors[test_mask], marker='s', s=25, alpha=0.7,
               edgecolors='red', linewidth=0.2, label=f'Test ({n_test})')
    
    ax1.set_title(f'{tokenizer_name} - Combined Train/Test (Spectral Colors)', fontsize=12, weight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train data only - spectral colors
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(tsne_results[train_mask, 0], tsne_results[train_mask, 1],
               c=all_colors[train_mask], s=30, alpha=0.7,
               edgecolors='black', linewidth=0.1)
    ax2.set_title(f'{tokenizer_name} - Train Data Only (Spectral)', fontsize=12, weight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test data only - spectral colors
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(tsne_results[test_mask, 0], tsne_results[test_mask, 1],
               c=all_colors[test_mask], s=30, alpha=0.7,
               edgecolors='black', linewidth=0.1)
    ax3.set_title(f'{tokenizer_name} - Test Data Only (Spectral)', fontsize=12, weight='bold')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Combined ternary systems with train/test markers
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Get unique ternary systems (limit for visibility)
    unique_systems = list(set(ternary_labels))[:max_systems]
    
    for i, system in enumerate(unique_systems):
        system_mask = np.array([label == system for label in ternary_labels])
        
        # Train samples of this system
        train_system_mask = system_mask & train_mask
        if np.sum(train_system_mask) > 0:
            ax4.scatter(tsne_results[train_system_mask, 0], tsne_results[train_system_mask, 1],
                       c=[ternary_colors[system]], marker='o', s=25, alpha=0.7,
                       edgecolors='black', linewidth=0.1, 
                       label=f'{system} Train ({np.sum(train_system_mask)})')
        
        # Test samples of this system
        test_system_mask = system_mask & test_mask
        if np.sum(test_system_mask) > 0:
            ax4.scatter(tsne_results[test_system_mask, 0], tsne_results[test_system_mask, 1],
                       c=[ternary_colors[system]], marker='s', s=25, alpha=0.7,
                       edgecolors='red', linewidth=0.1,
                       label=f'{system} Test ({np.sum(test_system_mask)})')
    
    ax4.set_title(f'{tokenizer_name} - Ternary Systems (Train=○, Test=□)', fontsize=12, weight='bold')
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.grid(True, alpha=0.3)
    
    # Add legend outside plot
    total_systems = len(set(ternary_labels))
    if len(unique_systems) <= 8:
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    else:
        ax4.text(1.05, 1.0, f'Showing top {len(unique_systems)} of {total_systems} systems\n(Legend abbreviated)', 
                transform=ax4.transAxes, fontsize=8, verticalalignment='top')
    
    # Plot 5: Train ternary systems only
    ax5 = fig.add_subplot(gs[1, 1])
    for i, system in enumerate(unique_systems):
        system_mask = np.array([label == system for label in ternary_labels])
        train_system_mask = system_mask & train_mask
        if np.sum(train_system_mask) > 0:
            ax5.scatter(tsne_results[train_system_mask, 0], tsne_results[train_system_mask, 1],
                       c=[ternary_colors[system]], marker=ternary_markers[system], s=30, alpha=0.7,
                       edgecolors='black', linewidth=0.1, 
                       label=f'{system} ({np.sum(train_system_mask)})')
    
    ax5.set_title(f'{tokenizer_name} - Train Ternary Systems', fontsize=12, weight='bold')
    ax5.set_xlabel('t-SNE Dimension 1')
    ax5.set_ylabel('t-SNE Dimension 2')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Test ternary systems only
    ax6 = fig.add_subplot(gs[1, 2])
    for i, system in enumerate(unique_systems):
        system_mask = np.array([label == system for label in ternary_labels])
        test_system_mask = system_mask & test_mask
        if np.sum(test_system_mask) > 0:
            ax6.scatter(tsne_results[test_system_mask, 0], tsne_results[test_system_mask, 1],
                       c=[ternary_colors[system]], marker=ternary_markers[system], s=30, alpha=0.7,
                       edgecolors='black', linewidth=0.1,
                       label=f'{system} ({np.sum(test_system_mask)})')
    
    ax6.set_title(f'{tokenizer_name} - Test Ternary Systems', fontsize=12, weight='bold')
    ax6.set_xlabel('t-SNE Dimension 1')
    ax6.set_ylabel('t-SNE Dimension 2')
    ax6.grid(True, alpha=0.3)
    
    # Add global title and info
    fig.suptitle(f'{tokenizer_name} Tokenizer - Combined Train/Test Analysis\n'
                f'Train: {n_train} samples (○), Test: {n_test} samples (□), '
                f'Embedding dim: {all_embeddings.shape[1]}', fontsize=16, weight='bold')
    
    # Add annotations
    if energy_info:
        energy_min, energy_max = energy_info
        fig.text(0.02, 0.02, 
                f'Spectral colors: {energy_min:.2f}-{energy_max:.2f} eV | '
                f'Tokenizer: {tokenizer_name} | '
                f'Similar clustering indicates good composition representation',
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the plot
    save_path = f'{save_dir}/{tokenizer_name}_combined_train_test_tsne.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {save_path}")
    
    return tsne_results, ternary_labels, dataset_labels


def process_tokenizer_dataset(tokenizer_path, data_path, indices, color_weights, dataset_name):
    """
    Process a single dataset (train or test) for a tokenizer.
    Note: Ternary filtering should be done on indices before calling this function.
    
    Returns:
        dict with 'embeddings', 'colors', 'compositions'
    """
    tokenizer_name = os.path.basename(tokenizer_path).replace('-embedding.json', '')
    print(f"Processing {dataset_name} data for tokenizer: {tokenizer_name}")
    
    # Load tokenizer
    tokenizer = CompositionTokenizer(tokenizer_path)
    
    # Load data and extract compositions
    hdf5_dataset = HDF5CompositionImageDataset(data_path, comp_format='dict')
    
    compositions = []
    for idx in tqdm(indices, desc=f"Loading {dataset_name} compositions"):
        comp_dict = hdf5_dataset.get_composition_dict(idx)
        compositions.append(comp_dict)
    
    # Load spectral colors
    spectra_colors, energy_info = load_spectra_colors(data_path, indices, color_weights)
    
    # Compute tokenizer embeddings
    embeddings = compute_tokenizer_embeddings(compositions, tokenizer)
    
    return {
        'embeddings': embeddings,
        'colors': spectra_colors,
        'compositions': compositions,
        'energy_info': energy_info
    }


def compare_tokenizers(tokenizer_paths, data_path, test_indices, train_indices, color_weights, 
                      max_samples, save_dir, use_both_datasets=False, max_systems=15):
    """
    Compare multiple tokenizers side by side
    """
    print(f"\n{'='*60}")
    print("COMPARING MULTIPLE TOKENIZERS")
    print(f"{'='*60}")
    
    results = {}
    
    for tokenizer_path in tokenizer_paths:
        tokenizer_name = os.path.basename(tokenizer_path).replace('-embedding.json', '')
        print(f"\nProcessing tokenizer: {tokenizer_name}")
        
        try:
            if use_both_datasets:
                # Process both train and test data
                limited_test_indices = test_indices[:max_samples//2] if max_samples < len(test_indices) else test_indices
                limited_train_indices = train_indices[:max_samples//2] if max_samples < len(train_indices) else train_indices
                
                print(f"Processing both datasets: {len(limited_train_indices)} train + {len(limited_test_indices)} test")
                
                train_results = process_tokenizer_dataset(
                    tokenizer_path, data_path, limited_train_indices, color_weights, "TRAIN"
                )
                
                test_results = process_tokenizer_dataset(
                    tokenizer_path, data_path, limited_test_indices, color_weights, "TEST"
                )
                
                # Create combined visualization
                tsne_results, ternary_labels, dataset_labels = create_combined_train_test_tokenizer_visualization(
                    train_results, test_results, tokenizer_name,
                    save_dir=save_dir, energy_info=train_results['energy_info'], max_systems=max_systems
                )
                
                results[tokenizer_name] = {
                    'train_results': train_results,
                    'test_results': test_results,
                    'combined_tsne': tsne_results,
                    'ternary_labels': ternary_labels,
                    'dataset_labels': dataset_labels
                }
                
            else:
                # Process single dataset (existing functionality)
                # Load data using HDF5 dataset
                hdf5_dataset = HDF5CompositionImageDataset(data_path, comp_format='dict')
                
                # Use test indices by default for comparison
                limited_indices = test_indices[:max_samples] if max_samples < len(test_indices) else test_indices
                
                # Load compositions
                all_compositions = []
                for idx in limited_indices:
                    comp_dict = hdf5_dataset.get_composition_dict(idx)
                    all_compositions.append(comp_dict)
                
                # Load spectral colors
                spectra_colors, energy_info = load_spectra_colors(data_path, limited_indices, color_weights)
                
                # Load tokenizer
                tokenizer = CompositionTokenizer(tokenizer_path)
                
                # Compute embeddings
                embeddings = compute_tokenizer_embeddings(all_compositions, tokenizer)
                
                # Create visualization
                tsne_results, ternary_labels = create_tokenizer_tsne_visualization(
                    embeddings, spectra_colors, all_compositions, tokenizer_name,
                    save_path=f'{save_dir}/{tokenizer_name}_tokenizer_tsne.png',
                    energy_info=energy_info,
                    dataset_type="test comparison",
                    max_systems=max_systems
                )
                
                results[tokenizer_name] = {
                    'embeddings': embeddings,
                    'tsne': tsne_results,
                    'ternary': ternary_labels
                }
            
        except Exception as e:
            print(f"Error processing {tokenizer_name}: {e}")
            continue
    
    return results


def filter_compositions_by_element_count(compositions, min_elements=None, max_elements=None, exact_elements=None):
    """
    Filter compositions based on the number of elements present
    
    Args:
        compositions: List of composition dictionaries
        min_elements: Minimum number of elements (inclusive)
        max_elements: Maximum number of elements (inclusive)
        exact_elements: Exact number of elements (overrides min/max)
        
    Returns:
        filtered_compositions: Filtered list of compositions
        filter_mask: Boolean mask indicating which compositions were kept
    """
    filter_mask = []
    
    for comp in compositions:
        num_elements = len(comp)
        
        if exact_elements is not None:
            keep = (num_elements == exact_elements)
        else:
            keep = True
            if min_elements is not None:
                keep = keep and (num_elements >= min_elements)
            if max_elements is not None:
                keep = keep and (num_elements <= max_elements)
        
        filter_mask.append(keep)
    
    filter_mask = np.array(filter_mask)
    filtered_compositions = [comp for i, comp in enumerate(compositions) if filter_mask[i]]
    
    return filtered_compositions, filter_mask


def filter_indices_for_ternary_compositions(data_path, indices, exact_elements=3):
    """
    Filter indices to only include compositions with exactly the specified number of elements.
    This filtering happens before sampling to ensure we get the requested number of ternary samples.
    
    Args:
        data_path: Path to HDF5 dataset
        indices: List of indices to filter
        exact_elements: Exact number of elements required (default 3 for ternary)
        
    Returns:
        filtered_indices: List of indices that correspond to compositions with exact_elements
    """
    print(f"🔬 Pre-filtering {len(indices)} indices for compositions with exactly {exact_elements} elements...")
    
    # Create a temporary dataset to access compositions
    hdf5_dataset = HDF5CompositionImageDataset(data_path, comp_format='dict')
    
    filtered_indices = []
    for idx in tqdm(indices, desc=f"Filtering for {exact_elements}-element compositions"):
        try:
            comp_dict = hdf5_dataset.get_composition_dict(idx)
            if len(comp_dict) == exact_elements:
                filtered_indices.append(idx)
        except Exception as e:
            print(f"Warning: Could not process index {idx}: {e}")
            continue
    
    print(f"✅ Found {len(filtered_indices)} compositions with exactly {exact_elements} elements out of {len(indices)} total")
    
    if len(filtered_indices) == 0:
        print(f"❌ No compositions found with exactly {exact_elements} elements!")
        return None
        
    return filtered_indices


def main():
    parser = argparse.ArgumentParser(description='Tokenizer Embedding t-SNE Visualization with Spectral Colors and Ternary Systems')
    
    parser.add_argument('--tokenizer_path', type=str, 
                       default='embedding/tokenizers/elem-embedding.json',
                       help='Path to tokenizer file')
    parser.add_argument('--tokenizer_paths', nargs='+', type=str,
                       help='Paths to multiple tokenizers for comparison')
    parser.add_argument('--data_path', type=str,
                       default='data/dataset_comp_image_spectra.h5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--test_indices', type=str,
                       default='splits/split_test_indices.txt',
                       help='Path to test indices file')
    parser.add_argument('--train_indices', type=str,
                       default='splits/split_train_indices.txt',
                       help='Path to train indices file')
    parser.add_argument('--use_train_data', action='store_true',
                       help='Use training data instead of test data')
    parser.add_argument('--use_both_datasets', action='store_true',
                       help='Use both train and test data with different markers')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to process')
    parser.add_argument('--output_dir', type=str, default='tokenizer_tsne_results',
                       help='Output directory for results')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--pooling', type=str, default='mean', 
                       choices=['mean', 'max', 'sum', 'cls'],
                       help='Pooling strategy for token embeddings')
    parser.add_argument('--color_weights', nargs=3, type=float, default=[1.5, 1.2, 0.5],
                       help='RGB color weights for spectral mapping (red green blue). Default: 1.5 1.2 0.5')
    parser.add_argument('--max_systems', type=int, default=15,
                       help='Maximum number of ternary systems to show in visualization (default: 15)')
    parser.add_argument('--ternary_only', action='store_true',
                       help='Only analyze compositions that are true ternary systems (exactly 3 elements)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length for tokenization')
    
    args = parser.parse_args()
    
    print(f"🔤 Tokenizer Embedding Analysis")
    print(f"{'='*40}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load indices
    with open(args.test_indices, 'r') as f:
        test_indices = [int(line.strip()) for line in f.readlines()]
    
    # Load train indices if needed
    train_indices = None
    if args.use_train_data or args.use_both_datasets:
        with open(args.train_indices, 'r') as f:
            train_indices = [int(line.strip()) for line in f.readlines()]
    
    # Decide which datasets to process
    if args.use_both_datasets:
        # Pre-filter for ternary compositions if requested
        if args.ternary_only:
            print("🔬 Pre-filtering both datasets for ternary compositions...")
            train_indices = filter_indices_for_ternary_compositions(args.data_path, train_indices, exact_elements=3)
            test_indices = filter_indices_for_ternary_compositions(args.data_path, test_indices, exact_elements=3)
            
            if train_indices is None or test_indices is None:
                print("❌ Error: Not enough ternary compositions found in one or both datasets!")
                return
        
        # Limit samples if requested (after ternary filtering)
        if args.max_samples < len(test_indices):
            test_indices = test_indices[:args.max_samples//2]
        if args.max_samples < len(train_indices):
            train_indices = train_indices[:args.max_samples//2]
        
        print(f"Processing BOTH datasets: {len(train_indices)} train + {len(test_indices)} test samples")
        
        color_weights = tuple(args.color_weights)
        
        # Check if comparing multiple tokenizers
        if args.tokenizer_paths:
            print(f"Comparing {len(args.tokenizer_paths)} tokenizers with both datasets...")
            results = compare_tokenizers(
                args.tokenizer_paths, args.data_path, test_indices, train_indices,
                color_weights, args.max_samples, args.output_dir, use_both_datasets=True,
                max_systems=args.max_systems
            )
            
            print(f"\n✅ Combined tokenizer comparison completed!")
            print(f"📁 Results saved to: {args.output_dir}/")
            print("Generated files:")
            for tokenizer_path in args.tokenizer_paths:
                tokenizer_name = os.path.basename(tokenizer_path).replace('-embedding.json', '')
                print(f"  • {tokenizer_name}_combined_train_test_tsne.png")
                
        else:
            # Single tokenizer with both datasets
            tokenizer_name = os.path.basename(args.tokenizer_path).replace('-embedding.json', '')
            print(f"Analyzing tokenizer: {tokenizer_name} with both datasets")
            
            # Process both datasets
            train_results = process_tokenizer_dataset(
                args.tokenizer_path, args.data_path, train_indices, color_weights, "TRAIN"
            )
            
            test_results = process_tokenizer_dataset(
                args.tokenizer_path, args.data_path, test_indices, color_weights, "TEST"
            )
            
            # Check if filtering removed all compositions
            if train_results is None or test_results is None:
                print("❌ No ternary compositions found in one or both datasets!")
                return
            
            # Create combined visualization
            tsne_results, ternary_labels, dataset_labels = create_combined_train_test_tokenizer_visualization(
                train_results, test_results, tokenizer_name,
                save_dir=args.output_dir, energy_info=train_results['energy_info'],
                max_systems=args.max_systems
            )
            
            # Save combined results
            combined_results = {
                'train_embeddings': train_results['embeddings'],
                'test_embeddings': test_results['embeddings'],
                'combined_tsne_results': tsne_results,
                'ternary_labels': ternary_labels,
                'dataset_labels': dataset_labels,
                'train_compositions': train_results['compositions'],
                'test_compositions': test_results['compositions'],
                'train_spectral_colors': train_results['colors'],
                'test_spectral_colors': test_results['colors']
            }
            
            np.savez_compressed(
                f'{args.output_dir}/{tokenizer_name}_combined_train_test_results.npz',
                **{k: v for k, v in combined_results.items() if k not in ['train_compositions', 'test_compositions']}
            )
            
            print(f"\n✅ Combined tokenizer analysis completed!")
            print(f"📁 Results saved to: {args.output_dir}/")
            print("Generated files:")
            print(f"  • {tokenizer_name}_combined_train_test_tsne.png - Combined visualization")
            print(f"  • {tokenizer_name}_combined_train_test_results.npz - Numerical results")
            print("")
            print("🔍 Interpretation:")
            print(f"  • Train samples: {len(train_results['compositions'])} (circles)")
            print(f"  • Test samples: {len(test_results['compositions'])} (squares)")
            print(f"  • Embedding dimension: {train_results['embeddings'].shape[1]}")
            print(f"  • Clear separation indicates different chemical spaces")
            print(f"  • Element-based tokenizer uses weighted sum of element embeddings")
            
    else:
        # Process single dataset (either train or test)
        if args.use_train_data:
            indices = train_indices
            dataset_name = "train"
        else:
            indices = test_indices  
            dataset_name = "test"
        
        # Pre-filter for ternary compositions if requested
        if args.ternary_only:
            indices = filter_indices_for_ternary_compositions(args.data_path, indices, exact_elements=3)
            if indices is None:
                print(f"❌ Error: No ternary compositions found in {dataset_name} dataset!")
                return
        
        # Limit samples if requested (after ternary filtering)
        if args.max_samples < len(indices):
            indices = indices[:args.max_samples]
    
        print(f"Processing {len(indices)} {dataset_name} samples...")
        print(f"Pooling strategy: {args.pooling}")
        print(f"Color weights: R={args.color_weights[0]}, G={args.color_weights[1]}, B={args.color_weights[2]}")
        
        color_weights = tuple(args.color_weights)
        
        # Check if comparing multiple tokenizers
        if args.tokenizer_paths:
            print(f"Comparing {len(args.tokenizer_paths)} tokenizers...")
            results = compare_tokenizers(
                args.tokenizer_paths, args.data_path, test_indices, train_indices, 
                color_weights, args.max_samples, args.output_dir, use_both_datasets=False,
                max_systems=args.max_systems
            )
            
            print(f"\n✅ Tokenizer comparison completed!")
            print(f"📁 Results saved to: {args.output_dir}/")
            print("Generated files:")
            for tokenizer_path in args.tokenizer_paths:
                tokenizer_name = os.path.basename(tokenizer_path).replace('-embedding.json', '')
                print(f"  • {tokenizer_name}_tokenizer_tsne.png")
            
        else:
            # Single tokenizer analysis
            tokenizer_name = os.path.basename(args.tokenizer_path).replace('-embedding.json', '')
            print(f"Analyzing tokenizer: {tokenizer_name}")
            
            # Load tokenizer
            tokenizer = CompositionTokenizer(args.tokenizer_path)
            print(f"Tokenizer embedding dimension: {tokenizer.embedding_dim}")
            print(f"Number of elements: {len(tokenizer.element_embeddings)}")
            
            # Load data and extract compositions
            hdf5_dataset = HDF5CompositionImageDataset(args.data_path, comp_format='dict')
            
            compositions = []
            for idx in tqdm(indices, desc="Loading compositions"):
                # Use the dataset's get_composition_dict method to get raw composition dict
                comp_dict = hdf5_dataset.get_composition_dict(idx)
                compositions.append(comp_dict)
            
            # Load spectral colors
            spectra_colors, energy_info = load_spectra_colors(args.data_path, indices, color_weights)
            
            # Compute tokenizer embeddings
            embeddings = compute_tokenizer_embeddings(
                compositions, tokenizer, max_length=args.max_length, pooling=args.pooling
            )
            
            # Create visualization
            tsne_results, ternary_labels = create_tokenizer_tsne_visualization(
                embeddings, spectra_colors, compositions, tokenizer_name,
                save_path=f'{args.output_dir}/{tokenizer_name}_tokenizer_tsne.png',
                energy_info=energy_info,
                dataset_type=dataset_name
            )
            
            # Save results
            results = {
                'embeddings': embeddings,
                'tsne_results': tsne_results,
                'ternary_labels': ternary_labels,
                'compositions': compositions,
                'spectral_colors': spectra_colors,
                'indices': indices
            }
            
            np.savez_compressed(
                f'{args.output_dir}/{tokenizer_name}_tokenizer_results.npz',
                **{k: v for k, v in results.items() if k != 'compositions'}
            )
            
            print(f"\n✅ Tokenizer analysis completed!")
            print(f"📁 Results saved to: {args.output_dir}/")
            print("Generated files:")
            print(f"  • {tokenizer_name}_tokenizer_tsne.png - t-SNE visualization with ternary systems")
            print(f"  • {tokenizer_name}_tokenizer_results.npz - Numerical results")
            print("")
            print("🔍 Interpretation:")
            print(f"  • Embedding dimension: {embeddings.shape[1]}")
            print(f"  • Clear ternary clustering indicates good composition representation")
            print(f"  • Element-based tokenizer uses weighted sum of element embeddings")


if __name__ == "__main__":
    main()
