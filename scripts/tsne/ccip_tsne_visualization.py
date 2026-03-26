"""
CCIP Embedding t-SNE Visualization with Spectral Colors

This script:
1. Loads a trained CCIP model
2. Generates embeddings for compositions and images
3. Calculates t-SNE for dimensionality reduction
4. Plots embeddings colored by RGB values derived from material spectra
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import h5py
from tqdm import tqdm
import argparse
import os
from PIL import Image

from eval_ccip import load_checkpoint, CCIPEvaluator, custom_collate_fn
from train_ccip_simple import CompositionDataset
from sd.data_helper import HDF5CompositionImageDataset
from embedding.element_tokenizer import CompositionTokenizer


def extract_dominant_colors_from_spectra(spectra, energy_eV=None, color_weights=(1.2, 1.1, 0.6)):
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
        # Method 1: Energy-based RGB mapping (physically meaningful)
        # Map different energy ranges to RGB channels
        
        # Define energy ranges for RGB mapping (adjusted for your data range)
        energy_min, energy_max = energy_eV.min(), energy_eV.max()
        energy_range = energy_max - energy_min
        
        # Alternative approach: Use overlapping ranges to reduce blue dominance
        # and create more balanced color mixing
        red_range = (energy_min, energy_min + energy_range * 0.5)      # Cover more of low-mid range
        green_range = (energy_min + energy_range * 0.25, energy_min + energy_range * 0.85)  # Broader mid range
        blue_range = (energy_min + energy_range * 0.6, energy_max)     # Smaller high-energy range
        
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
        
        # Apply weights to balance the color channels and reduce blue dominance
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
        # Method 2: Band-based mapping (when no energy info available)
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


def extract_colors_from_images(images):
    """
    Extract dominant colors from material images
    
    Args:
        images: Tensor of shape (N, C, H, W) or numpy array
        
    Returns:
        colors: Array of shape (N, 3) with RGB values [0, 1]
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()
    
    # Handle different input formats
    if images.ndim == 4:  # (N, C, H, W)
        if images.shape[1] == 3:  # RGB format
            # Convert to (N, H, W, C)
            images = np.transpose(images, (0, 2, 3, 1))
        else:
            # Handle other channel configurations
            images = images[:, 0:3, :, :].transpose(0, 2, 3, 1)
    
    n_samples = images.shape[0]
    colors = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        img = images[i]
        
        # Ensure image is in correct format and range
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Calculate mean color
        mean_color = img.mean(axis=(0, 1))
        
        # Alternative: Use histogram-based dominant color
        # Reshape image for k-means clustering
        img_reshaped = img.reshape(-1, 3)
        
        # Simple dominant color: median of each channel
        dominant_color = np.median(img_reshaped, axis=0)
        
        colors[i] = dominant_color / 255.0  # Normalize to [0, 1]
    
    return colors


def compute_embeddings_with_data(evaluator, dataloader, include_spectra=True):
    """
    Compute embeddings and extract additional data for visualization
    
    Returns:
        comp_embeddings: Composition embeddings (N, dim)
        img_embeddings: Image embeddings (N, dim)
        colors: RGB colors from spectra/images (N, 3)
        compositions: List of composition dictionaries
        indices: Dataset indices
    """
    all_comp_embeddings = []
    all_img_embeddings = []
    all_colors = []
    all_compositions = []
    all_indices = []
    
    print("Computing embeddings and extracting colors...")
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        composition_tokens = batch['composition_tokens']
        attention_mask = batch['attention_mask']
        images = batch['image']
        composition_dicts = batch['composition_dict']
        indices = batch['index']
        
        # Compute embeddings
        comp_embeddings = evaluator.encode_compositions(composition_tokens, attention_mask)
        img_embeddings = evaluator.encode_images(images)
        
        # Extract colors from images
        batch_colors = extract_colors_from_images(images)
        
        # Store results
        all_comp_embeddings.append(comp_embeddings.cpu())
        all_img_embeddings.append(img_embeddings.cpu())
        all_colors.append(batch_colors)
        all_compositions.extend(composition_dicts)
        all_indices.extend(indices.tolist())
    
    # Concatenate all results
    comp_embeddings = torch.cat(all_comp_embeddings, dim=0)
    img_embeddings = torch.cat(all_img_embeddings, dim=0)
    colors = np.vstack(all_colors)
    
    return comp_embeddings, img_embeddings, colors, all_compositions, all_indices


def load_spectra_colors(data_path, indices, color_weights=(1.2, 1.1, 0.6)):
    """
    Load spectra and energy data, then extract RGB colors
    
    Args:
        data_path: Path to HDF5 file
        indices: List of sample indices
        color_weights: Tuple of (red_weight, green_weight, blue_weight)
        
    Returns:
        colors: RGB colors from spectra (N, 3)
    """
    print("Loading spectra and energy data...")
    
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
            energy_info = None
            if 'energy_eV' in f.keys():
                energy_eV = f['energy_eV'][:]
                energy_info = (energy_eV.min(), energy_eV.max())
                print(f"Loaded energy data shape: {energy_eV.shape}")
                print(f"Energy range: {energy_eV.min():.2f} - {energy_eV.max():.2f} eV")
            else:
                print("Warning: 'energy_eV' not found in dataset")
                # Check for alternative energy keys
                energy_candidates = [key for key in available_keys if 'energy' in key.lower() or 'ev' in key.lower()]
                if energy_candidates:
                    print(f"Found potential energy datasets: {energy_candidates}")
                    # Try to use the first one
                    energy_key = energy_candidates[0]
                    energy_eV = f[energy_key][:]
                    energy_info = (energy_eV.min(), energy_eV.max())
                    print(f"Using '{energy_key}' as energy data: shape {energy_eV.shape}")
            
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
            energy_info = None
    
    return colors, energy_info


def create_tsne_visualization_with_ternary(embeddings, colors, compositions, labels, title, 
                                         save_path=None, perplexity=30, energy_info=None, dataset_type="", max_systems=10):
    """
    Create t-SNE visualization with colored points and ternary system annotations
    
    Args:
        embeddings: Embeddings array (N, dim)
        colors: RGB colors (N, 3)
        compositions: List of composition dictionaries
        labels: Point labels for legend
        title: Plot title
        save_path: Path to save figure
        perplexity: t-SNE perplexity parameter
        energy_info: Tuple of (energy_min, energy_max) for annotation
        dataset_type: "train" or "test" for labeling
    """
    print(f"Computing t-SNE for {embeddings.shape[0]} samples...")
    
    # Ensure embeddings are numpy array
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()
    
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
    ternary_labels, ternary_colors, ternary_markers = identify_ternary_regions(compositions, max_systems)
    
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
    
    ax1.set_title(f'{title} - Spectral Colors\n({dataset_type} dataset)', 
                  fontsize=12, weight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Ternary system regions
    unique_systems = list(set(ternary_labels))
    legend_elements = []
    
    for i, system in enumerate(unique_systems[:15]):  # Show top 15 systems in legend
        mask = np.array([label == system for label in ternary_labels])
        if np.sum(mask) > 0:
            scatter2 = ax2.scatter(
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
            if i < 15:  # Only add to legend if it fits
                legend_elements.append(scatter2)
    
    ax2.set_title(f'{title} - Ternary Systems\n({dataset_type} dataset)', 
                  fontsize=12, weight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.grid(True, alpha=0.3)
    
    # Add legend for ternary systems (outside plot)
    if len(unique_systems) <= 15:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax2.text(1.05, 1.0, f'Showing {len(unique_systems)} ternary systems\n(Top 15 in legend)', 
                transform=ax2.transAxes, fontsize=9, verticalalignment='top')
        ax2.legend(bbox_to_anchor=(1.05, 0.9), loc='upper left', fontsize=7)
    
    # Add annotations
    if energy_info:
        energy_min, energy_max = energy_info
        fig.text(0.02, 0.02, 
                f'Left: Spectral colors from {energy_min:.2f}-{energy_max:.2f} eV\n'
                f'Right: Ternary composition systems\n'
                'Similar clustering indicates good CCIP embedding quality',
                fontsize=9, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    
    return tsne_results, ternary_labels


def create_combined_visualization(comp_embeddings, img_embeddings, 
                                comp_colors, img_colors, compositions, save_dir="tsne_results", energy_info=None, max_systems=10):
    """
    Create combined visualization of composition and image embeddings with ternary regions
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Composition embeddings only
    print("\n" + "="*50)
    print("Creating composition embeddings t-SNE...")
    comp_tsne, comp_ternary = create_tsne_visualization_with_ternary(
        comp_embeddings, comp_colors, compositions,
        labels=['Compositions'],
        title='CCIP Composition Embeddings',
        save_path=f'{save_dir}/composition_tsne.png',
        energy_info=energy_info,
        dataset_type="test",
        max_systems=max_systems
    )
    
    # 2. Image embeddings only
    print("\n" + "="*50)
    print("Creating image embeddings t-SNE...")
    img_tsne, img_ternary = create_tsne_visualization_with_ternary(
        img_embeddings, img_colors, compositions,
        labels=['Images'],
        title='CCIP Image Embeddings', 
        save_path=f'{save_dir}/image_tsne.png',
        energy_info=energy_info,
        dataset_type="test",
        max_systems=max_systems
    )
    
    # 3. Combined embeddings
    print("\n" + "="*50)
    print("Creating combined embeddings t-SNE...")
    
    # Combine embeddings
    combined_embeddings = np.vstack([comp_embeddings.numpy(), img_embeddings.numpy()])
    combined_colors = np.vstack([comp_colors, img_colors])
    
    # Create labels for different types
    n_comp = comp_embeddings.shape[0]
    n_img = img_embeddings.shape[0]
    
    # Combined t-SNE
    tsne_combined = TSNE(
        n_components=2,
        perplexity=min(30, combined_embeddings.shape[0] - 1),
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_embeddings)
    combined_tsne_results = tsne_combined.fit_transform(combined_scaled)
    
    # Plot combined results
    plt.figure(figsize=(14, 10))
    
    # Plot compositions
    plt.scatter(
        combined_tsne_results[:n_comp, 0],
        combined_tsne_results[:n_comp, 1],
        c=combined_colors[:n_comp],
        s=30,
        alpha=0.7,
        marker='o',
        label='Compositions',
        edgecolors='black',
        linewidth=0.2
    )
    
    # Plot images
    plt.scatter(
        combined_tsne_results[n_comp:, 0],
        combined_tsne_results[n_comp:, 1],
        c=combined_colors[n_comp:],
        s=30,
        alpha=0.7,
        marker='s',
        label='Images',
        edgecolors='red',
        linewidth=0.2
    )
    
    plt.title('CCIP Combined Embeddings\nt-SNE Visualization (Circles=Compositions, Squares=Images)', 
              fontsize=14, weight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.figtext(0.02, 0.02,
                'Colors represent RGB values from material spectra/images\n'
                'Close points indicate similar learned representations',
                fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/combined_tsne.png', dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to: {save_dir}/combined_tsne.png")
    
    plt.show()
    
    return {
        'composition_tsne': comp_tsne,
        'image_tsne': img_tsne,
        'combined_tsne': combined_tsne_results
    }


def identify_ternary_regions(compositions, max_systems=10):
    """
    Identify ternary composition regions and assign colors/markers
    
    Args:
        compositions: List of composition dictionaries
        max_systems: Maximum number of ternary systems to show (default: 10)
        
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
    
    # Get unique ternary systems and limit to max_systems most common
    unique_systems = list(set(ternary_systems))
    print(f"\nFound {len(unique_systems)} unique ternary systems:")
    
    # Count compositions per system
    from collections import Counter
    system_counts = Counter(ternary_systems)
    
    # Get the top max_systems systems by count
    top_systems = [system for system, count in system_counts.most_common(max_systems)]
    
    # Map less common systems to "Other"
    mapped_ternary_systems = []
    for system in ternary_systems:
        if system in top_systems:
            mapped_ternary_systems.append(system)
        else:
            mapped_ternary_systems.append("Other")
    
    # Update unique systems to include "Other" if needed
    unique_mapped_systems = list(set(mapped_ternary_systems))
    
    # Print top systems
    for system, count in system_counts.most_common(min(max_systems, 10)):
        print(f"  {system}: {count} compositions")
    
    if len(unique_systems) > max_systems:
        other_count = sum(count for system, count in system_counts.items() if system not in top_systems)
        print(f"  Other ({len(unique_systems) - max_systems} systems): {other_count} compositions")
    
    # Assign colors and markers to ternary systems
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    
    # Use distinct colors for different systems
    num_systems = len(unique_mapped_systems)
    colors = plt.cm.tab20(np.linspace(0, 1, min(num_systems, 20)))
    if num_systems > 20:
        # Use additional colormap for more systems
        extra_colors = plt.cm.Set3(np.linspace(0, 1, num_systems - 20))
        colors = np.vstack([colors, extra_colors])
    
    # Assign markers
    markers = ['o', 's', '^', 'v', '<', '>', 'D', 'h', 'p', '*', '+', 'x', '8', 'H', 'd', '|', '_']
    marker_cycle = markers * ((num_systems // len(markers)) + 1)
    
    ternary_colors = {system: colors[i] for i, system in enumerate(unique_mapped_systems)}
    ternary_markers = {system: marker_cycle[i] for i, system in enumerate(unique_mapped_systems)}
    
    return mapped_ternary_systems, ternary_colors, ternary_markers


def process_dataset(evaluator, data_path, indices, tokenizer, model_args, batch_size, color_weights, dataset_name=""):
    """
    Process a dataset and extract embeddings and colors.
    Note: Ternary filtering should be done on indices before calling this function.
    """
    print(f"\n{'='*50}")
    print(f"PROCESSING {dataset_name.upper()} DATASET")
    print(f"{'='*50}")
    
    hdf5_dataset = HDF5CompositionImageDataset(data_path, tokenizer=tokenizer)
    
    # Create dataset and dataloader
    dataset = CompositionDataset(
        hdf5_dataset, tokenizer, indices=indices,
        max_length=model_args.max_length
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
    )
    
    # Compute embeddings and extract colors
    comp_embeddings, img_embeddings, img_colors, compositions, processed_indices = compute_embeddings_with_data(
        evaluator, dataloader
    )
    
    # Get colors from spectra
    spectra_colors, energy_info = load_spectra_colors(data_path, processed_indices, color_weights)
    
    return {
        'comp_embeddings': comp_embeddings,
        'img_embeddings': img_embeddings,
        'spectra_colors': spectra_colors,
        'img_colors': img_colors,
        'compositions': compositions,
        'indices': processed_indices,
        'energy_info': energy_info
    }


def create_combined_train_test_visualization(train_results, test_results, save_dir="tsne_results", max_systems=10):
    """
    Create visualization combining both train and test data with different markers
    """
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*50}")
    print("CREATING COMBINED TRAIN/TEST VISUALIZATION")
    print(f"{'='*50}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Combine embeddings
    all_comp_embeddings = torch.cat([train_results['comp_embeddings'], test_results['comp_embeddings']], dim=0)
    all_img_embeddings = torch.cat([train_results['img_embeddings'], test_results['img_embeddings']], dim=0)
    all_colors = np.vstack([train_results['spectra_colors'], test_results['spectra_colors']])
    all_compositions = train_results['compositions'] + test_results['compositions']
    
    # Create dataset labels
    n_train = len(train_results['compositions'])
    n_test = len(test_results['compositions'])
    dataset_labels = ['train'] * n_train + ['test'] * n_test
    
    print(f"Combined data: {n_train} train + {n_test} test = {len(all_compositions)} total samples")
    
    # Compute t-SNE for combined composition embeddings
    print("\nComputing combined composition embeddings t-SNE...")
    
    # Ensure embeddings are numpy array
    embeddings = all_comp_embeddings.cpu().numpy()
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Compute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(50, embeddings.shape[0] - 1),
        random_state=42,
        n_iter=1000,
        verbose=1
    )
    
    tsne_results = tsne.fit_transform(embeddings_scaled)
    
    # Identify ternary regions
    ternary_labels, ternary_colors, ternary_markers = identify_ternary_regions(all_compositions, max_systems)
    
    # Create comprehensive plot
    fig = plt.figure(figsize=(24, 16))
    
    # Create 2x3 subplot layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: All data with spectral colors, train/test markers
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Train data
    train_mask = np.array(dataset_labels) == 'train'
    ax1.scatter(tsne_results[train_mask, 0], tsne_results[train_mask, 1],
               c=all_colors[train_mask], marker='o', s=25, alpha=0.7,
               edgecolors='black', linewidth=0.2, label=f'Train ({n_train})')
    
    # Test data  
    test_mask = np.array(dataset_labels) == 'test'
    ax1.scatter(tsne_results[test_mask, 0], tsne_results[test_mask, 1],
               c=all_colors[test_mask], marker='s', s=25, alpha=0.7,
               edgecolors='red', linewidth=0.2, label=f'Test ({n_test})')
    
    ax1.set_title('Combined Data - Spectral Colors\n(Circles=Train, Squares=Test)', fontsize=12, weight='bold')
    ax1.set_xlabel('t-SNE Dimension 1')
    ax1.set_ylabel('t-SNE Dimension 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Train data only with ternary systems
    ax2 = fig.add_subplot(gs[0, 1])
    
    train_ternary = [ternary_labels[i] for i in range(len(ternary_labels)) if train_mask[i]]
    
    # Count train systems by frequency (not alphabetically!)
    from collections import Counter
    train_system_counts = Counter(train_ternary)
    unique_train_systems = [system for system, count in train_system_counts.most_common(10)]  # Top 10 by frequency
    
    for system in unique_train_systems:
        system_mask = np.array([label == system and train_mask[i] for i, label in enumerate(ternary_labels)])
        if np.sum(system_mask) > 0:
            ax2.scatter(tsne_results[system_mask, 0], tsne_results[system_mask, 1],
                       c=[ternary_colors[system]], marker=ternary_markers[system], s=30, alpha=0.7,
                       label=f'{system} ({np.sum(system_mask)})', edgecolors='black', linewidth=0.1)
    
    ax2.set_title(f'Train Data - Ternary Systems\n({n_train} samples)', fontsize=12, weight='bold')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Test data only with ternary systems
    ax3 = fig.add_subplot(gs[0, 2])
    
    test_ternary = [ternary_labels[i] for i in range(len(ternary_labels)) if test_mask[i]]
    
    # Count test systems by frequency (not alphabetically!)
    test_system_counts = Counter(test_ternary)
    unique_test_systems = [system for system, count in test_system_counts.most_common(10)]  # Top 10 by frequency
    
    for system in unique_test_systems:
        system_mask = np.array([label == system and test_mask[i] for i, label in enumerate(ternary_labels)])
        if np.sum(system_mask) > 0:
            ax3.scatter(tsne_results[system_mask, 0], tsne_results[system_mask, 1],
                       c=[ternary_colors[system]], marker=ternary_markers[system], s=30, alpha=0.7,
                       label=f'{system} ({np.sum(system_mask)})', edgecolors='black', linewidth=0.1)
    
    ax3.set_title(f'Test Data - Ternary Systems\n({n_test} samples)', fontsize=12, weight='bold')
    ax3.set_xlabel('t-SNE Dimension 1')
    ax3.set_ylabel('t-SNE Dimension 2')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: All ternary systems combined (bottom row, spanning full width)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Show all unique ternary systems
    all_unique_systems = list(set(ternary_labels))
    from collections import Counter
    system_counts = Counter(ternary_labels)
    
    # Plot top 20 most common systems
    top_systems = [system for system, count in system_counts.most_common(20)]
    
    for system in top_systems:
        system_mask = np.array([label == system for label in ternary_labels])
        if np.sum(system_mask) > 0:
            # Use different markers for train/test within each system
            train_system_mask = system_mask & train_mask
            test_system_mask = system_mask & test_mask
            
            if np.sum(train_system_mask) > 0:
                ax4.scatter(tsne_results[train_system_mask, 0], tsne_results[train_system_mask, 1],
                           c=[ternary_colors[system]], marker='o', s=25, alpha=0.7,
                           edgecolors='black', linewidth=0.1)
            
            if np.sum(test_system_mask) > 0:
                ax4.scatter(tsne_results[test_system_mask, 0], tsne_results[test_system_mask, 1],
                           c=[ternary_colors[system]], marker='s', s=25, alpha=0.7,
                           edgecolors='red', linewidth=0.1)
    
    ax4.set_title(f'All Ternary Systems (Top 20)\nCircles=Train, Squares=Test | Total: {len(all_unique_systems)} unique systems', 
                  fontsize=14, weight='bold')
    ax4.set_xlabel('t-SNE Dimension 1')
    ax4.set_ylabel('t-SNE Dimension 2')
    ax4.grid(True, alpha=0.3)
    
    # Add overall annotation
    fig.text(0.02, 0.02, 
             f'CCIP Model Evaluation: Train vs Test Data Embedding Quality\n'
             f'Good model: Similar ternary systems cluster together in both train and test\n'
             f'Train: {n_train} samples, Test: {n_test} samples, Ternary systems: {len(all_unique_systems)}',
             fontsize=10, style='italic')
    
    plt.savefig(f'{save_dir}/combined_train_test_tsne.png', dpi=300, bbox_inches='tight')
    print(f"Saved combined train/test plot to: {save_dir}/combined_train_test_tsne.png")
    
    return tsne_results, ternary_labels, dataset_labels


def analyze_clusters(tsne_results, colors, compositions, n_clusters=5):
    """
    Analyze clusters in t-SNE space and their spectral characteristics
    """
    from sklearn.cluster import KMeans
    
    print(f"\nAnalyzing {n_clusters} clusters in t-SNE space...")
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tsne_results)
    
    # Analyze each cluster
    for i in range(n_clusters):
        cluster_mask = cluster_labels == i
        cluster_colors = colors[cluster_mask]
        cluster_compositions = [compositions[j] for j in range(len(compositions)) if cluster_mask[j]]
        
        print(f"\nCluster {i} ({np.sum(cluster_mask)} samples):")
        avg_color = cluster_colors.mean(axis=0)
        std_color = cluster_colors.std(axis=0)
        print(f"  Average color: RGB({avg_color[0]:.3f}, {avg_color[1]:.3f}, {avg_color[2]:.3f})")
        print(f"  Color std: RGB({std_color[0]:.3f}, {std_color[1]:.3f}, {std_color[2]:.3f})")
        
        # Most common elements in this cluster
        if cluster_compositions:
            all_elements = []
            for comp in cluster_compositions[:10]:  # Sample first 10
                all_elements.extend(comp.keys())
            
            from collections import Counter
            element_counts = Counter(all_elements)
            print(f"  Common elements: {list(element_counts.most_common(5))}")
    
    return cluster_labels


def filter_compositions_by_element_count(compositions, indices, min_elements=None, max_elements=None, exact_elements=None):
    """
    Filter compositions based on the number of elements present
    
    Args:
        compositions: List of composition dictionaries
        indices: Corresponding indices 
        min_elements: Minimum number of elements (inclusive)
        max_elements: Maximum number of elements (inclusive)
        exact_elements: Exact number of elements (overrides min/max)
        
    Returns:
        filtered_compositions: Filtered list of compositions
        filtered_indices: Corresponding filtered indices
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
    filtered_indices = [idx for i, idx in enumerate(indices) if filter_mask[i]]
    
    return filtered_compositions, filtered_indices, filter_mask


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
    hdf5_dataset = HDF5CompositionImageDataset(data_path)
    
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
    parser = argparse.ArgumentParser(description='CCIP t-SNE Visualization with Spectral Colors')
    
    parser.add_argument('--checkpoint_path', type=str, 
                       default='checkpoints/ccip/ccip_best.pth',
                       help='Path to CCIP checkpoint')
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
    parser.add_argument('--embedding_name', type=str, default='elem',
                       help='Embedding tokenizer name')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for processing')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to process')
    parser.add_argument('--output_dir', type=str, default='tsne_results',
                       help='Output directory for results')
    parser.add_argument('--perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--use_spectra_colors', action='store_true',
                       help='Use colors from spectra instead of images')
    parser.add_argument('--color_weights', nargs=3, type=float, default=[1.2, 1.1, 0.6],
                       help='RGB color weights for spectral mapping (red green blue). Default: 1.2 1.1 0.6')
    parser.add_argument('--max_systems', type=int, default=10,
                       help='Maximum number of ternary systems to show in visualizations (default: 10)')
    parser.add_argument('--ternary_only', action='store_true',
                       help='Only analyze compositions that are true ternary systems (exactly 3 elements)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading CCIP model...")
    composition_encoder, image_encoder, model_args = load_checkpoint(args.checkpoint_path, device)
    evaluator = CCIPEvaluator(composition_encoder, image_encoder, device)
    
    # Load data
    print("Loading dataset...")
    tokenizer = CompositionTokenizer(f'embedding/tokenizers/{args.embedding_name}-embedding.json')
    hdf5_dataset = HDF5CompositionImageDataset(args.data_path, tokenizer=tokenizer)
    
    # Load test indices
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
        
        # Process both datasets
        color_weights = tuple(args.color_weights)
        print(f"Using color weights: R={color_weights[0]}, G={color_weights[1]}, B={color_weights[2]}")
        
        train_results = process_dataset(
            evaluator, args.data_path, train_indices, tokenizer, 
            model_args, args.batch_size, color_weights, "TRAIN"
        )
        
        test_results = process_dataset(
            evaluator, args.data_path, test_indices, tokenizer, 
            model_args, args.batch_size, color_weights, "TEST"
        )
        
        # Create combined visualization
        combined_tsne, ternary_labels, dataset_labels = create_combined_train_test_visualization(
            train_results, test_results, save_dir=args.output_dir, max_systems=args.max_systems
        )
        
        print(f"\n✅ Combined train/test results saved to: {args.output_dir}/")
        print("Generated files:")
        print(f"  • combined_train_test_tsne.png - Combined train/test visualization with ternary systems")
        
    else:
        # Process single dataset (either train or test)
        if args.use_train_data:
            indices = train_indices
            dataset_name = "TRAIN"
        else:
            indices = test_indices  
            dataset_name = "TEST"
        
        # Pre-filter for ternary compositions if requested
        if args.ternary_only:
            indices = filter_indices_for_ternary_compositions(args.data_path, indices, exact_elements=3)
            if indices is None:
                print(f"❌ Error: No ternary compositions found in {dataset_name} dataset!")
                return
        
        # Limit samples if requested (after ternary filtering)
        if args.max_samples < len(indices):
            indices = indices[:args.max_samples]
        
        print(f"Processing {len(indices)} {dataset_name.lower()} samples...")
        
        # Create dataset and dataloader
        test_dataset = CompositionDataset(
            hdf5_dataset, tokenizer, indices=indices,
            max_length=model_args.max_length
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True, collate_fn=custom_collate_fn
        )
        
        # Compute embeddings and extract colors
        comp_embeddings, img_embeddings, img_colors, compositions, indices = compute_embeddings_with_data(
            evaluator, test_loader
        )
        
        # Get colors from spectra if requested
        energy_info = None
        if args.use_spectra_colors:
            print("\n" + "="*50)
            print("EXTRACTING COLORS FROM SPECTRAL DATA")
            print("="*50)
            color_weights = tuple(args.color_weights)
            print(f"Using color weights: R={color_weights[0]}, G={color_weights[1]}, B={color_weights[2]}")
            spectra_colors, energy_info = load_spectra_colors(args.data_path, indices, color_weights)
            comp_colors = spectra_colors
            img_colors = spectra_colors
            print("Using spectral data for both composition and image colors")
        else:
            print("\n" + "="*50)
            print("USING COLORS FROM MATERIAL IMAGES")
            print("="*50)
            comp_colors = img_colors
            print("Using image-derived colors for both composition and image points")
        
        print(f"Generated embeddings:")
        print(f"  Compositions: {comp_embeddings.shape}")
        print(f"  Images: {img_embeddings.shape}")
        print(f"  Colors: {comp_colors.shape}")
        
        # Create visualizations
        tsne_results = create_combined_visualization(
            comp_embeddings, img_embeddings,
            comp_colors, img_colors, compositions,
            save_dir=args.output_dir, energy_info=energy_info,
            max_systems=args.max_systems
        )
        
        # Analyze clusters
        print("\n" + "="*50)
        comp_clusters = analyze_clusters(
            tsne_results['composition_tsne'], comp_colors, compositions
        )
        
        # Save results
        results = {
            'composition_embeddings': comp_embeddings.numpy(),
            'image_embeddings': img_embeddings.numpy(),
            'colors': comp_colors,
            'compositions': compositions,
            'indices': indices,
            'tsne_results': tsne_results
        }
        
        np.savez_compressed(
            f'{args.output_dir}/ccip_tsne_results.npz',
            **{k: v for k, v in results.items() if k != 'tsne_results'}
        )
        
        print(f"\n✅ Results saved to: {args.output_dir}/")
        print("Generated files:")
        print(f"  • composition_tsne.png - Composition embeddings t-SNE with ternary systems")
        print(f"  • image_tsne.png - Image embeddings t-SNE with ternary systems") 
        print(f"  • combined_tsne.png - Combined embeddings t-SNE")
        print(f"  • ccip_tsne_results.npz - Numerical results")


if __name__ == "__main__":
    main()
