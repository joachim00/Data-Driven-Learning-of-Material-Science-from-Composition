#!/usr/bin/env python3
"""Assemble individual ternary plot PNGs into a multi-page PDF with grid layout.

This script takes a directory of individual ternary plot images and assembles them
into a high-quality PDF with specified paper size and grid layout. It also generates
a CSV index file mapping compositions to their page/row/column locations.

Uses PIL and img2pdf for maximum compatibility.

Example usage:
    python assemble_pdf_simple.py \
        --input-dir ternary_plots_1000 \
        --output-pdf ternary_a2_poster.pdf \
        --paper-size A2 \
        --nrow 8 \
        --ncol 11 \
        --margin 0.5
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Paper size definitions (width x height in pixels at 300 DPI, landscape)
PAPER_SIZES_MM = {
    'A4': (297, 210),
    'A3': (420, 297),
    'A2': (594, 420),
    'letter': (279.4, 215.9),
}


def mm_to_pixels(mm: float, dpi: int = 300) -> int:
    """Convert millimeters to pixels at given DPI."""
    return int(mm * dpi / 25.4)


def load_metadata(metadata_path: Path) -> List[dict]:
    """Load metadata CSV file created during plot generation."""
    metadata = []
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return metadata


def load_filter_indices(filter_path: Path) -> set:
    """Load indices from a CSV file.
    
    The CSV should have an 'index' column with the plot indices to include.
    """
    indices = set()
    with open(filter_path, 'r') as f:
        reader = csv.DictReader(f)
        if 'index' not in reader.fieldnames:
            raise ValueError(f"Filter CSV must have an 'index' column. Found: {reader.fieldnames}")
        for row in reader:
            try:
                indices.add(int(row['index']))
            except (ValueError, KeyError) as e:
                print(f"Warning: Could not parse index from row: {row}")
    return indices


def load_filter_elements(filter_path: Path) -> set:
    """Load element combinations from a CSV file.
    
    The CSV should have element columns (e.g., 'element_1', 'element_2', 'element_3')
    or 'element_indices' column.
    """
    combinations = set()
    with open(filter_path, 'r') as f:
        reader = csv.DictReader(f)
        
        # Check which columns are available
        has_element_cols = all(col in reader.fieldnames for col in ['element_1', 'element_2', 'element_3'])
        has_indices_col = 'element_indices' in reader.fieldnames
        
        if not has_element_cols and not has_indices_col:
            raise ValueError(
                f"Filter CSV must have either element columns (element_1, element_2, element_3) "
                f"or element_indices column. Found: {reader.fieldnames}"
            )
        
        for row in reader:
            try:
                if has_element_cols:
                    # Use element names
                    elements = tuple(sorted([row['element_1'], row['element_2'], row['element_3']]))
                    combinations.add(elements)
                else:
                    # Use element indices
                    combinations.add(row['element_indices'])
            except KeyError as e:
                print(f"Warning: Missing required column in row: {row}")
    
    return combinations


def filter_metadata(
    metadata: List[dict],
    filter_indices: Optional[set] = None,
    filter_elements: Optional[set] = None,
) -> List[dict]:
    """Filter metadata based on indices or element combinations.
    
    Args:
        metadata: Full metadata list
        filter_indices: Set of plot indices to include (optional)
        filter_elements: Set of element combinations to include (optional)
        
    Returns:
        Filtered metadata list
    """
    if filter_indices is None and filter_elements is None:
        return metadata
    
    filtered = []
    for meta in metadata:
        # Check index filter
        if filter_indices is not None:
            try:
                idx = int(meta.get('index', -1))
                if idx not in filter_indices:
                    continue
            except ValueError:
                continue
        
        # Check element filter
        if filter_elements is not None:
            # Try element names first
            if all(k in meta for k in ['element_1', 'element_2', 'element_3']):
                elements = tuple(sorted([meta['element_1'], meta['element_2'], meta['element_3']]))
                if elements not in filter_elements:
                    # Also try element_indices
                    if meta.get('element_indices') not in filter_elements:
                        continue
            elif meta.get('element_indices') not in filter_elements:
                continue
        
        filtered.append(meta)
    
    return filtered


def create_pdf_page_image(
    page_metadata: List[dict],
    image_dir: Path,
    page_width: int,
    page_height: int,
    nrow: int,
    ncol: int,
    margin: int,
    page_num: int,
    total_pages: int,
    add_labels: bool = True,
    add_page_numbers: bool = True,
    show_composition: bool = False,
) -> Image.Image:
    """Create a single PDF page as a PIL Image - OPTIMIZED."""
    
    import time
    start_time = time.time()
    
    # Create blank white page
    page = Image.new('RGB', (page_width, page_height), color='white')
    draw = ImageDraw.Draw(page)
    
    # Calculate grid dimensions
    grid_width = page_width - 2 * margin
    grid_height = page_height - 2 * margin
    
    cell_width = grid_width // ncol
    cell_height = grid_height // nrow
    
    # Try to load a nice font for labels
    try:
        # Try to find a system font
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
        font_tiny = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
            font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()
    
    # Add page number
    if add_page_numbers:
        text = f'Page {page_num + 1}/{total_pages}'
        # Calculate text position (center top)
        bbox = draw.textbbox((0, 0), text, font=font_large)
        text_width = bbox[2] - bbox[0]
        text_x = (page_width - text_width) // 2
        text_y = margin // 3
        draw.text((text_x, text_y), text, fill='black', font=font_large)
    
    # Place images in grid
    images_loaded = 0
    for plot_idx, meta in enumerate(page_metadata):
        row = plot_idx // ncol
        col = plot_idx % ncol
        
        # Calculate position (top-left corner of cell)
        x = margin + col * cell_width
        y = margin + row * cell_height
        
        # Load image (use filename from metadata + input directory)
        # This allows the script to work even if the metadata filepath is wrong
        filename = meta.get('filename', '')
        if filename:
            img_path = image_dir / filename
        else:
            # Fallback to filepath from metadata
            img_path = Path(meta['filepath'])
        
        if img_path.exists():
            try:
                # Open and immediately convert to RGB to avoid issues
                img = Image.open(img_path).convert('RGB')
                
                # Resize to fit cell while maintaining aspect ratio
                img.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                
                # Center image in cell
                img_x = x + (cell_width - img.width) // 2
                img_y = y + (cell_height - img.height) // 2
                
                # Paste image
                page.paste(img, (img_x, img_y))
                img.close()  # Free memory immediately
                images_loaded += 1
                
                # Add label
                if add_labels:
                    label = f"R{row + 1}C{col + 1}"
                    
                    # Get composition if requested
                    comp_text = ""
                    if show_composition:
                        # Try to get element names
                        e1 = meta.get('element_1', '')
                        e2 = meta.get('element_2', '')
                        e3 = meta.get('element_3', '')
                        if e1 and e2 and e3:
                            comp_text = f"{e1}-{e2}-{e3}"
                    
                    # Calculate label dimensions
                    label_bbox = draw.textbbox((0, 0), label, font=font_small)
                    label_width = label_bbox[2] - label_bbox[0]
                    label_height = label_bbox[3] - label_bbox[1]
                    
                    # Position label at top center of cell
                    label_x = x + (cell_width - label_width) // 2
                    label_y = y + 10
                    
                    # Draw label background and text
                    bg_padding = 5
                    total_height = label_height
                    
                    # If showing composition, calculate total height
                    if comp_text:
                        comp_bbox = draw.textbbox((0, 0), comp_text, font=font_tiny)
                        comp_width = comp_bbox[2] - comp_bbox[0]
                        comp_height = comp_bbox[3] - comp_bbox[1]
                        total_height += comp_height + 2
                        max_width = max(label_width, comp_width)
                    else:
                        max_width = label_width
                    
                    # Draw white background for labels
                    draw.rectangle([
                        x + (cell_width - max_width) // 2 - bg_padding,
                        label_y - bg_padding,
                        x + (cell_width - max_width) // 2 + max_width + bg_padding,
                        label_y + total_height + bg_padding
                    ], fill='white', outline='black')
                    
                    # Draw position label
                    draw.text((label_x, label_y), label, fill='black', font=font_small)
                    
                    # Draw composition if requested
                    if comp_text:
                        comp_x = x + (cell_width - comp_width) // 2
                        comp_y = label_y + label_height + 2
                        draw.text((comp_x, comp_y), comp_text, fill='darkblue', font=font_tiny)
                
            except Exception as e:
                print(f"    Warning: Failed to load {img_path.name}: {e}")
                # Draw error placeholder
                draw.rectangle([x, y, x + cell_width, y + cell_height], outline='red', width=2)
        else:
            print(f"    Warning: Image not found: {img_path}")
    
    elapsed = time.time() - start_time
    print(f"    Loaded {images_loaded}/{len(page_metadata)} images in {elapsed:.1f}s")
    
    return page


def create_pdf_from_images(
    image_dir: Path,
    output_pdf: Path,
    metadata: List[dict],
    *,
    paper_size: str = 'A2',
    dpi: int = 300,
    nrow: int = 8,
    ncol: int = 11,
    margin_inches: float = 0.5,
    add_labels: bool = True,
    add_page_numbers: bool = True,
    show_composition: bool = False,
) -> List[dict]:
    """Assemble PNG images into a multi-page PDF.
    
    Args:
        image_dir: Directory containing PNG files
        output_pdf: Output PDF path
        metadata: List of metadata dicts from plots_metadata.csv
        paper_size: Paper size name ('A2', 'A3', 'A4', 'letter')
        dpi: DPI for the PDF
        nrow: Number of rows per page
        ncol: Number of columns per page
        margin_inches: Margin in inches
        add_labels: Add row/column labels to each plot
        add_page_numbers: Add page numbers
        show_composition: Show element composition below labels
        
    Returns:
        List of index entries with page/row/column information
    """
    if paper_size not in PAPER_SIZES_MM:
        raise ValueError(f"Unknown paper size '{paper_size}'. Choose from {list(PAPER_SIZES_MM.keys())}")
    
    # Convert paper size to pixels
    width_mm, height_mm = PAPER_SIZES_MM[paper_size]
    page_width = mm_to_pixels(width_mm, dpi)
    page_height = mm_to_pixels(height_mm, dpi)
    margin = mm_to_pixels(margin_inches * 25.4, dpi)
    
    # Calculate grid dimensions
    grid_width = page_width - 2 * margin
    grid_height = page_height - 2 * margin
    
    cell_width = grid_width // ncol
    cell_height = grid_height // nrow
    
    # Calculate plots per page
    plots_per_page = nrow * ncol
    total_plots = len(metadata)
    num_pages = math.ceil(total_plots / plots_per_page)
    
    print(f"Creating PDF: {output_pdf}")
    print(f"  Paper size: {paper_size} ({width_mm}mm x {height_mm}mm)")
    print(f"  Resolution: {page_width}px x {page_height}px @ {dpi} DPI")
    print(f"  Grid: {nrow} rows x {ncol} columns = {plots_per_page} plots/page")
    print(f"  Total plots: {total_plots}")
    print(f"  Total pages: {num_pages}")
    print(f"  Cell size: {cell_width}px x {cell_height}px")
    
    # Create pages and save incrementally to avoid memory issues
    import time
    overall_start = time.time()
    first_page = None
    saved_pages = []
    index_entries = []
    
    for page_num in range(num_pages):
        page_start = time.time()
        print(f"\n  Page {page_num + 1}/{num_pages}...")
        
        start_idx = page_num * plots_per_page
        end_idx = min(start_idx + plots_per_page, total_plots)
        page_metadata = metadata[start_idx:end_idx]
        
        # Create page image
        page_img = create_pdf_page_image(
            page_metadata,
            image_dir,
            page_width,
            page_height,
            nrow,
            ncol,
            margin,
            page_num,
            num_pages,
            add_labels,
            add_page_numbers,
            show_composition,
        )
        
        # Store first page separately, collect others
        if page_num == 0:
            first_page = page_img
        else:
            saved_pages.append(page_img)
        
        # Create index entries
        for plot_idx, meta in enumerate(page_metadata):
            row = plot_idx // ncol
            col = plot_idx % ncol
            
            index_entries.append({
                'page': page_num + 1,
                'row': row + 1,
                'column': col + 1,
                'position_label': f"R{row + 1}C{col + 1}",
                'index': meta.get('index', start_idx + plot_idx),
                'element_indices': meta.get('element_indices', ''),
                'element_1': meta.get('element_1', ''),
                'element_2': meta.get('element_2', ''),
                'element_3': meta.get('element_3', ''),
                'combination': meta.get('combination', ''),
                'num_samples': meta.get('num_samples', ''),
            })
        
        page_time = time.time() - page_start
        elapsed = time.time() - overall_start
        avg_time = elapsed / (page_num + 1)
        remaining = avg_time * (num_pages - page_num - 1)
        print(f"    Page time: {page_time:.1f}s | Total: {elapsed:.1f}s | ETA: {remaining:.1f}s")
    
    # Save all pages as PDF
    print(f"\n  Saving PDF with {num_pages} pages...")
    save_start = time.time()
    if first_page is None:
        raise ValueError("No pages to save")
    
    if len(saved_pages) == 0:
        first_page.save(output_pdf, 'PDF', resolution=dpi, save_all=False)
    else:
        first_page.save(output_pdf, 'PDF', resolution=dpi, save_all=True, append_images=saved_pages)
    
    # Clean up
    first_page.close()
    for page in saved_pages:
        page.close()
    
    save_time = time.time() - save_start
    total_time = time.time() - overall_start
    print(f"  PDF save time: {save_time:.1f}s")
    print(f"✓ PDF created in {total_time:.1f}s total ({total_time/num_pages:.1f}s per page)")
    
    return index_entries


def save_index_csv(index_entries: List[dict], output_path: Path) -> None:
    """Save index entries to CSV file."""
    if not index_entries:
        return
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = list(index_entries[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_entries)
    
    print(f"✓ Index CSV saved to {output_path}")


def save_index_json(index_entries: List[dict], output_path: Path) -> None:
    """Save index entries to JSON file for easier programmatic access."""
    
    # Create a lookup dictionary: composition -> location
    composition_lookup = {}
    for entry in index_entries:
        # Use element composition as key
        elements = sorted([
            entry['element_1'],
            entry['element_2'], 
            entry['element_3']
        ])
        comp_key = '_'.join(elements)
        
        composition_lookup[comp_key] = {
            'page': entry['page'],
            'row': entry['row'],
            'column': entry['column'],
            'position_label': entry['position_label'],
            'element_indices': entry['element_indices'],
            'num_samples': entry['num_samples'],
        }
    
    with open(output_path, 'w') as f:
        json.dump({
            'total_entries': len(index_entries),
            'entries': index_entries,
            'composition_lookup': composition_lookup,
        }, f, indent=2)
    
    print(f"✓ Index JSON saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble individual ternary plot PNGs into a multi-page PDF."
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing PNG files and plots_metadata.csv'
    )
    parser.add_argument(
        '--output-pdf',
        type=Path,
        required=True,
        help='Output PDF file path'
    )
    parser.add_argument(
        '--paper-size',
        choices=list(PAPER_SIZES_MM.keys()),
        default='A2',
        help='Paper size (default: A2)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF (default: 300)'
    )
    parser.add_argument(
        '--nrow',
        type=int,
        default=8,
        help='Number of rows per page (default: 8)'
    )
    parser.add_argument(
        '--ncol',
        type=int,
        default=11,
        help='Number of columns per page (default: 11)'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.5,
        help='Margin in inches (default: 0.5)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Disable row/column labels on plots'
    )
    parser.add_argument(
        '--no-page-numbers',
        action='store_true',
        help='Disable page numbers'
    )
    parser.add_argument(
        '--show-composition',
        action='store_true',
        help='Show element composition under the row/column labels'
    )
    parser.add_argument(
        '--metadata-file',
        type=Path,
        help='Path to metadata CSV file (default: <input-dir>/plots_metadata.csv)'
    )
    parser.add_argument(
        '--filter-indices',
        type=Path,
        help='CSV file with plot indices to include (must have "index" column with integer values)'
    )
    parser.add_argument(
        '--filter-elements',
        type=Path,
        help='CSV file with element combinations to include (must have element_1/element_2/element_3 or element_indices columns)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Check input directory
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    
    # Load metadata
    metadata_file = args.metadata_file or (args.input_dir / 'plots_metadata.csv')
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_file}\n"
            f"Make sure the input directory contains plots_metadata.csv"
        )
    
    print(f"Loading metadata from {metadata_file}...")
    metadata = load_metadata(metadata_file)
    print(f"  Found {len(metadata)} plots")
    
    # Apply filters if provided
    filter_indices = None
    filter_elements = None
    
    if args.filter_indices:
        if not args.filter_indices.exists():
            raise FileNotFoundError(f"Filter indices file not found: {args.filter_indices}")
        print(f"\nLoading filter indices from {args.filter_indices}...")
        filter_indices = load_filter_indices(args.filter_indices)
        print(f"  Found {len(filter_indices)} indices to include")
    
    if args.filter_elements:
        if not args.filter_elements.exists():
            raise FileNotFoundError(f"Filter elements file not found: {args.filter_elements}")
        print(f"\nLoading filter elements from {args.filter_elements}...")
        filter_elements = load_filter_elements(args.filter_elements)
        print(f"  Found {len(filter_elements)} element combinations to include")
    
    # Apply filters
    if filter_indices is not None or filter_elements is not None:
        print("\nApplying filters...")
        original_count = len(metadata)
        metadata = filter_metadata(metadata, filter_indices, filter_elements)
        print(f"  Filtered from {original_count} to {len(metadata)} plots")
        
        if len(metadata) == 0:
            raise ValueError("No plots remaining after filtering!")
    
    # Create PDF
    index_entries = create_pdf_from_images(
        args.input_dir,
        args.output_pdf,
        metadata,
        paper_size=args.paper_size,
        dpi=args.dpi,
        nrow=args.nrow,
        ncol=args.ncol,
        margin_inches=args.margin,
        add_labels=not args.no_labels,
        add_page_numbers=not args.no_page_numbers,
        show_composition=args.show_composition,
    )
    
    # Save index files
    index_csv_path = args.output_pdf.with_suffix('.csv')
    index_json_path = args.output_pdf.with_suffix('.json')
    
    save_index_csv(index_entries, index_csv_path)
    save_index_json(index_entries, index_json_path)
    
    print(f"\n✓ Done! Created:")
    print(f"  - PDF: {args.output_pdf}")
    print(f"  - Index CSV: {index_csv_path}")
    print(f"  - Index JSON: {index_json_path}")
    print(f"\nTo lookup a composition, use the JSON file's 'composition_lookup' field.")
    print(f"Example: composition_lookup['Element1_Element2_Element3'] -> {{page, row, column}}")


if __name__ == '__main__':
    main()
