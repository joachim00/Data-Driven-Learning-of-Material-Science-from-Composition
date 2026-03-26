#!/usr/bin/env python3
"""Assemble individual ternary plot PNGs into a multi-page PDF with grid layout.

This script takes a directory of individual ternary plot images and assembles them
into a high-quality PDF with specified paper size and grid layout. It also generates
a CSV index file mapping compositions to their page/row/column locations.

Example usage:
    python assemble_pdf.py \
        --input-dir ternary_plots_1000 \
        --output-pdf ternary_a2_poster.pdf \
        --paper-size A2 \
        --dpi 300 \
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
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, A2, A3, letter
from reportlab.lib.units import inch


# Paper size definitions (width x height in points, landscape)
PAPER_SIZES = {
    'A4': (11.69 * inch, 8.27 * inch),
    'A3': (16.54 * inch, 11.69 * inch),
    'A2': (23.39 * inch, 16.54 * inch),
    'letter': (11 * inch, 8.5 * inch),
}


def load_metadata(metadata_path: Path) -> List[dict]:
    """Load metadata CSV file created during plot generation."""
    metadata = []
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return metadata


def create_pdf_from_images(
    image_dir: Path,
    output_pdf: Path,
    metadata: List[dict],
    *,
    paper_size: str = 'A2',
    dpi: int = 300,
    nrow: int = 8,
    ncol: int = 11,
    margin: float = 0.5,
    add_labels: bool = True,
    add_page_numbers: bool = True,
) -> List[dict]:
    """Assemble PNG images into a multi-page PDF.
    
    Args:
        image_dir: Directory containing PNG files
        output_pdf: Output PDF path
        metadata: List of metadata dicts from plots_metadata.csv
        paper_size: Paper size name ('A2', 'A3', 'A4', 'letter')
        dpi: DPI for the PDF (doesn't affect input images, just PDF metadata)
        nrow: Number of rows per page
        ncol: Number of columns per page
        margin: Margin in inches
        add_labels: Add row/column labels to each plot
        add_page_numbers: Add page numbers
        
    Returns:
        List of index entries with page/row/column information
    """
    if paper_size not in PAPER_SIZES:
        raise ValueError(f"Unknown paper size '{paper_size}'. Choose from {list(PAPER_SIZES.keys())}")
    
    page_width, page_height = PAPER_SIZES[paper_size]
    margin_pts = margin * inch
    
    # Calculate grid dimensions
    grid_width = page_width - 2 * margin_pts
    grid_height = page_height - 2 * margin_pts
    
    cell_width = grid_width / ncol
    cell_height = grid_height / nrow
    
    # Calculate plots per page
    plots_per_page = nrow * ncol
    total_plots = len(metadata)
    num_pages = math.ceil(total_plots / plots_per_page)
    
    print(f"Creating PDF: {output_pdf}")
    print(f"  Paper size: {paper_size} ({page_width/inch:.2f}\" x {page_height/inch:.2f}\")")
    print(f"  Grid: {nrow} rows x {ncol} columns = {plots_per_page} plots/page")
    print(f"  Total plots: {total_plots}")
    print(f"  Total pages: {num_pages}")
    print(f"  Cell size: {cell_width/inch:.3f}\" x {cell_height/inch:.3f}\"")
    
    # Create PDF
    c = canvas.Canvas(str(output_pdf), pagesize=(page_width, page_height))
    
    index_entries = []
    
    for page_num in range(num_pages):
        print(f"\n  Page {page_num + 1}/{num_pages}...")
        
        start_idx = page_num * plots_per_page
        end_idx = min(start_idx + plots_per_page, total_plots)
        page_metadata = metadata[start_idx:end_idx]
        
        # Add page number
        if add_page_numbers:
            c.setFont("Helvetica-Bold", 24)
            c.drawCentredString(page_width / 2, page_height - margin_pts / 2, 
                               f'Page {page_num + 1}/{num_pages}')
        
        # Place images in grid
        for plot_idx, meta in enumerate(page_metadata):
            row = plot_idx // ncol
            col = plot_idx % ncol
            
            # Calculate position (origin is bottom-left in reportlab)
            x = margin_pts + col * cell_width
            y = page_height - margin_pts - (row + 1) * cell_height
            
            # Load and place image
            img_path = Path(meta['filepath'])
            if img_path.exists():
                try:
                    # Draw image
                    c.drawImage(str(img_path), x, y, 
                               width=cell_width, height=cell_height,
                               preserveAspectRatio=True, anchor='c')
                    
                    # Add label
                    if add_labels:
                        label = f"R{row + 1}C{col + 1}"
                        c.setFont("Helvetica-Bold", 12)
                        c.setFillColorRGB(0, 0, 0)
                        # Position label at top-center of cell
                        c.drawCentredString(x + cell_width / 2, y + cell_height - 15, label)
                    
                except Exception as e:
                    print(f"    Warning: Failed to load {img_path.name}: {e}")
            else:
                print(f"    Warning: Image not found: {img_path}")
            
            # Create index entry
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
        
        # Finish page
        c.showPage()
        print(f"    Added {len(page_metadata)} plots")
    
    # Save PDF
    c.save()
    print(f"\n✓ PDF saved to {output_pdf}")
    
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
        comp_key = tuple(sorted([
            entry['element_1'],
            entry['element_2'], 
            entry['element_3']
        ]))
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
            'composition_lookup': {str(k): v for k, v in composition_lookup.items()},
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
        choices=list(PAPER_SIZES.keys()),
        default='A2',
        help='Paper size (default: A2)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PDF metadata (default: 300)'
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
        '--metadata-file',
        type=Path,
        help='Path to metadata CSV file (default: <input-dir>/plots_metadata.csv)'
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
    
    # Create PDF
    index_entries = create_pdf_from_images(
        args.input_dir,
        args.output_pdf,
        metadata,
        paper_size=args.paper_size,
        dpi=args.dpi,
        nrow=args.nrow,
        ncol=args.ncol,
        margin=args.margin,
        add_labels=not args.no_labels,
        add_page_numbers=not args.no_page_numbers,
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


if __name__ == '__main__':
    main()
