#!/usr/bin/env python3
"""Find optimal grid dimensions to minimize empty cells on the last page.

This script analyzes different grid configurations and suggests the best
row/column combination to minimize wasted space on the last page of a PDF.

Example usage:
    python find_optimal_grid.py --total-plots 11344 --paper-size A2
    python find_optimal_grid.py --total-plots 11344 --min-plots-per-page 50 --max-plots-per-page 150
"""

import argparse
import math
from typing import List, Tuple


# Paper size aspect ratios (width / height in landscape)
PAPER_ASPECT_RATIOS = {
    'A4': 11.69 / 8.27,
    'A3': 16.54 / 11.69,
    'A2': 23.39 / 16.54,
    'letter': 11.0 / 8.5,
}


def calculate_grid_stats(total_plots: int, nrow: int, ncol: int) -> dict:
    """Calculate statistics for a given grid configuration."""
    plots_per_page = nrow * ncol
    num_pages = math.ceil(total_plots / plots_per_page)
    total_cells = num_pages * plots_per_page
    empty_cells = total_cells - total_plots
    empty_cells_last_page = plots_per_page - (total_plots % plots_per_page) if total_plots % plots_per_page != 0 else 0
    utilization = total_plots / total_cells
    last_page_utilization = (plots_per_page - empty_cells_last_page) / plots_per_page
    
    return {
        'nrow': nrow,
        'ncol': ncol,
        'plots_per_page': plots_per_page,
        'num_pages': num_pages,
        'total_cells': total_cells,
        'empty_cells': empty_cells,
        'empty_cells_last_page': empty_cells_last_page,
        'utilization': utilization,
        'last_page_utilization': last_page_utilization,
    }


def find_optimal_grids(
    total_plots: int,
    paper_size: str = 'A2',
    min_plots_per_page: int = 50,
    max_plots_per_page: int = 200,
    prefer_square: bool = True,
    max_col_row_ratio: float = 3.0,
) -> List[dict]:
    """Find optimal grid configurations.
    
    Args:
        total_plots: Total number of plots to place
        paper_size: Paper size name for aspect ratio
        min_plots_per_page: Minimum plots per page to consider
        max_plots_per_page: Maximum plots per page to consider
        prefer_square: Prefer more square-like grids
        max_col_row_ratio: Maximum ratio of columns to rows (default: 3.0)
        
    Returns:
        List of grid configurations sorted by quality
    """
    aspect_ratio = PAPER_ASPECT_RATIOS.get(paper_size, 1.414)  # Default to A2
    
    candidates = []
    
    # Try different plots per page values
    for plots_per_page in range(min_plots_per_page, max_plots_per_page + 1):
        # Find factor pairs (nrow, ncol) for this plots_per_page
        for nrow in range(1, int(math.sqrt(plots_per_page)) + 1):
            if plots_per_page % nrow == 0:
                ncol = plots_per_page // nrow
                
                # Apply column-to-row ratio constraint
                col_row_ratio = ncol / nrow
                if col_row_ratio > max_col_row_ratio:
                    continue
                
                # Check if aspect ratio is reasonable for the paper
                grid_aspect = ncol / nrow
                aspect_diff = abs(grid_aspect - aspect_ratio)
                
                # Calculate stats
                stats = calculate_grid_stats(total_plots, nrow, ncol)
                
                # Score based on multiple factors
                # 1. Last page utilization (most important)
                last_page_score = stats['last_page_utilization'] * 100
                
                # 2. Overall utilization
                overall_score = stats['utilization'] * 10
                
                # 3. Aspect ratio match (prefer grids matching paper aspect)
                aspect_score = max(0, 10 - aspect_diff * 5)
                
                # 4. Prefer moderate page counts (not too many, not too few)
                ideal_pages = max(10, total_plots / 100)
                page_score = max(0, 10 - abs(stats['num_pages'] - ideal_pages) / ideal_pages * 10)
                
                # 5. Prefer more square grids if requested
                if prefer_square:
                    squareness = min(nrow, ncol) / max(nrow, ncol)
                    square_score = squareness * 5
                else:
                    square_score = 0
                
                total_score = last_page_score + overall_score + aspect_score + page_score + square_score
                
                candidates.append({
                    **stats,
                    'aspect_ratio': grid_aspect,
                    'aspect_diff': aspect_diff,
                    'score': total_score,
                    'last_page_score': last_page_score,
                })
    
    # Sort by score (descending)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return candidates


def print_grid_info(grid: dict, rank: int = None):
    """Print detailed information about a grid configuration."""
    if rank is not None:
        print(f"\n{'='*70}")
        print(f"Rank #{rank}")
    
    print(f"Grid: {grid['nrow']} rows × {grid['ncol']} columns = {grid['plots_per_page']} plots/page")
    print(f"Pages: {grid['num_pages']}")
    print(f"Total cells: {grid['total_cells']} ({grid['empty_cells']} empty)")
    print(f"Last page: {grid['plots_per_page'] - grid['empty_cells_last_page']}/{grid['plots_per_page']} filled " +
          f"({grid['last_page_utilization']*100:.1f}% utilization)")
    print(f"Overall utilization: {grid['utilization']*100:.1f}%")
    print(f"Aspect ratio: {grid['aspect_ratio']:.3f} (diff: {grid['aspect_diff']:.3f})")
    print(f"Score: {grid['score']:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal grid dimensions for PDF generation"
    )
    parser.add_argument(
        '--total-plots',
        type=int,
        required=True,
        help='Total number of plots to arrange'
    )
    parser.add_argument(
        '--paper-size',
        choices=list(PAPER_ASPECT_RATIOS.keys()),
        default='A2',
        help='Paper size (default: A2)'
    )
    parser.add_argument(
        '--min-plots-per-page',
        type=int,
        default=50,
        help='Minimum plots per page to consider (default: 50)'
    )
    parser.add_argument(
        '--max-plots-per-page',
        type=int,
        default=200,
        help='Maximum plots per page to consider (default: 200)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top configurations to show (default: 10)'
    )
    parser.add_argument(
        '--prefer-square',
        action='store_true',
        help='Prefer more square-like grids'
    )
    parser.add_argument(
        '--max-col-row-ratio',
        type=float,
        default=3.0,
        help='Maximum ratio of columns to rows (default: 3.0, e.g., 12 cols / 4 rows = 3.0)'
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Output results as CSV'
    )
    
    args = parser.parse_args()
    
    print(f"Finding optimal grid for {args.total_plots} plots on {args.paper_size} paper...")
    print(f"Considering {args.min_plots_per_page}-{args.max_plots_per_page} plots per page")
    print(f"Max column-to-row ratio: {args.max_col_row_ratio}:1")
    
    # Find optimal grids
    candidates = find_optimal_grids(
        args.total_plots,
        args.paper_size,
        args.min_plots_per_page,
        args.max_plots_per_page,
        args.prefer_square,
        args.max_col_row_ratio,
    )
    
    if not candidates:
        print("No valid grid configurations found!")
        return
    
    # Show top N
    top_n = min(args.top_n, len(candidates))
    
    if args.csv:
        # CSV output
        print("\nnrow,ncol,plots_per_page,num_pages,empty_last_page,last_page_util%,overall_util%,aspect_ratio,score")
        for i, grid in enumerate(candidates[:top_n]):
            print(f"{grid['nrow']},{grid['ncol']},{grid['plots_per_page']},{grid['num_pages']}," +
                  f"{grid['empty_cells_last_page']},{grid['last_page_utilization']*100:.1f}," +
                  f"{grid['utilization']*100:.1f},{grid['aspect_ratio']:.3f},{grid['score']:.1f}")
    else:
        # Human-readable output
        print(f"\nTop {top_n} configurations:")
        for i, grid in enumerate(candidates[:top_n], 1):
            print_grid_info(grid, i)
        
        # Recommendation
        best = candidates[0]
        print(f"\n{'='*70}")
        print("RECOMMENDATION:")
        print(f"  --nrow {best['nrow']} --ncol {best['ncol']}")
        print(f"\nThis will create {best['num_pages']} pages with only {best['empty_cells_last_page']} " +
              f"empty cells on the last page ({best['last_page_utilization']*100:.1f}% filled)")


if __name__ == '__main__':
    main()
