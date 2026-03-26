#!/usr/bin/env python3
"""
Merge multiple HDF5 files into a single output file by concatenating datasets along axis 0.

Features
- Recursively discovers all datasets in input files (by path)
- Validates compatibility: same dtype and same shape[1:] across files
- Concatenates along the first dimension (axis 0)
- Memory-efficient batched copying to handle very large datasets
- Optional include/exclude dataset filters
- Preserves dataset attributes (from the first file) and writes a small merge manifest to file attrs

Assumptions/Notes
- All input files should share the same dataset structure for the datasets being merged
- Datasets are concatenated only when present in every input (unless --skip-missing)
- Scalars or 0-D datasets are copied from the first file only

Examples
  # Merge three files into one
  ./.venv/bin/python scripts/merge_h5_files.py \
      --inputs data/a.h5 data/b.h5 data/c.h5 \
      --output data/merged.h5

  # Merge using a glob and include only select datasets
  ./.venv/bin/python scripts/merge_h5_files.py \
      --inputs data/run_*.h5 \
      --include /images /image_names /compositions \
      --output data/merged.h5

  # Allow files missing some datasets (those datasets will be skipped entirely)
  ./.venv/bin/python scripts/merge_h5_files.py \
      --inputs data/*.h5 --skip-missing --output data/merged.h5
"""
from __future__ import annotations
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import h5py
import numpy as np


def walk_datasets(h5: h5py.File | h5py.Group, base: str = "/") -> Iterator[Tuple[str, h5py.Dataset]]:
    """Yield (path, dataset) for all datasets under base recursively."""
    def _walk(g: h5py.Group, prefix: str):
        for key, item in g.items():
            path = prefix + key if prefix.endswith('/') else prefix + '/' + key
            if isinstance(item, h5py.Dataset):
                yield path, item
            elif isinstance(item, h5py.Group):
                yield from _walk(item, path)
    if isinstance(h5, h5py.File):
        yield from _walk(h5, '/')
    else:
        yield from _walk(h5, base)


def is_string_dtype(dtype) -> bool:
    try:
        return h5py.check_string_dtype(dtype) is not None or dtype.kind in ("S", "O", "U")
    except Exception:
        # Fallback if dtype has no kind (rare)
        return False


def ensure_groups(out_file: h5py.File, dataset_path: str) -> None:
    """Ensure all parent groups for dataset_path exist in out_file."""
    parts = dataset_path.strip('/').split('/')[:-1]
    cur = out_file
    built = ''
    for p in parts:
        built = built + '/' + p if built else '/' + p
        if p not in cur:
            cur = cur.create_group(p)
        else:
            cur = cur[p]


def copy_dataset_attrs(src: h5py.Dataset, dst: h5py.Dataset) -> None:
    for k, v in src.attrs.items():
        try:
            dst.attrs[k] = v
        except Exception:
            # Some attrs may not be directly writable (e.g., references). Skip silently.
            pass


def gather_schema(files: List[h5py.File], include: Optional[set[str]], exclude: Optional[set[str]], skip_missing: bool) -> Dict[str, Dict]:
    """Collect schema info for datasets to merge.

    Returns mapping: dataset_path -> {
        'dtype': dtype,
        'shape_tail': tuple,  # shape[1:]
        'lengths': [int, ...],  # per file
        'present': [bool, ...],
    }
    """
    schema: Dict[str, Dict] = {}
    # Build union of dataset paths across all files
    all_paths: set[str] = set()
    per_file_paths: List[set[str]] = []
    for f in files:
        paths = {p for p, _ in walk_datasets(f)}
        per_file_paths.append(paths)
        all_paths |= paths

    # Filter include/exclude
    if include:
        all_paths = {p for p in all_paths if p in include}
    if exclude:
        all_paths = {p for p in all_paths if p not in exclude}

    # Build schema entries
    for p in sorted(all_paths):
        present = [(p in paths) for paths in per_file_paths]
        if not skip_missing and not all(present):
            missing = [i for i, ok in enumerate(present) if not ok]
            raise ValueError(f"Dataset {p} missing in files at indices: {missing}")
        if not any(present):
            continue
        # Use first file where present as reference
        ref_idx = present.index(True)
        ref_ds = files[ref_idx][p]
        shape = ref_ds.shape
        if len(shape) == 0:
            # Scalar dataset -> will copy from first file only
            schema[p] = {
                'dtype': ref_ds.dtype,
                'shape_tail': (),
                'lengths': [0 if not ok else 1 for ok in present],
                'present': present,
                'ref_idx': ref_idx,
                'scalar': True,
            }
            continue
        # 1D or ND with axis-0 concatenation
        shape_tail = shape[1:]
        dtype = ref_ds.dtype
        # Validate across files where present
        for i, ok in enumerate(present):
            if not ok:
                continue
            ds_i = files[i][p]
            if len(ds_i.shape) != len(shape):
                raise ValueError(f"Rank mismatch for {p}: file {i} has shape {ds_i.shape}, ref {shape}")
            if ds_i.shape[1:] != shape_tail:
                raise ValueError(f"Shape tail mismatch for {p}: file {i} has shape {ds_i.shape}, ref tail {shape_tail}")
            if ds_i.dtype != dtype:
                # Allow compatible string dtypes with different encodings/lengths by treating as string
                if not (is_string_dtype(ds_i.dtype) and is_string_dtype(dtype)):
                    raise ValueError(f"Dtype mismatch for {p}: file {i} has {ds_i.dtype}, ref {dtype}")
        lengths = [0 if not ok else files[i][p].shape[0] for i, ok in enumerate(present)]
        schema[p] = {
            'dtype': dtype,
            'shape_tail': shape_tail,
            'lengths': lengths,
            'present': present,
            'ref_idx': ref_idx,
            'scalar': False,
        }
    return schema


def create_output_datasets(out: h5py.File, schema: Dict[str, Dict], files: List[h5py.File], compression: Optional[str], compression_opts: Optional[int]) -> Dict[str, h5py.Dataset]:
    """Create output datasets based on schema and return mapping path->dataset.

    If no compression is explicitly provided, this preserves the reference dataset's
    compression and chunking settings to avoid inflating output size.
    """
    out_dsets: Dict[str, h5py.Dataset] = {}
    for p, info in schema.items():
        ensure_groups(out, p)
        if info.get('scalar', False):
            # Copy scalar from first present file
            ref_idx = info.get('ref_idx', 0)
            ref_f = files[ref_idx]
            if p in ref_f:
                data = ref_f[p][()]  # scalar value
                d = out.create_dataset(p, data=data)
                copy_dataset_attrs(ref_f[p], d)
                out_dsets[p] = d
            continue
        total_len = sum(info['lengths'])
        shape_tail = tuple(info['shape_tail'])
        shape = (total_len,) + shape_tail
        dtype = info['dtype']
        # If string dtypes differ, prefer ref dtype
        if is_string_dtype(dtype):
            # Keep original dtype to avoid decode/encode overhead
            target_dtype = dtype
        else:
            target_dtype = dtype
        # Determine reference dataset to mirror storage settings when not overridden
        ref_idx = info.get('ref_idx', 0)
        ref_ds = files[ref_idx][p]
        # Decide compression/chunking
        use_compression = compression
        use_compression_opts = compression_opts
        use_chunks = None
        if is_string_dtype(target_dtype):
            # Avoid compression for variable-length/string datasets
            use_compression = None
            use_compression_opts = None
        if use_compression is None:
            # Mirror reference dataset's compression if available
            use_compression = ref_ds.compression
            use_compression_opts = ref_ds.compression_opts
        # Mirror chunks when compression is used or reference is chunked
        try:
            if ref_ds.chunks is not None:
                use_chunks = ref_ds.chunks
        except Exception:
            use_chunks = None

        # Create dataset with chosen storage params
        d = out.create_dataset(
            p,
            shape=shape,
            dtype=target_dtype,
            chunks=use_chunks,
            compression=use_compression,
            compression_opts=use_compression_opts,
        )
        # Copy attrs from reference (first file where present)
        copy_dataset_attrs(ref_ds, d)
        out_dsets[p] = d
    return out_dsets


def batched_copy(src: h5py.Dataset, dst: h5py.Dataset, dst_start: int, batch_elems: int) -> int:
    """Copy from src into dst starting at dst_start in batches of batch_elems along axis 0.

    Returns the number of elements copied.
    """
    n = src.shape[0]
    copied = 0
    pos = 0
    while pos < n:
        end = min(pos + batch_elems, n)
        dst[dst_start + pos: dst_start + end, ...] = src[pos:end, ...]
        copied += (end - pos)
        pos = end
    return copied


def estimate_batch_elems(ds: h5py.Dataset, target_bytes: int = 64 * 1024 * 1024) -> int:
    """Estimate number of elements (axis-0) per batch to keep around target_bytes in memory."""
    if len(ds.shape) == 0:
        return 1
    elems_per_item = int(np.prod(ds.shape[1:])) if len(ds.shape) > 1 else 1
    itemsize = 8 if is_string_dtype(ds.dtype) else int(ds.dtype.itemsize)
    bytes_per_item = max(1, elems_per_item * itemsize)
    batch = max(1, target_bytes // bytes_per_item)
    # Ensure not too small
    return int(max(1, min(batch, ds.shape[0] or 1)))


def merge_files(
    input_paths: List[str],
    output_path: str,
    include: Optional[List[str]],
    exclude: Optional[List[str]],
    skip_missing: bool,
    overwrite: bool,
    compression: Optional[str],
    compression_opts: Optional[int],
    target_bytes: int,
    rdcc_nbytes: Optional[int],
    rdcc_nslots: Optional[int],
    virtual: bool,
) -> None:
    inputs_expanded: List[str] = []
    for p in input_paths:
        hits = glob.glob(p)
        if hits:
            inputs_expanded.extend(sorted(hits))
        else:
            inputs_expanded.append(p)
    if len(inputs_expanded) < 2:
        raise SystemExit("Provide at least two input files to merge.")
    for p in inputs_expanded:
        if not os.path.exists(p):
            raise SystemExit(f"Input file not found: {p}")

    if os.path.exists(output_path):
        if not overwrite:
            raise SystemExit(f"Output file exists: {output_path} (use --overwrite to replace)")
        else:
            os.remove(output_path)

    # Resolve absolute paths for robust VDS references (prevents zero-fill if CWD changes)
    abs_inputs = [os.path.abspath(p) for p in inputs_expanded]

    # Open input files (read-only)
    files = [h5py.File(p, 'r') for p in inputs_expanded]
    try:
        inc_set = set(include) if include else None
        exc_set = set(exclude) if exclude else None
        schema = gather_schema(files, inc_set, exc_set, skip_missing=skip_missing)

        # File open cache tuning for faster IO
        out_open_kwargs = {'mode': 'w'}
        if rdcc_nbytes is not None:
            out_open_kwargs['rdcc_nbytes'] = int(rdcc_nbytes)
        if rdcc_nslots is not None:
            out_open_kwargs['rdcc_nslots'] = int(rdcc_nslots)

        with h5py.File(output_path, **out_open_kwargs) as out:
            # Merge manifest
            out.attrs['merged_from'] = json.dumps(abs_inputs)
            out.attrs['merge_tool'] = 'merge_h5_files.py'

            # Always perform a zero-copy virtual merge using VDS (Virtual Datasets)
            for p, info in schema.items():
                if info.get('scalar', False):
                    # Copy scalar value as-is
                    ref_idx = info.get('ref_idx', 0)
                    ref_f = files[ref_idx]
                    if p in ref_f:
                        data = ref_f[p][()]
                        d = out.create_dataset(p, data=data)
                        copy_dataset_attrs(ref_f[p], d)
                    continue
                total_len = sum(info['lengths'])
                shape_tail = tuple(info['shape_tail'])
                shape = (total_len,) + shape_tail
                dtype = info['dtype']
                layout = h5py.VirtualLayout(shape=shape, dtype=dtype)
                offset = 0
                for fi, f in enumerate(files):
                    if not info['present'][fi]:
                        continue
                    src_ds = f[p]
                    n = src_ds.shape[0]
                    if n == 0:
                        continue
                    # Use absolute source paths for robust resolution
                    vsrc = h5py.VirtualSource(abs_inputs[fi], p, shape=src_ds.shape)
                    layout[offset:offset + n, ...] = vsrc[0:n, ...]
                    offset += n
                d = out.create_virtual_dataset(p, layout)
                # Copy attrs from reference
                ref_idx = info.get('ref_idx', 0)
                copy_dataset_attrs(files[ref_idx][p], d)
            print(f"Created VDS (virtual) merged view of {len(inputs_expanded)} files -> {output_path}")
    finally:
        for f in files:
            try:
                f.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description='Merge multiple HDF5 files by concatenating datasets along axis 0.')
    ap.add_argument('--inputs', nargs='+', required=True, help='Input HDF5 files (supports globs)')
    ap.add_argument('--output', required=True, help='Output HDF5 file path')
    ap.add_argument('--include', nargs='*', help='Dataset paths to include (default: all)')
    ap.add_argument('--exclude', nargs='*', help='Dataset paths to exclude')
    ap.add_argument('--skip-missing', action='store_true', help='Skip datasets not present in all files (exclude them entirely)')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite output if it exists')
    ap.add_argument('--compression', choices=['gzip','lzf'], help='Compression for output datasets (optional)')
    ap.add_argument('--compression-opts', type=int, help='Compression level/options (e.g., 4 for gzip)')
    ap.add_argument('--target-bytes', type=int, default=64*1024*1024, help='Approximate bytes per copy batch (default: 64MB). Increase for speed if memory allows.')
    ap.add_argument('--rdcc-nbytes', type=int, help='Raw data chunk cache size in bytes for output file (e.g., 128*1024*1024)')
    ap.add_argument('--rdcc-nslots', type=int, help='Number of chunk slots in raw data chunk cache (power of two recommended)')
    ap.add_argument('--virtual', action='store_true', help='[Ignored] Virtual merge is always used now (VDS-only)')
    args = ap.parse_args()

    merge_files(
        input_paths=args.inputs,
        output_path=args.output,
        include=args.include,
        exclude=args.exclude,
        skip_missing=args.skip_missing,
        overwrite=args.overwrite,
        compression=args.compression,
        compression_opts=args.compression_opts,
        target_bytes=args.target_bytes,
        rdcc_nbytes=args.rdcc_nbytes,
        rdcc_nslots=args.rdcc_nslots,
        virtual=args.virtual,
    )


if __name__ == '__main__':
    main()
