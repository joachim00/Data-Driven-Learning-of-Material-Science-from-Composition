#!/usr/bin/env python3
"""Filter HDF5 file by compositions.

This script reads an input HDF5 file with a `compositions` dataset (vlen JSON strings or similar)
and copies all datasets into a new output HDF5 file but omits entries where any composition
element string contains a '.' character. It processes in batches to avoid high memory usage
and attempts to preserve per-dataset compression/chunking for the output file where possible.

Usage:
    python scripts/filter_h5_by_composition.py --input path/to/input.h5 --output path/to/cleaned.h5

The script will default to scanning the dataset named 'compositions' and will assume
concatenation axis is 0 for all non-scalar datasets.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Iterable, List, Tuple

import h5py
import numpy as np


def format_bytes(n: int) -> str:
    """Return human-readable file size."""
    if n is None:
        return 'N/A'
    step = 1024.0
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    i = 0
    x = float(n)
    while x >= step and i < len(units) - 1:
        x /= step
        i += 1
    return f"{x:.2f} {units[i]}"


def iter_batches(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < total:
        j = min(total, i + batch_size)
        yield i, j
        i = j


def read_composition_record(raw) -> str:
    # Handle bytes and str vlen types
    if isinstance(raw, bytes):
        try:
            return raw.decode('utf-8')
        except Exception:
            return raw.decode('latin1')
    return str(raw)


def composition_has_dot(comp_str: str) -> bool:
    """Return True if any element in the composition contains a '.' character.

    comp_str is expected to be a JSON object mapping element symbols to numbers, e.g. '{"Fe": 0.5, "Zn": 0.5}'.
    If parsing fails, we conservatively treat the raw string and look for '.' in tokens.
    """
    try:
        parsed = json.loads(comp_str)
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                # treat keys and stringified values
                if '.' in str(k):
                    return True
                if isinstance(v, str) and '.' in v:
                    return True
            return False
        # fallback: if parsed isn't a dict, scan string
    except Exception:
        pass
    # last-resort: look for '.' in the raw string tokens separated by non-alphanum
    return '.' in comp_str


def filter_h5(input_path: str, output_path: str, compositions_name: str = 'compositions', batch_size: int = 4096,
              dry_run: bool = False, sample_n: int = 10, dry_run_output: str | None = None,
              max_copy_chunk: int = 4096, disable_compression: bool = False,
              show_progress: bool = False):
    assert os.path.exists(input_path), f"Input file does not exist: {input_path}"
    with h5py.File(input_path, 'r') as inf:
        if compositions_name not in inf:
            raise KeyError(f"Dataset '{compositions_name}' not found in input file")
        comps = inf[compositions_name]
        total = comps.shape[0]

        # First pass: find indices to keep
        keep_mask = np.zeros((total,), dtype=bool)
        for a, b in iter_batches(total, batch_size):
            raw_slice = comps[a:b]
            for i, raw in enumerate(raw_slice, start=a):
                s = read_composition_record(raw)
                if not composition_has_dot(s):
                    keep_mask[i] = True

        keep_indices = np.nonzero(keep_mask)[0]
        kept = keep_indices.shape[0]
        removed_indices = np.nonzero(~keep_mask)[0]
        print(f"Total records: {total}, kept: {kept}, removed: {total-kept}")

        # If dry-run / verification mode, print example entries and exit without writing
        if dry_run:
            n_show = max(0, int(sample_n))
            # If requested, write all entries to a TSV with their status
            if dry_run_output:
                with open(dry_run_output, 'w', encoding='utf-8') as outf_tsv:
                    outf_tsv.write('index\tstatus\tcomposition\n')
                    for i in range(total):
                        try:
                            raw = comps[i]
                            s = read_composition_record(raw)
                        except Exception as e:
                            s = f"<read error: {e}>"
                        status = 'kept' if keep_mask[i] else 'removed'
                        # sanitize tabs/newlines in composition
                        s_clean = s.replace('\t', ' ').replace('\n', ' ')
                        outf_tsv.write(f"{i}\t{status}\t{s_clean}\n")

            print(f"\nSample of up to {n_show} removed records (index : composition):")
            for idx in removed_indices[:n_show]:
                try:
                    raw = comps[idx]
                    s = read_composition_record(raw)
                except Exception as e:
                    s = f"<read error: {e}>"
                print(f"  {idx} : {s}")

            print(f"\nSample of up to {n_show} kept records (index : composition):")
            for idx in keep_indices[:n_show]:
                try:
                    raw = comps[idx]
                    s = read_composition_record(raw)
                except Exception as e:
                    s = f"<read error: {e}>"
                print(f"  {idx} : {s}")

            print("\nDry-run complete. No output file written.")
            return

        # Create output file and copy datasets in batches according to keep_indices
        with h5py.File(output_path, 'w') as outf:
            # copy scalar attrs and simple scalar datasets
            for name, ds in inf.items():
                if ds.shape == ():
                    # scalar dataset: copy directly
                    outf.create_dataset(name, data=ds[()])

            # Now handle non-scalar datasets by creating output datasets with the right shape
            for name, ds in inf.items():
                if ds.shape == ():
                    continue
                src_shape = ds.shape
                new_shape = (kept,) + src_shape[1:]

                # Determine dtype and storage params
                dtype = ds.dtype
                compression = ds.compression
                compression_opts = ds.compression_opts
                chunks = ds.chunks

                # For variable-length string/object dtypes, avoid compression choices that h5py disallows
                # Respect the disable_compression flag: if True, don't pass compression options to create_dataset
                if dtype.kind in ('O',) or getattr(dtype, 'metadata', None) is not None:
                    # create vlen string dtype if source is vlen bytes/str
                    out_ds = outf.create_dataset(name, shape=new_shape, dtype=dtype)
                else:
                    if disable_compression:
                        # create without compression to speed up writes
                        out_ds = outf.create_dataset(name, shape=new_shape, dtype=dtype, chunks=chunks)
                    else:
                        out_ds = outf.create_dataset(name, shape=new_shape, dtype=dtype, compression=compression,
                                                     compression_opts=compression_opts, chunks=chunks)

                # Copy data using contiguous runs and sub-chunks to avoid large fancy-index allocations
                def iter_runs(idxs_arr: np.ndarray):
                    if idxs_arr.size == 0:
                        return
                    s = int(idxs_arr[0])
                    e = s + 1
                    for v in idxs_arr[1:]:
                        v = int(v)
                        if v == e:
                            e += 1
                        else:
                            yield s, e
                            s = v
                            e = v + 1
                    yield s, e

                write_ptr = 0
                # Prepare progress display if requested
                if show_progress:
                    try:
                        from tqdm import tqdm
                        use_tqdm = True
                    except Exception:
                        use_tqdm = False

                total_to_copy = keep_indices.shape[0]
                copied = 0

                # We'll process kept indices in windows of at most max_copy_chunk items to
                # reduce the number of distinct HDF5 reads. For each window we read one
                # contiguous slab [s:e] that covers the window, then select the rows we need.
                window = int(max_copy_chunk)
                if window <= 0:
                    window = 4096

                # Try to align window to chunk size for better IO, if available
                try:
                    src_chunk0 = ds.chunks[0] if ds.chunks is not None else None
                except Exception:
                    src_chunk0 = None
                if src_chunk0:
                    mult = (window + src_chunk0 - 1) // src_chunk0
                    window = max(window, mult * src_chunk0)

                # Iterate over keep_indices in windows (by position, not by value)
                n_kept = total_to_copy
                pos = 0
                while pos < n_kept:
                    batch_idxs = keep_indices[pos:pos + window]
                    s = int(batch_idxs[0])
                    e = int(batch_idxs[-1]) + 1
                    try:
                        slab = ds[s:e]
                    except Exception:
                        # fallback to slow per-item reads if slab read fails
                        for idx in batch_idxs:
                            data = ds[int(idx)]
                            out_ds[write_ptr:write_ptr + data.shape[0]] = data
                            write_ptr += data.shape[0]
                            copied += 1
                            if show_progress:
                                if 'use_tqdm' in locals() and use_tqdm:
                                    if not hasattr(filter_h5, '_tqdm_cache'):
                                        filter_h5._tqdm_cache = {}
                                    key = name
                                    if key not in filter_h5._tqdm_cache:
                                        filter_h5._tqdm_cache[key] = tqdm(total=total_to_copy, desc=f"{name}", unit='it')
                                    filter_h5._tqdm_cache[key].update(1)
                                    if copied >= total_to_copy:
                                        filter_h5._tqdm_cache[key].close()
                                        del filter_h5._tqdm_cache[key]
                                else:
                                    pct = int((copied / total_to_copy) * 100) if total_to_copy else 100
                                    print(f"{name}: {copied}/{total_to_copy} ({pct}%)", end='\r')
                                    if copied >= total_to_copy:
                                        print()
                        pos += window
                        continue

                    # compute offsets of desired rows inside the slab
                    offsets = (batch_idxs - s).astype(int)
                    # slice out the needed rows
                    try:
                        data_batch = slab[offsets]
                    except Exception:
                        # if direct advanced indexing fails, fallback to list comprehension
                        data_batch = np.stack([slab[int(off)] for off in offsets], axis=0)

                    out_ds[write_ptr:write_ptr + data_batch.shape[0]] = data_batch
                    write_ptr += data_batch.shape[0]
                    copied += data_batch.shape[0]

                    # update progress
                    if show_progress:
                        if 'use_tqdm' in locals() and use_tqdm:
                            if not hasattr(filter_h5, '_tqdm_cache'):
                                filter_h5._tqdm_cache = {}
                            key = name
                            if key not in filter_h5._tqdm_cache:
                                filter_h5._tqdm_cache[key] = tqdm(total=total_to_copy, desc=f"{name}", unit='it')
                            filter_h5._tqdm_cache[key].update(data_batch.shape[0])
                            if copied >= total_to_copy:
                                filter_h5._tqdm_cache[key].close()
                                del filter_h5._tqdm_cache[key]
                        else:
                            pct = int((copied / total_to_copy) * 100) if total_to_copy else 100
                            print(f"{name}: {copied}/{total_to_copy} ({pct}%)", end='\r')
                            if copied >= total_to_copy:
                                print()

                    pos += window

            # copy file-level attrs
            for k, v in inf.attrs.items():
                outf.attrs[k] = v


def main():
    p = argparse.ArgumentParser(description="Filter HDF5 by composition entries containing '.'")
    p.add_argument('--input', '-i', required=True, help='Input HDF5 path')
    p.add_argument('--output', '-o', required=False, help='Output HDF5 path (required for consolidated non-dry-run mode)')
    p.add_argument('--compositions-name', default='compositions', help='Dataset name for compositions')
    p.add_argument('--batch-size', type=int, default=4096, help='Batch size when scanning/copying')
    p.add_argument('--dry-run', action='store_true', help='Scan and show examples but do not write output')
    p.add_argument('--sample-n', type=int, default=10, help='Number of example removed/kept records to show in dry-run')
    p.add_argument('--dry-run-output', help='If set, write all composition entries and their keep/remove status to this TSV file (index<TAB>status<TAB>composition)')
    p.add_argument('--max-copy-chunk', type=int, default=4096, help='Maximum contiguous copy chunk size when writing cleaned files (items)')
    p.add_argument('--disable-compression', action='store_true', help='Disable compression on output datasets to speed up writing')
    p.add_argument('--mode', choices=['consolidated', 'per-shard', 'inplace', 'truncate'], default='consolidated',
                   help='Operation mode: consolidated (single input -> single output), per-shard (process each file in input dir), inplace (rewrite shards in-place with backups), truncate (in-place shrink when removals are a trailing suffix)')
    p.add_argument('--shard-pattern', default='*.h5', help='Glob pattern to match shard files when using per-shard or inplace')
    p.add_argument('--output-dir', help='Output directory for per-shard mode (defaults to same dir as each shard)')
    p.add_argument('--backup-dir', help='Backup directory for inplace mode (defaults to <shard_dir>/backups)')
    p.add_argument('--no-backup', action='store_true', help='(DANGEROUS) In inplace mode, overwrite original shards without creating backups')
    args = p.parse_args()

    # Validate output requirement: for consolidated non-dry-run we need an output
    mode = args.mode
    if mode == 'consolidated' and not args.dry_run and not args.output:
        p.error("--output/-o is required in 'consolidated' mode when not using --dry-run")

    # Choose mode
    if mode == 'consolidated':
        # provide a placeholder output when dry-run is used and no output was given
        out_path = args.output or '/tmp/unused.h5'
        filter_h5(args.input, out_path, compositions_name=args.compositions_name,
                  batch_size=args.batch_size, dry_run=args.dry_run, sample_n=args.sample_n,
                  dry_run_output=args.dry_run_output if hasattr(args, 'dry_run_output') else None,
                  max_copy_chunk=args.max_copy_chunk, disable_compression=args.disable_compression)
        return

    # For per-shard and inplace modes, input must be a directory
    import glob
    inp = args.input
    if not os.path.isdir(inp):
        raise ValueError(f"For mode '{mode}', --input must be a directory containing shard files (got: {inp})")

    pattern = os.path.join(inp, args.shard_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return

    # Prepare backup dir for inplace mode
    backup_dir = args.backup_dir
    if mode == 'inplace':
        if backup_dir is None:
            backup_dir = os.path.join(inp, 'backups')
        os.makedirs(backup_dir, exist_ok=True)

    for f in files:
        fname = os.path.basename(f)
        fdir = os.path.dirname(f)
        try:
            orig_size = os.path.getsize(f)
        except Exception:
            orig_size = None
        if args.output_dir:
            out_dir = args.output_dir
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = fdir

        out_name = os.path.splitext(fname)[0] + '_cleaned.h5'
        out_path = os.path.join(out_dir, out_name)

        print(f"Processing shard: {f} -> {out_path} (mode={mode})")
    # Dry-run: just show examples for this shard
        if args.dry_run:
            # determine per-shard dry-run output path
            if args.dry_run_output:
                # if user provided a directory, put TSV there; if provided a file, append shard name
                if os.path.isdir(args.dry_run_output):
                    per_shard_tsv = os.path.join(args.dry_run_output, os.path.splitext(fname)[0] + '_compositions.tsv')
                else:
                    per_shard_tsv = args.dry_run_output + '.' + os.path.splitext(fname)[0] + '.tsv'
            else:
                per_shard_tsv = os.path.join(out_dir, os.path.splitext(fname)[0] + '_compositions.tsv')

            filter_h5(f, out_path, compositions_name=args.compositions_name,
                      batch_size=args.batch_size, dry_run=True, sample_n=args.sample_n,
                      dry_run_output=per_shard_tsv, show_progress=True)
            continue

        if mode == 'truncate':
            # Attempt to truncate the datasets in-place if removed entries are only a trailing suffix
            print(f"Attempting in-place truncate for shard: {f}")
            with h5py.File(f, 'r+') as fh:
                if args.compositions_name not in fh:
                    print(f"  compositions dataset '{args.compositions_name}' not found, skipping")
                    continue
                comps = fh[args.compositions_name]
                total = comps.shape[0]
                keep_mask = np.zeros((total,), dtype=bool)
                for a, b in iter_batches(total, args.batch_size):
                    raw_slice = comps[a:b]
                    for i, raw in enumerate(raw_slice, start=a):
                        s = read_composition_record(raw)
                        if not composition_has_dot(s):
                            keep_mask[i] = True

                keep_indices = np.nonzero(keep_mask)[0]
                kept = keep_indices.shape[0]
                if kept == 0:
                    new_len = 0
                else:
                    last_kept = int(keep_indices.max())
                    # verify kept indices are contiguous from 0..last_kept
                    if kept != (last_kept + 1):
                        print(f"  Cannot truncate shard {f}: kept indices are not contiguous from 0..{last_kept} (kept count {kept})")
                        continue
                    new_len = last_kept + 1

                print(f"  total={total}, kept={kept}, new_len={new_len}")

                # Now resize all non-scalar datasets along axis 0 to new_len
                for name, ds in fh.items():
                    if ds.shape == ():
                        continue
                    # check whether dataset is resizable
                    maxshape = ds.maxshape
                    if maxshape is not None and maxshape[0] is not None and maxshape[0] < new_len:
                        print(f"  Cannot resize dataset {name}: maxshape {maxshape} smaller than new_len {new_len}")
                        raise RuntimeError(f"Dataset {name} not resizable to new length {new_len}")
                    # perform resize
                    try:
                        ds.resize((new_len,) + ds.shape[1:])
                        print(f"  Resized dataset {name} -> {ds.shape}")
                    except Exception as e:
                        print(f"  Failed to resize dataset {name}: {e}")
                        raise

            print(f"Truncate complete for shard: {f}")
            continue

        # Non-dry-run: write cleaned shard to temporary path then move depending on mode
        tmp_out = out_path + '.tmp'
        # If inplace and no-backup is requested, favor a faster write path: disable compression
        # and increase batch/chunk sizes to speed up writes. This sacrifices disk space and
        # possibly larger temporary files but avoids creating a backup copy.
        fast_mode = False
        fast_batch = args.batch_size
        fast_copy_chunk = args.max_copy_chunk
        fast_disable_compression = args.disable_compression
        if args.mode == 'inplace' and args.no_backup:
            fast_mode = True
            # sensible faster defaults; these can be tuned
            fast_batch = max(args.batch_size, 65536)
            fast_copy_chunk = max(args.max_copy_chunk, 65536)
            # Respect user's disable_compression flag; don't force it on by default to avoid larger outputs
            fast_disable_compression = args.disable_compression
            if fast_disable_compression:
                print(f"FAST MODE: --no-backup detected; disable_compression=True, batch_size={fast_batch}, max_copy_chunk={fast_copy_chunk}")
            else:
                print(f"FAST MODE: --no-backup detected; running with compression enabled, batch_size={fast_batch}, max_copy_chunk={fast_copy_chunk}")

        filter_h5(f, tmp_out, compositions_name=args.compositions_name,
                  batch_size=fast_batch, dry_run=False, sample_n=0,
                  max_copy_chunk=fast_copy_chunk, disable_compression=fast_disable_compression,
                  show_progress=True)

        if mode == 'per-shard':
            # Already wrote tmp_out to out_path (tmp name), move into place
            os.replace(tmp_out, out_path)
            print(f"Wrote cleaned shard to: {out_path}")
            # report sizes
            try:
                new_size = os.path.getsize(out_path)
            except Exception:
                new_size = None
            if orig_size is not None and new_size is not None:
                delta = new_size - orig_size
                pct = (delta / orig_size) * 100 if orig_size else 0.0
                print(f"Size: original={format_bytes(orig_size)} ({orig_size} bytes), cleaned={format_bytes(new_size)} ({new_size} bytes), delta={format_bytes(delta)} ({delta} bytes) ({pct:.2f}%)")
        elif mode == 'inplace':
            # Move original to backup and replace with cleaned, or overwrite without backup if requested
            if args.no_backup:
                # safety: ensure there's enough free space in destination dir to place temp file
                dest_dir = os.path.dirname(f)
                try:
                    tmp_size = os.path.getsize(tmp_out)
                except Exception:
                    tmp_size = None
                if tmp_size is not None:
                    free = shutil.disk_usage(dest_dir).free
                    if free < tmp_size:
                        raise RuntimeError(f"Not enough free space to overwrite {f} without backup: need ~{tmp_size} bytes free, have {free}")
                print("WARNING: --no-backup specified: original will be overwritten WITHOUT backup")
                os.replace(tmp_out, f)
                print(f"Replaced original shard with cleaned data (no backup): {f}")
                # report sizes
                try:
                    final_size = os.path.getsize(f)
                except Exception:
                    final_size = None
                if orig_size is not None and final_size is not None:
                    delta = final_size - orig_size
                    pct = (delta / orig_size) * 100 if orig_size else 0.0
                    print(f"Size: original={format_bytes(orig_size)} ({orig_size} bytes), cleaned={format_bytes(final_size)} ({final_size} bytes), delta={format_bytes(delta)} ({delta} bytes) ({pct:.2f}%)")
            else:
                backup_name = os.path.join(backup_dir, fname + '.backup')
                print(f"Backing up original shard to: {backup_name}")
                os.replace(f, backup_name)
                os.replace(tmp_out, f)
                print(f"Replaced original shard with cleaned data: {f}")
                # report sizes
                try:
                    final_size = os.path.getsize(f)
                except Exception:
                    final_size = None
                if orig_size is not None and final_size is not None:
                    delta = final_size - orig_size
                    pct = (delta / orig_size) * 100 if orig_size else 0.0
                    print(f"Size: original={format_bytes(orig_size)} ({orig_size} bytes), cleaned={format_bytes(final_size)} ({final_size} bytes), delta={format_bytes(delta)} ({delta} bytes) ({pct:.2f}%)")



if __name__ == '__main__':
    main()
