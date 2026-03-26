#!/usr/bin/env python3
"""Compare two large JSON arrays of composition dicts and record differing indices.

Designed for very large files (>>50MB). Uses a streaming parser to avoid
loading entire arrays into memory.

Each file must contain a top-level JSON array; each element is expected to be
an object (dict) with string keys and numeric values.

Outputs a JSON summary with differing indices. Optionally writes a detailed
differences file with per-index objects if requested.

Example:
  python scripts/compare_compositions.py \
      --original data/compositions_for_gen.json \
      --fixed data/compositions_for_gen_fixed_dot.json \
      --output data/compositions_diff_indices.json \
      --details-file data/compositions_diff_details.json --max-details 1000

"""
from __future__ import annotations
import argparse
import json
import math
import os
from typing import Iterator, Dict, Any, Optional, List


def stream_json_array(path: str) -> Iterator[Any]:
    """Yield JSON objects from a file containing a single top-level JSON array.

    Streaming parser that reads incrementally and extracts complete JSON
    objects delimited by balanced braces at top level. Assumes array items
    are objects (dicts). Whitespace and commas between objects are ignored.
    """
    buf_size = 1 << 20  # 1 MiB
    with open(path, 'r', encoding='utf-8') as f:
        # Skip leading whitespace until '['
        while True:
            ch = f.read(1)
            if ch == '':
                raise ValueError(f"Unexpected EOF before '[' in {path}")
            if ch.isspace():
                continue
            if ch == '[':
                break
            raise ValueError(f"File {path} does not start with '[' (found {ch!r})")

        in_string = False
        escape = False
        depth = 0
        obj_chars: List[str] = []
        while True:
            chunk = f.read(buf_size)
            if chunk == '':  # EOF
                break
            for ch in chunk:
                if depth == 0 and not in_string:
                    if ch == ']':
                        # End of array
                        return
                    if ch in ' \n\r\t,' :
                        continue
                    if ch == '{':
                        depth = 1
                        obj_chars = ['{']
                        continue
                    raise ValueError(f"Unexpected character {ch!r} at top-level in array in {path}")
                else:
                    obj_chars.append(ch)
                    if in_string:
                        if escape:
                            escape = False
                        elif ch == '\\':
                            escape = True
                        elif ch == '"':
                            in_string = False
                    else:
                        if ch == '"':
                            in_string = True
                        elif ch == '{':
                            depth += 1
                        elif ch == '}':
                            depth -= 1
                            if depth == 0:
                                # Complete object
                                obj_text = ''.join(obj_chars)
                                try:
                                    yield json.loads(obj_text)
                                except Exception as e:
                                    raise ValueError(f"Error parsing JSON object: {e}\nObject snippet: {obj_text[:200]}...") from e
                                obj_chars = []
        if depth != 0:
            raise ValueError(f"Unbalanced braces in {path}")


def canonicalize(comp: Dict[str, Any], rounding: Optional[int]) -> Dict[str, float]:
    """Return a canonical mapping of element->rounded float for comparison."""
    out = {}
    for k, v in comp.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if rounding is not None:
            fv = round(fv, rounding)
        out[k] = fv
    return out


def comps_equal(a: Dict[str, float], b: Dict[str, float], rel_tol: float, abs_tol: float) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        av = a[k]
        bv = b[k]
        if not math.isclose(av, bv, rel_tol=rel_tol, abs_tol=abs_tol):
            return False
    return True


def compare_files(original_path: str, fixed_path: str, rounding: Optional[int], rel_tol: float, abs_tol: float,
                  max_details: Optional[int], details_path: Optional[str]) -> dict:
    orig_iter = stream_json_array(original_path)
    fix_iter = stream_json_array(fixed_path)

    differing_indices: List[int] = []
    details: List[dict] = []

    total_compared = 0
    while True:
        try:
            o = next(orig_iter)
        except StopIteration:
            o = None
        try:
            f = next(fix_iter)
        except StopIteration:
            f = None

        if o is None or f is None:
            break

        co = canonicalize(o, rounding)
        cf = canonicalize(f, rounding)
        if not comps_equal(co, cf, rel_tol=rel_tol, abs_tol=abs_tol):
            differing_indices.append(total_compared)
            if details_path and (max_details is None or len(details) < max_details):
                details.append({
                    'index': total_compared,
                    'original': o,
                    'fixed': f
                })
        total_compared += 1
        if total_compared % 100000 == 0:
            print(f"Compared {total_compared:,} entries... diff so far: {len(differing_indices):,}")

    # Count remaining extras
    orig_remaining = sum(1 for _ in orig_iter)
    fix_remaining = sum(1 for _ in fix_iter)

    summary = {
        'original_file': original_path,
        'fixed_file': fixed_path,
        'rounding': rounding,
        'rel_tol': rel_tol,
        'abs_tol': abs_tol,
        'total_original': total_compared + orig_remaining,
        'total_fixed': total_compared + fix_remaining,
        'compared_prefix': total_compared,
        'differing_indices': differing_indices,
        'differing_count': len(differing_indices),
        'original_only_count': orig_remaining,
        'fixed_only_count': fix_remaining,
    }

    if details_path:
        with open(details_path, 'w', encoding='utf-8') as df:
            json.dump({'differences': details, 'truncated': (max_details is not None and len(differing_indices) > max_details)}, df, indent=2)
            print(f"Wrote details for {len(details)} differing entries to {details_path}")

    return summary


def main():
    ap = argparse.ArgumentParser(description='Compare two large JSON arrays of composition dicts.')
    ap.add_argument('--original', required=True, help='Path to original compositions JSON')
    ap.add_argument('--fixed', required=True, help='Path to fixed compositions JSON')
    ap.add_argument('--output', required=True, help='Path to write summary diff JSON')
    ap.add_argument('--details-file', help='Optional path to write detailed differing entries')
    ap.add_argument('--max-details', type=int, default=500, help='Limit number of detailed entries (default 500, use -1 for no limit)')
    ap.add_argument('--rounding', type=int, default=6, help='Round floats to this many decimals before comparison (default 6, use -1 for none)')
    ap.add_argument('--rel-tol', type=float, default=1e-9, help='Relative tolerance for float comparison (post rounding)')
    ap.add_argument('--abs-tol', type=float, default=1e-12, help='Absolute tolerance for float comparison (post rounding)')
    args = ap.parse_args()

    rounding = None if args.rounding < 0 else args.rounding
    max_details = None if args.max_details is not None and args.max_details < 0 else args.max_details

    if not os.path.exists(args.original):
        raise SystemExit(f"Original file not found: {args.original}")
    if not os.path.exists(args.fixed):
        raise SystemExit(f"Fixed file not found: {args.fixed}")

    summary = compare_files(
        original_path=args.original,
        fixed_path=args.fixed,
        rounding=rounding,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
        max_details=max_details,
        details_path=args.details_file,
    )

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {args.output}")
    print(f"Differing indices: {summary['differing_count']:,}")


if __name__ == '__main__':
    main()
