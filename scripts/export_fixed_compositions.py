#!/usr/bin/env python3
"""Export fixed compositions to a JSON file (streaming, memory efficient).

Supports two modes:
  1) all        -> write all compositions from the fixed file
  2) differing  -> write only compositions at indices listed in a diff summary
                   produced by scripts/compare_compositions.py

Examples:
  # Write all fixed compositions (pretty-printed):
  ./.venv/bin/python scripts/export_fixed_compositions.py \
      --fixed data/compositions_for_gen_fixed_dot.json \
      --output data/compositions_fixed_only.json \
      --mode all --indent 2

  # Write only the differing entries from the fixed file:
  ./.venv/bin/python scripts/export_fixed_compositions.py \
      --fixed data/compositions_for_gen_fixed_dot.json \
      --diff-summary data/compositions_diff_indices.json \
      --output data/compositions_fixed_differences.json \
      --mode differing
"""
from __future__ import annotations
import argparse
import json
from typing import Iterator, Any, List, Set


def stream_json_array(path: str) -> Iterator[Any]:
    """Yield JSON objects from a file containing a single top-level JSON array."""
    buf_size = 1 << 20  # 1 MiB
    with open(path, 'r', encoding='utf-8') as f:
        # Skip to '['
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
            if chunk == '':
                break
            for ch in chunk:
                if depth == 0 and not in_string:
                    if ch == ']':
                        return
                    if ch in ' \n\r\t,':
                        continue
                    if ch == '{':
                        depth = 1
                        obj_chars = ['{']
                        continue
                    raise ValueError(f"Unexpected char {ch!r} at top-level in {path}")
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
                                obj_text = ''.join(obj_chars)
                                yield json.loads(obj_text)
                                obj_chars = []
        if depth != 0:
            raise ValueError(f"Unbalanced braces in {path}")


def write_json_array_stream(output_path: str, items: Iterator[Any], indent: int | None = None) -> None:
    """Write items as a single JSON array to output_path, streaming."""
    with open(output_path, 'w', encoding='utf-8') as out:
        if indent and indent > 0:
            # Pretty-printed array: write '[' and newline, then items with commas
            out.write('[\n')
            first = True
            for obj in items:
                if not first:
                    out.write(',\n')
                out.write(json.dumps(obj, ensure_ascii=False, indent=indent))
                first = False
            out.write('\n]\n')
        else:
            # Minified
            out.write('[')
            first = True
            for obj in items:
                if not first:
                    out.write(',')
                out.write(json.dumps(obj, ensure_ascii=False, separators=(',', ':')))
                first = False
            out.write(']')


def export_all(fixed_path: str, output_path: str, indent: int | None) -> None:
    write_json_array_stream(output_path, stream_json_array(fixed_path), indent=indent)


def export_differing(fixed_path: str, diff_summary_path: str, output_path: str, indent: int | None) -> None:
    with open(diff_summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    indices: List[int] = summary.get('differing_indices', [])
    idx_set: Set[int] = set(indices)

    def gen():
        for i, obj in enumerate(stream_json_array(fixed_path)):
            if i in idx_set:
                yield obj

    write_json_array_stream(output_path, gen(), indent=indent)


def main():
    ap = argparse.ArgumentParser(description='Export fixed compositions to JSON (all or only differing).')
    ap.add_argument('--fixed', required=True, help='Path to fixed compositions JSON')
    ap.add_argument('--output', required=True, help='Path to output JSON file')
    ap.add_argument('--mode', choices=['all', 'differing'], default='all', help='What to export (default: all)')
    ap.add_argument('--diff-summary', help='Path to diff summary JSON (required if mode=differing)')
    ap.add_argument('--indent', type=int, default=0, help='Pretty print with this indent (0 for minified)')
    args = ap.parse_args()

    indent = None if args.indent is None or args.indent <= 0 else args.indent

    if args.mode == 'all':
        export_all(args.fixed, args.output, indent)
    else:
        if not args.diff_summary:
            raise SystemExit('--diff-summary is required when --mode differing')
        export_differing(args.fixed, args.diff_summary, args.output, indent)

    print(f"Wrote {args.mode} compositions from {args.fixed} -> {args.output}")


if __name__ == '__main__':
    main()
