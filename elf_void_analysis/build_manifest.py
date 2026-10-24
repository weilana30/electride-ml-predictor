#!/usr/bin/env python3
"""Index ELFNet POSCAR/ELFCAR pairs for the atom-and-void GNN dataset.

This is a metadata pass only: it does not modify or copy source files. By default
it hashes source files and writes a Parquet manifest using pyarrow.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ints(line: str) -> list[int] | None:
    fields = line.split()
    if not fields or not all(re.fullmatch(r"\d+", token) for token in fields):
        return None
    return [int(token) for token in fields]


def parse_poscar(path: Path) -> tuple[int, list[str]]:
    """Return atom count and element symbols for standard VASP4/5/6 POSCARs."""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        lines = stream.readlines()
    if len(lines) < 7:
        raise ValueError("POSCAR has fewer than seven lines")
    # POSCAR line 6 is either symbols (VASP 5+) or counts (VASP 4).
    line6 = lines[5].split()
    counts = _ints(lines[6])
    if counts is None:
        # VASP4: line 6 contains counts and the title may carry no species.
        counts = _ints(lines[5])
        symbols = []
    else:
        symbols = line6
    if counts is None or not counts or any(n < 0 for n in counts):
        raise ValueError("could not parse POSCAR element counts")
    if symbols and len(symbols) != len(counts):
        raise ValueError("POSCAR symbol and count lengths differ")
    return sum(counts), symbols


def parse_elfcar_grid(path: Path, n_atoms: int) -> tuple[int, int, int]:
    """Read the first volumetric grid dimensions after the ELFCAR structure."""
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        # VASP4 has counts on line 6 and no species line; VASP5/6 has a
        # symbols line then counts on line 7. The counts-line test distinguishes
        # the two without mistaking an integer-looking first coordinate for it.
        prefix = list(itertools.islice(stream, 6))
        if len(prefix) < 6:
            raise ValueError("ELFCAR is too short to contain a POSCAR header")
        if _ints(prefix[5]) is not None:
            mode = stream.readline()  # VASP4 coordinate mode
        else:
            counts_line = stream.readline()  # VASP5/6 atom counts
            if _ints(counts_line) is None:
                raise ValueError("could not locate ELFCAR atom-count line")
            mode = stream.readline()
        if not mode:
            raise ValueError("missing ELFCAR coordinate mode")
        if mode.strip().lower().startswith("s"):
            mode = stream.readline()
            if not mode:
                raise ValueError("missing ELFCAR coordinate mode after Selective dynamics")
        for _ in range(n_atoms):
            if not stream.readline():
                raise ValueError("ELFCAR ended before all atom coordinates")
        grid_line = stream.readline()
        while grid_line and not grid_line.strip():
            grid_line = stream.readline()
    if not grid_line:
        raise ValueError("missing ELFCAR grid dimensions")
    dims = _ints(grid_line)
    if dims is None or len(dims) != 3 or any(n <= 0 for n in dims):
        raise ValueError("invalid ELFCAR grid dimensions")
    return dims[0], dims[1], dims[2]


def discover_samples(source_root: Path) -> list[Path]:
    # Only leaf sample directories containing one or both expected files count.
    dirs = {p.parent for p in source_root.rglob("POSCAR")}
    dirs.update(p.parent for p in source_root.rglob("ELFCAR"))
    return sorted(dirs, key=lambda p: p.relative_to(source_root).as_posix())


def make_record(sample_dir: Path, source_root: Path, hash_files: bool) -> dict[str, Any]:
    rel_dir = sample_dir.relative_to(source_root).as_posix()
    parts = Path(rel_dir).parts
    category = parts[0] if len(parts) >= 1 else None
    # Group the same parent phase across top-level categories to prevent leakage.
    phase = parts[1] if len(parts) >= 2 else (parts[0] if parts else "unknown")
    family_key = phase.strip().casefold()
    poscar = sample_dir / "POSCAR"
    elfcar = sample_dir / "ELFCAR"
    record: dict[str, Any] = {
        "sample_id": stable_id("elf", rel_dir),
        "family_id": stable_id("family", family_key),
        "structure_family_id": stable_id("family", family_key),
        "category": category,
        "phase_id": phase,
        "source_relative_dir": rel_dir,
        "poscar_relative_path": poscar.relative_to(source_root).as_posix() if poscar.is_file() else None,
        "elfcar_relative_path": elfcar.relative_to(source_root).as_posix() if elfcar.is_file() else None,
        "poscar_sha256": None,
        "elfcar_sha256": None,
        "atom_count": None,
        "element_symbols": None,
        "grid_shape_x": None,
        "grid_shape_y": None,
        "grid_shape_z": None,
        "grid_shape": None,
        "processing_status": "pending",
        "failure_reason": None,
    }
    reasons: list[str] = []
    if not poscar.is_file():
        reasons.append("missing_poscar")
    if not elfcar.is_file():
        reasons.append("missing_elfcar")
    if reasons:
        record["processing_status"] = "failed"
        record["failure_reason"] = ";".join(reasons)
        return record
    try:
        atom_count, symbols = parse_poscar(poscar)
        record["atom_count"] = atom_count
        record["element_symbols"] = symbols
    except (OSError, ValueError) as exc:
        reasons.append(f"invalid_poscar:{exc}")
    if not reasons:
        try:
            gx, gy, gz = parse_elfcar_grid(elfcar, record["atom_count"])
            record.update(grid_shape_x=gx, grid_shape_y=gy, grid_shape_z=gz)
            record["grid_shape"] = [gx, gy, gz]
        except (OSError, ValueError) as exc:
            reasons.append(f"invalid_elfcar:{exc}")
    if hash_files:
        try:
            record["poscar_sha256"] = sha256_file(poscar)
            record["elfcar_sha256"] = sha256_file(elfcar)
        except OSError as exc:
            reasons.append(f"hash_error:{exc}")
    record["processing_status"] = "ready" if not reasons else "failed"
    record["failure_reason"] = ";".join(reasons) if reasons else None
    return record


def build_manifest(source_root: str | Path, hash_files: bool = True,
                   limit: int | None = None) -> list[dict[str, Any]]:
    """Return manifest rows without writing them; useful to dataset builders.

    Paths in each row are relative to ``source_root``. This function only uses
    the standard library; Parquet output is isolated to the CLI entry point.
    """
    root = Path(source_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {root}")
    samples = discover_samples(root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        samples = samples[:limit]
    return [make_record(path, root, hash_files) for path in samples]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Root containing category/phase/variant/POSCAR and ELFCAR")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output manifest .parquet path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N sample directories (pilot use)")
    parser.add_argument("--no-hashes", action="store_true",
                        help="Skip SHA-256 (faster, but source integrity is not recorded)")
    parser.add_argument("--json-report", type=Path, default=None,
                        help="Optional JSON summary path")
    args = parser.parse_args()
    root = args.source_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"source root does not exist or is not a directory: {root}")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:
        print("ERROR: Parquet output requires a working pyarrow installation; "
              f"import failed: {exc}", file=sys.stderr)
        return 2
    samples = discover_samples(root)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        samples = samples[:args.limit]
    records = [make_record(path, root, not args.no_hashes) for path in samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, args.output)
    counts: dict[str, int] = {}
    for record in records:
        status = record["processing_status"]
        counts[status] = counts.get(status, 0) + 1
    report = {"source_root": str(root), "manifest_path": str(args.output.resolve()),
              "sample_count": len(records), "status_counts": counts,
              "hashes_enabled": not args.no_hashes}
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
