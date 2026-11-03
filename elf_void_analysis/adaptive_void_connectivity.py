#!/usr/bin/env python3
"""Explore probe-accessible space from saved-style void sites with adaptive blocks.

Only whole blocks certified by the nearest-atom distance bound are traversed.
Blocks touching an atom-exclusion boundary are subdivided. A resolution or
block-budget limit leaves an explicit unresolved frontier, never a guessed
zero-dimensional cavity. XSF files rasterize the *certified* components for
VESTA; the raster is a visualization, not the connectivity calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .build_graph_records import build_graph_record
from .export_open_space_vesta import write_xsf
from .label_void_dimensionality import _grid_shape, _periodic_tree


Key = tuple[int, int, int, int]  # depth, integer x/y/z coordinates at that depth
STEPS = tuple((axis, sign) for axis in range(3) for sign in (-1, 1))


@dataclass
class Block:
    key: Key
    status: str = "unknown"
    children: tuple[Key, ...] | None = None
    center_clearance_A: float | None = None


class AdaptiveSpace:
    """Lazy periodic octree with certified open/blocked leaves."""

    def __init__(self, lattice: np.ndarray, atoms: np.ndarray, probe_A: float,
                 min_width_A: float, max_blocks: int = 1_000_000):
        self.lattice = np.asarray(lattice, dtype=float)
        self.atoms = np.asarray(atoms, dtype=float)
        if self.lattice.shape != (3, 3) or not np.isfinite(self.lattice).all() or abs(np.linalg.det(self.lattice)) < 1e-10:
            raise ValueError("lattice must be a finite nonsingular [3,3] matrix")
        if self.atoms.ndim != 2 or self.atoms.shape[1:] != (3,) or not len(self.atoms) or not np.isfinite(self.atoms).all():
            raise ValueError("atoms must be a nonempty finite [N,3] array")
        if not np.isfinite(probe_A) or probe_A <= 0 or not np.isfinite(min_width_A) or min_width_A <= 0:
            raise ValueError("probe radius and minimum width must be positive and finite")
        if max_blocks < 1:
            raise ValueError("max_blocks must be positive")
        self.probe_A = float(probe_A)
        self.min_width_A = float(min_width_A)
        self.max_blocks = int(max_blocks)
        longest = max(np.linalg.norm(row) for row in self.lattice)
        self.max_depth = max(0, math.ceil(math.log2(longest / min_width_A)))
        self.resolution = 1 << self.max_depth
        self.nodes: dict[Key, Block] = {(0, 0, 0, 0): Block((0, 0, 0, 0))}
        self.distance_queries = 0
        self.budget_hit = False
        self.root_radius_A = 0.5 * max(
            np.linalg.norm((2 * np.asarray(corner) - 1) @ self.lattice)
            for corner in np.ndindex(2, 2, 2)
        )

        # Any point in the cell is within the cell's Cartesian diameter of at
        # least one basis atom. Tiling to this radius therefore includes the
        # true nearest periodic image even when the query is far from atoms.
        diameter = 2 * self.root_radius_A
        self.tree: cKDTree = _periodic_tree(self.lattice, self.atoms % 1.0, diameter)

    def distance(self, frac: np.ndarray) -> float:
        self.distance_queries += 1
        return float(self.tree.query((np.asarray(frac) % 1.0) @ self.lattice, k=1)[0])

    @staticmethod
    def bounds(key: Key) -> tuple[np.ndarray, np.ndarray]:
        level, x, y, z = key
        lo = np.array([x, y, z], dtype=float) / (1 << level)
        return lo, lo + 1.0 / (1 << level)

    def classify(self, key: Key) -> str:
        node = self.nodes[key]
        if node.status != "unknown":
            return node.status
        lo, hi = self.bounds(key)
        center = (lo + hi) / 2
        radius = self.root_radius_A / (1 << key[0])
        d = self.distance(center)
        node.center_clearance_A = d
        # The nearest-atom distance is 1-Lipschitz in Cartesian coordinates.
        # A small safety margin also avoids claiming certainty at roundoff.
        eps = 1e-9
        if d - radius >= self.probe_A + eps:
            node.status = "open"
        elif d + radius < self.probe_A - eps:
            node.status = "blocked"
        else:
            node.status = "mixed"
        return node.status

    def split(self, key: Key) -> bool:
        node = self.nodes[key]
        if key[0] >= self.max_depth or len(self.nodes) + 8 > self.max_blocks:
            self.budget_hit |= len(self.nodes) + 8 > self.max_blocks
            return False
        if node.children is None:
            level, x, y, z = key
            node.children = tuple((level + 1, 2*x+i, 2*y+j, 2*z+k)
                                  for i in range(2) for j in range(2) for k in range(2))
            self.nodes.update((child, Block(child)) for child in node.children)
        return True

    def locate(self, frac: np.ndarray) -> Key:
        frac = np.asarray(frac, dtype=float) % 1.0
        key: Key = (0, 0, 0, 0)
        while self.nodes[key].children is not None:
            level = key[0] + 1
            xyz = np.floor(frac * (1 << level)).astype(int)
            key = (level, *(xyz.tolist()))
            if key not in self.nodes:
                raise RuntimeError("octree child is missing")
        return key

    @staticmethod
    def _integer_bounds(key: Key, depth: int) -> tuple[np.ndarray, np.ndarray]:
        factor = 1 << (depth - key[0])
        lo = np.asarray(key[1:], dtype=int) * factor
        return lo, lo + factor

    def _leaves_intersecting(self, key: Key, lo: np.ndarray, hi: np.ndarray) -> list[Key]:
        a, b = self._integer_bounds(key, self.max_depth)
        if np.any(b <= lo) or np.any(a >= hi):
            return []
        children = self.nodes[key].children
        if children is None:
            return [key]
        result: list[Key] = []
        for child in children:
            result.extend(self._leaves_intersecting(child, lo, hi))
        return result

    def face_neighbors(self, key: Key, axis: int, sign: int) -> tuple[list[Key], np.ndarray]:
        """Find all leaves touching a face, including across a periodic seam."""
        lo, hi = self._integer_bounds(key, self.max_depth)
        wrap = np.zeros(3, dtype=int)
        slab_lo, slab_hi = lo.copy(), hi.copy()
        coordinate = hi[axis] if sign > 0 else lo[axis] - 1
        if coordinate == self.resolution:
            coordinate = 0
            wrap[axis] = 1
        elif coordinate == -1:
            coordinate = self.resolution - 1
            wrap[axis] = -1
        slab_lo[axis], slab_hi[axis] = coordinate, coordinate + 1
        return self._leaves_intersecting((0, 0, 0, 0), slab_lo, slab_hi), wrap

    def touching_resolved_neighbors(self, key: Key, axis: int, sign: int) -> tuple[list[Key], np.ndarray]:
        """Refine mixed neighbors, then query the touching face again."""
        while True:
            neighbors, wrap = self.face_neighbors(key, axis, sign)
            changed = False
            for other in neighbors:
                if self.classify(other) == "mixed" and self.split(other):
                    changed = True
            if not changed:
                return self.face_neighbors(key, axis, sign)

    def explore(self, voids: np.ndarray) -> dict[str, Any]:
        voids = np.asarray(voids, dtype=float)
        if voids.ndim != 2 or voids.shape[1:] != (3,) or not np.isfinite(voids).all():
            raise ValueError("voids must be finite [N,3] coordinates")
        regions: list[dict[str, Any]] = []
        membership: dict[Key, int] = {}
        seeds: list[Key | None] = []
        invalid: list[bool] = []
        for frac in voids:
            if self.distance(frac) < self.probe_A:
                seeds.append(None)
                invalid.append(True)
                continue
            invalid.append(False)
            key = self.locate(frac)
            while self.classify(key) == "mixed" and self.split(key):
                key = self.locate(frac)
            if self.classify(key) != "open":
                seeds.append(None)
                continue
            seeds.append(key)

        # Locate every seed before exploration, so a later seed cannot refine
        # a formerly unresolved frontier and silently join two prior regions.
        discovered = set(key for key in seeds if key is not None)
        queue = deque(discovered)
        edges: dict[Key, list[tuple[Key, np.ndarray]]] = {}
        unresolved_by_key: dict[Key, set[Key]] = {}
        while queue:
            here = queue.popleft()
            edges[here] = []
            unresolved_by_key[here] = set()
            for axis, sign in STEPS:
                neighbors, wrap = self.touching_resolved_neighbors(here, axis, sign)
                for other in neighbors:
                    status = self.classify(other)
                    if status == "mixed":
                        unresolved_by_key[here].add(other)
                    elif status == "open":
                        edges[here].append((other, wrap.copy()))
                        if other not in discovered:
                            discovered.add(other)
                            queue.append(other)

        for seed in seeds:
            if seed is None or seed in membership:
                continue
            region_id = len(regions)
            component_queue = deque([seed])
            lifts = {seed: np.zeros(3, dtype=int)}
            membership[seed] = region_id
            cycles: list[np.ndarray] = []
            unresolved: set[Key] = set()
            while component_queue:
                here = component_queue.popleft()
                unresolved.update(unresolved_by_key[here])
                for other, wrap in edges[here]:
                    proposed = lifts[here] + wrap
                    if other not in membership:
                        membership[other] = region_id
                        lifts[other] = proposed
                        component_queue.append(other)
                    elif membership[other] == region_id:
                        winding = proposed - lifts[other]
                        if np.any(winding):
                            cycles.append(winding)
            rank = int(np.linalg.matrix_rank(np.asarray(cycles, dtype=float))) if cycles else 0
            # Three certified independent winding directions are conclusive;
            # uncertain boundary blocks cannot add a fourth direction.
            regions.append({"region_id": region_id, "dimension": rank if not unresolved or rank == 3 else None,
                            "certified_winding_rank": rank, "uncertain": bool(unresolved),
                            "open_block_count": sum(v == region_id for v in membership.values()),
                            "unresolved_block_count": len(unresolved),
                            "_unresolved_keys": unresolved})
        assignments = [membership.get(key) if key is not None else None for key in seeds]
        return {"regions": regions, "assignments": assignments, "invalid": invalid,
                "membership": membership}

    def raster(self, keys: set[Key], spacing_A: float) -> np.ndarray:
        """Draw a union of certified leaves on a regular periodic XSF grid."""
        shape = _grid_shape(self.lattice, spacing_A)
        if np.prod(shape, dtype=np.int64) > 10_000_000:
            raise ValueError("XSF grid exceeds 10 million points; increase export spacing")
        out = np.zeros(shape, dtype=bool)
        if not keys:
            return out
        for key in keys:
            lo, hi = self.bounds(key)
            # Sample at i/n, as in the existing grid analyzer. A known-open
            # leaf certifies each included sample without a new atom query.
            lower = np.ceil(lo * shape - 1e-12).astype(int)
            upper = np.ceil(hi * shape - 1e-12).astype(int)
            slices = tuple(slice(max(0, int(a)), min(int(b), n)) for a, b, n in zip(lower, upper, shape))
            out[slices] = True
        return out


def run(structure_path: Path, output: Path, *, probe_A: float = 1.0,
        min_width_A: float = 0.25, export_spacing_A: float = 0.25,
        max_blocks: int = 1_000_000, report_only: bool = False) -> dict[str, Any]:
    """Recompute the established GNN void sites and export certified regions."""
    structure_path = structure_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if not report_only and (not np.isfinite(export_spacing_A) or export_spacing_A <= 0):
        raise ValueError("export spacing must be finite and positive")
    graph = build_graph_record(structure_path)
    structure_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
    structure_id = structure_hash[:16]
    analysis_settings = {"source_sha256": structure_hash,
                         "probe_radius_angstrom": float(probe_A),
                         "minimum_block_width_angstrom": float(min_width_A),
                         "max_blocks": int(max_blocks)}
    analysis_id = hashlib.sha256(json.dumps(analysis_settings, sort_keys=True).encode()).hexdigest()[:16]
    lattice = np.asarray(graph["lattice_angstrom"], dtype=float)
    atoms = np.asarray(graph["atoms"]["fractional_coords"], dtype=float)
    voids = np.asarray(graph["voids"]["fractional_coords"], dtype=float)
    space = AdaptiveSpace(lattice, atoms, probe_A, min_width_A, max_blocks)
    analysis = space.explore(voids)
    output.mkdir(parents=True, exist_ok=True)
    if not report_only:
        atomic_numbers = np.asarray(graph["atoms"]["atomic_numbers"], dtype=int)
        cart_coords = atoms @ lattice
    region_rows = []
    for region in analysis["regions"]:
        region_id = region["region_id"]
        region_key = f"region-{region_id:03d}"
        row = {k: v for k, v in region.items() if not k.startswith("_")}
        row.update({"region_key": region_key,
                    "region_uid": f"{analysis_id}:{region_key}",
                    "connectivity_label": f"{region['certified_winding_rank']}D",
                    "label_status": "exact" if region["dimension"] is not None else "lower_bound",
                    "void_ids": [void_id for void_id, assignment in zip(
                        graph["voids"]["void_id"], analysis["assignments"])
                        if assignment == region_id],
                    "xsf_file": None, "export_grid_shape": None, "export_open_samples": None})
        if not report_only:
            keys = {key for key, value in analysis["membership"].items() if value == region_id}
            mask = space.raster(keys, export_spacing_A)
            filename = f"{region_key}.xsf"
            write_xsf(output / filename, lattice, atomic_numbers, cart_coords, mask)
            row.update({"xsf_file": filename, "export_grid_shape": list(mask.shape),
                        "export_open_samples": int(mask.sum())})
        region_rows.append(row)
    unresolved_keys = set().union(*(r["_unresolved_keys"] for r in analysis["regions"])) if analysis["regions"] else set()
    if unresolved_keys and not report_only:
        write_xsf(output / "unresolved.xsf", lattice, atomic_numbers, cart_coords,
                  space.raster(unresolved_keys, export_spacing_A))
    void_rows = [{"void_id": void_id, "fractional_coords": np.asarray(frac, dtype=float).tolist(),
                  "region_id": region_id, "invalid": invalid,
                  "assignment_status": "below_probe_cutoff" if invalid else
                                       "unresolved_seed" if region_id is None else "assigned",
                  "region_uid": region_rows[region_id]["region_uid"] if region_id is not None else None,
                  "connectivity_label": region_rows[region_id]["connectivity_label"] if region_id is not None else None,
                  "label_status": region_rows[region_id]["label_status"] if region_id is not None else None,
                  "dimension": region_rows[region_id]["dimension"] if region_id is not None else None}
                 for void_id, frac, region_id, invalid in zip(
                     graph["voids"]["void_id"], voids, analysis["assignments"], analysis["invalid"])]
    report = {"source_structure": str(structure_path), "structure_id": structure_id,
              "analysis_id": analysis_id,
              "source_sha256": structure_hash, "report_only": bool(report_only),
              "probe_radius_angstrom": float(probe_A),
              "minimum_block_width_angstrom": float(min_width_A),
              "export_grid_spacing_angstrom": None if report_only else float(export_spacing_A),
              "max_depth": space.max_depth, "max_blocks": space.max_blocks,
              "distance_queries": space.distance_queries,
              "allocated_blocks": len(space.nodes), "budget_hit": space.budget_hit,
              "regions": region_rows, "voids": void_rows,
              "unresolved_xsf": "unresolved.xsf" if unresolved_keys and not report_only else None,
              "display_in_vesta": None if report_only else "Open region-*.xsf and display accessible_space isosurface at 0.5.",
              "interpretation": "Geometric space accessible to a spherical probe, not ELF or electronic dimensionality.",
              "export_note": "XSF is a raster of certified open blocks; uncertain boundary space is omitted from region files."
                             if not report_only else "No XSF files requested; region connectivity was still calculated."}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path, help="POSCAR, CIF, or another pymatgen-readable structure")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory outside the repository")
    parser.add_argument("--probe-radius", type=float, default=1.0, help="Atom-center exclusion radius in Å")
    parser.add_argument("--min-block-width", type=float, default=0.25, help="Maximum side-vector length at finest level in Å")
    parser.add_argument("--export-spacing", type=float, default=0.25, help="Approximate XSF grid spacing in Å")
    parser.add_argument("--report-only", action="store_true", help="Write connectivity report without XSF grids")
    parser.add_argument("--max-blocks", type=int, default=1_000_000)
    args = parser.parse_args()
    if not args.report_only and (not np.isfinite(args.export_spacing) or args.export_spacing <= 0):
        parser.error("export spacing must be finite and positive")
    report = run(args.structure, args.output, probe_A=args.probe_radius,
                 min_width_A=args.min_block_width, export_spacing_A=args.export_spacing,
                 max_blocks=args.max_blocks, report_only=args.report_only)
    mode = "connectivity report" if args.report_only else f"{len(report['regions'])} region XSF files"
    print(f"Wrote {mode} to {args.output.resolve()}; "
          f"{report['distance_queries']} atom-distance checks")


if __name__ == "__main__":
    main()
