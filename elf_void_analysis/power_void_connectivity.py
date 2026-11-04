#!/usr/bin/env python3
"""Exact probe-accessible void connectivity from a periodic power diagram.

Each atom i is an excluded ball of radius R_i = r_i + probe, where r_i is an
element radius (zero reproduces the uniform atom-center cutoff). A point x is
open when its power distance |x - c_i|^2 - R_i^2 is positive for every atom.

Inside the power (radical Voronoi) cell of atom i, the union of all balls is
exactly ball i. Pushing open points away from a point of that ball along
straight rays therefore retracts the open part of each cell onto the open
part of its faces, then edges. The open space and the open power-diagram
1-skeleton have the same components and periodic winding directions, so every
component gets an exact 0D-3D label. The power vertices and edges come from
the weighted Delaunay triangulation (the lower convex hull of lifted points)
of enough periodic atom images, with an emptiness check on every vertex used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from collections import deque
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, cKDTree

from .build_graph_records import build_graph_record
from .export_open_space_vesta import write_xsf
from .label_void_dimensionality import _grid_shape


RADIUS_SCHEMES = ("none", "ionic", "atomic", "custom")
TOL = 1e-9


def element_radii(symbols: list[str], scheme: str,
                  custom: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, str]]:
    """Return a radius per element in Å and a note on where each came from.

    ``ionic`` uses pymatgen's Shannon crystal radii (six-coordinate default).
    Electrides are not charge balanced, so oxidation states are assigned per
    element: the most electronegative element takes its common negative state
    and every other element its smallest common positive state with a radius.
    """
    unique = sorted(set(symbols))
    if scheme == "none":
        return {s: 0.0 for s in unique}, {s: "zero (uniform atom-center cutoff)" for s in unique}
    if scheme == "custom":
        if custom is None:
            raise ValueError("custom radius scheme needs a radius table")
        missing = [s for s in unique if s not in custom]
        if missing:
            raise ValueError(f"custom radius table lacks {missing}")
        return {s: float(custom[s]) for s in unique}, {s: "custom table" for s in unique}
    from pymatgen.core import Element, Species
    if scheme == "atomic":
        return ({s: float(Element(s).atomic_radius) for s in unique},
                {s: "pymatgen empirical atomic radius" for s in unique})
    if scheme != "ionic":
        raise ValueError(f"unknown radius scheme {scheme!r}")
    anion = max(unique, key=lambda s: Element(s).X) if len(unique) > 1 else None
    radii, notes = {}, {}
    for symbol in unique:
        element = Element(symbol)
        states = sorted(element.common_oxidation_states)
        if symbol == anion and any(o < 0 for o in states):
            candidates = [o for o in states if o < 0]
        else:
            candidates = [o for o in states if o > 0]
        for oxidation in candidates:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                radius = Species(symbol, oxidation).ionic_radius
            if radius is not None:
                radii[symbol] = float(radius)
                notes[symbol] = f"Shannon crystal radius for {symbol}{oxidation:+d}"
                break
        else:
            radii[symbol] = float(element.atomic_radius)
            notes[symbol] = f"no Shannon radius for {symbol}{candidates[:1]}; empirical atomic radius"
    return radii, notes


class PowerSpace:
    """Open power-diagram network of a periodic set of excluded balls."""

    def __init__(self, lattice: np.ndarray, atoms: np.ndarray, radii_A: np.ndarray):
        self.lattice = np.asarray(lattice, dtype=float)
        self.atoms = np.asarray(atoms, dtype=float) % 1.0
        self.radii = np.asarray(radii_A, dtype=float)
        if self.lattice.shape != (3, 3) or not np.isfinite(self.lattice).all() or abs(np.linalg.det(self.lattice)) < 1e-10:
            raise ValueError("lattice must be a finite nonsingular [3,3] matrix")
        if self.atoms.ndim != 2 or self.atoms.shape[1:] != (3,) or not len(self.atoms) or not np.isfinite(self.atoms).all():
            raise ValueError("atoms must be a nonempty finite [N,3] array")
        if self.radii.shape != (len(self.atoms),) or not np.isfinite(self.radii).all() or (self.radii <= 0).any():
            raise ValueError("every excluded-ball radius must be positive and finite")
        # Interplanar spacings convert a Cartesian margin to fractional padding.
        self.spacing = 1.0 / np.linalg.norm(np.linalg.inv(self.lattice), axis=0)
        self.origin = 0.5 * self.lattice.sum(axis=0)
        padding = max(2.5 * float(self.radii.max()), 6.0)
        for _ in range(6):
            if self._build(padding):
                return
            padding *= 1.5
        raise RuntimeError("periodic image shell never covered every power vertex")

    # ----------------------------------------------------------------- build
    def _build(self, padding_A: float) -> bool:
        pad = padding_A / self.spacing
        reach = np.ceil(pad).astype(int) + 1
        shifts = np.array(list(product(*(range(-r, r + 1) for r in reach))), dtype=int)
        frac = (self.atoms[None] + shifts[:, None]).reshape(-1, 3)
        keep = np.all((frac >= -pad) & (frac <= 1 + pad), axis=1)
        self.image_atom = np.tile(np.arange(len(self.atoms)), len(shifts))[keep]
        self.image_shift = np.repeat(shifts, len(self.atoms), axis=0)[keep]
        self.image_frac = frac[keep]
        self.centers = self.image_frac @ self.lattice - self.origin
        self.weights = self.radii[self.image_atom] ** 2
        lifted = np.column_stack([self.centers, (self.centers ** 2).sum(1) - self.weights])
        hull = ConvexHull(lifted, qhull_options="Qt Qc")
        lower = hull.equations[:, 3] < -TOL
        normal, offset = hull.equations[:, :3], hull.equations[:, 4]
        # Lower facet: h = a.x + b with a = 2*o, power at o = |o|^2 + b.
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(lower[:, None], -normal / hull.equations[:, 3:4], np.nan)
            b = np.where(lower, -offset / hull.equations[:, 3], np.nan)
        self.vertex_xyz = a / 2
        self.vertex_power = (self.vertex_xyz ** 2).sum(1) + b
        self.simplices, self.neighbors, self.lower = hull.simplices, hull.neighbors, lower
        frac_v = (np.nan_to_num(self.vertex_xyz) + self.origin) @ np.linalg.inv(self.lattice)

        # A power vertex is correct when every site that could lie inside its
        # orthosphere (distance < sqrt(power + R_max^2)) is present.
        rmax = float(self.radii.max())
        margin = np.min(np.minimum(frac_v + pad, 1 + pad - frac_v) * self.spacing, axis=1)
        reach = np.sqrt(np.maximum(np.nan_to_num(self.vertex_power), 0.0) + rmax ** 2)
        self.valid = lower & (margin > reach)
        central = lower & np.all((frac_v > -0.05) & (frac_v < 1.05), axis=1)
        if not self.valid[central].all() or not self.valid[self.neighbors[central]].all():
            return False

        # Degenerate (cospherical) sites are triangulated differently in
        # translated copies, so power vertices are identified by position
        # modulo the lattice, not by the tetrahedra that produced them.
        self.shift = np.floor(frac_v + 1e-7).astype(int)
        base = (frac_v - self.shift) @ self.lattice
        facets = np.flatnonzero(self.valid)
        parent = {int(f): int(f) for f in facets}

        def find(f: int) -> int:
            while parent[f] != f:
                parent[f] = parent[parent[f]]
                f = parent[f]
            return f
        for i, j in cKDTree(base[facets]).query_pairs(1e-5):
            parent[find(int(facets[i]))] = find(int(facets[j]))
        roots = sorted({find(int(f)) for f in facets})
        index = {root: n for n, root in enumerate(roots)}
        self.node = {int(f): index[find(int(f))] for f in facets}
        self.node_power = np.zeros(len(roots))
        for f, n in self.node.items():
            self.node_power[n] = self.vertex_power[f]
        return True

    # ----------------------------------------------------------- open graph
    def segment_min_power(self, start: np.ndarray, end: np.ndarray, site: int) -> float:
        """Minimum power w.r.t. one image site along a straight segment."""
        d = end - start
        rel = start - self.centers[site]
        dd = float(d @ d)
        t = 0.0 if dd < 1e-24 else min(1.0, max(0.0, -float(d @ rel) / dd))
        point = rel + t * d
        return float(point @ point - self.weights[site])

    def explore(self) -> dict[str, Any]:
        """Components of the open network with their periodic winding rank."""
        open_nodes = {n for n in range(len(self.node_power)) if self.node_power[n] > TOL}
        edges: dict[int, set[tuple[int, tuple[int, int, int]]]] = {n: set() for n in open_nodes}
        for f, here in self.node.items():
            if here not in open_nodes:
                continue
            for k, g in enumerate(self.neighbors[f]):
                other = self.node.get(int(g))
                if other is None or other not in open_nodes:
                    continue
                wrap = self.shift[g] - self.shift[f]
                if other == here and not wrap.any():
                    continue
                # Along a power edge every site of the shared triangle has the
                # same power; its minimum decides whether the passage is open.
                site = self.simplices[f][(k + 1) % 4]
                if self.segment_min_power(self.vertex_xyz[f], self.vertex_xyz[g], site) <= TOL:
                    continue
                edges[here].add((other, tuple(int(w) for w in wrap)))
                edges[other].add((here, tuple(-int(w) for w in wrap)))
        membership: dict[int, int] = {}
        regions = []
        for start in sorted(open_nodes):
            if start in membership:
                continue
            region_id = len(regions)
            lifts = {start: np.zeros(3, dtype=int)}
            membership[start] = region_id
            queue, cycles = deque([start]), []
            while queue:
                here = queue.popleft()
                for other, wrap in edges[here]:
                    proposed = lifts[here] + wrap
                    if other not in membership:
                        membership[other], lifts[other] = region_id, proposed
                        queue.append(other)
                    elif membership[other] != region_id:
                        raise RuntimeError("open power-diagram edges are not symmetric")
                    elif np.any(proposed != lifts[other]):
                        cycles.append(proposed - lifts[other])
            rank = int(np.linalg.matrix_rank(np.asarray(cycles, dtype=float))) if cycles else 0
            nodes = [k for k, r in membership.items() if r == region_id]
            regions.append({"region_id": region_id, "dimension": rank,
                            "power_vertex_count": len(nodes),
                            "max_vertex_power_A2": float(self.node_power[nodes].max())})
        self.membership = membership
        return {"regions": regions, "membership": membership}

    # -------------------------------------------------------- seed placement
    def _power_all(self, x: np.ndarray) -> np.ndarray:
        return ((self.centers - x) ** 2).sum(1) - self.weights

    def _exit(self, cell: int, start: np.ndarray, direction: np.ndarray, plane: tuple[int, ...]) -> tuple[float, list[int]]:
        """Distance along a ray until another site's power drops below ``cell``'s."""
        ci, wi = self.centers[cell], self.weights[cell]
        diff_c = self.centers - ci
        const = (self.centers ** 2).sum(1) - self.weights - (ci @ ci - wi)
        # power_k - power_i = const_k - 2 x.(c_k - c_i), linear along the ray.
        value = const - 2 * diff_c @ start
        slope = -2 * diff_c @ direction
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(slope < -1e-14, np.maximum(-value / slope, 0.0), np.inf)
        t[list(plane)] = np.inf
        best = float(np.min(t))
        if not np.isfinite(best):
            raise RuntimeError("ray never leaves its power cell")
        return best, [int(k) for k in np.flatnonzero(t <= best + 1e-7)]

    def _facets_with(self, sites: list[int]) -> list[int]:
        mask = (np.isin(self.simplices, sites).sum(1) == len(sites)) & self.lower
        return [int(f) for f in np.flatnonzero(mask)]

    def locate(self, frac: np.ndarray) -> tuple[int | None, str]:
        """Find the open component holding a point by straight retractions."""
        x = (np.asarray(frac, dtype=float) % 1.0) @ self.lattice - self.origin
        power = self._power_all(x)
        cell = int(np.argmin(power))
        if power[cell] <= TOL:
            return None, "below_probe_cutoff"
        sites = [cell]
        # Cell -> face -> edge: each step moves away from the covered ball
        # along a ray, so the open component never changes.
        for _ in range(3):
            if len(sites) == 4:
                break
            if len(sites) == 1:
                anchor = self.centers[cell]
            else:
                anchor = self._foot(sites)
            if self._power_all(anchor)[cell] >= 0:
                # The covered ball misses this face/edge's whole span, so a
                # straight segment to any of its vertices stays open.
                break
            direction = x - anchor
            if float(direction @ direction) < 1e-20:
                return None, "seed_at_ball_center"
            t, crossing = self._exit(cell, x, direction, tuple(sites))
            x = x + t * direction
            sites = sites + crossing[: 4 - len(sites)]
        # Every move above increased the cell's power, so x is still open.
        # Symmetric structures can triangulate a degenerate vertex
        # differently from the crossing order, so also try smaller site sets.
        for size in range(len(sites), 0, -1):
            for f in self._facets_with(sites[:size]):
                if (self.node.get(f) in self.membership
                        and self.segment_min_power(x, self.vertex_xyz[f], cell) > TOL):
                    return self.node[f], "assigned"
        return None, "unplaced_seed"

    def _foot(self, sites: list[int]) -> np.ndarray:
        """Lowest-power point of the face (2 sites) or edge (3 sites) affine span."""
        c, w = self.centers[sites], self.weights[sites]
        rows = 2 * (c[1:] - c[0])
        rhs = (c[1:] ** 2).sum(1) - w[1:] - (c[0] @ c[0] - w[0])
        # Minimize |x - c0|^2 - w0 on {rows x = rhs}; the solution lies in the
        # span of c0 plus the row space.
        basis = rows.T
        coeff = np.linalg.solve(rows @ basis, rhs - rows @ c[0])
        return c[0] + basis @ coeff

    # ---------------------------------------------------------------- raster
    def raster(self, spacing_A: float, samples: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
        """Grid of exact region IDs for display (-1 is covered space).

        Open grid points are exact. Grid components are then given the exact
        region of a few of their most open points; a component whose samples
        disagree (two regions closer than the grid spacing) is located point
        by point. The grid only draws the regions; it never decides them.
        """
        shape = _grid_shape(self.lattice, spacing_A)
        if np.prod(shape, dtype=np.int64) > 10_000_000:
            raise ValueError("XSF grid exceeds 10 million points; increase export spacing")
        frac = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape], indexing="ij"), -1).reshape(-1, 3)
        x = frac @ self.lattice - self.origin
        tree = cKDTree(self.centers)
        k = min(32, len(self.centers))
        wmax = float(self.weights.max())
        power = np.empty(len(x))
        for start in range(0, len(x), 200_000):
            chunk = x[start:start + 200_000]
            dist, idx = tree.query(chunk, k=k)
            dist, idx = dist.reshape(len(chunk), -1), idx.reshape(len(chunk), -1)
            best = (dist ** 2 - self.weights[idx]).min(1)
            # A farther site can only win if its distance bound allows it.
            unsure = np.flatnonzero(dist[:, -1] ** 2 - wmax < best)
            for i in unsure:
                best[i] = self._power_all(chunk[i]).min()
            power[start:start + len(chunk)] = best
        power = power.reshape(shape)
        open_mask = power > TOL
        labels, count = ndimage.label(open_mask)
        parent = list(range(count + 1))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for axis in range(3):
            first = np.take(labels, 0, axis=axis)
            last = np.take(labels, -1, axis=axis)
            for a, b in set(zip(first[(first > 0) & (last > 0)], last[(first > 0) & (last > 0)])):
                parent[find(int(a))] = find(int(b))
        roots = np.array([find(i) for i in range(count + 1)])
        components = roots[labels]
        out = np.full(shape, -1, dtype=int)
        conflicts = unplaced = 0
        for comp in np.unique(components[open_mask]):
            points = np.flatnonzero((components == comp).ravel())
            order = points[np.argsort(power.ravel()[points])[::-1][:samples]]
            found = {self.membership.get(self.locate(frac[i])[0]) for i in order}
            found.discard(None)
            if len(found) == 1:
                out.ravel()[points] = found.pop()
                continue
            conflicts += len(found) > 1
            for i in points:
                node = self.locate(frac[i])[0]
                region = self.membership.get(node)
                if region is None:
                    unplaced += 1
                else:
                    out.ravel()[i] = region
        self._grid_xyz = frac @ self.lattice
        return out, {"export_grid_shape": list(shape), "grid_components": int(len(np.unique(components[open_mask]))),
                     "grid_components_with_several_regions": int(conflicts),
                     "unplaced_open_grid_points": int(unplaced)}

    def probe_reach(self, grid: np.ndarray, probe_A: float) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Per region: grid points where a probe center fits, and points a fitting probe covers.

        A region too small to contain any grid point gets the grid points
        nearest its exact power vertices as centers, so every pocket is drawn.
        Filled space is grown from those centers and the exact vertices.
        """
        xyz = self._grid_xyz
        node_xyz: dict[int, np.ndarray] = {}
        for f, n in self.node.items():
            node_xyz.setdefault(n, self.vertex_xyz[f] + self.origin)
        inverse = np.linalg.inv(self.lattice)
        reach = int(math.ceil(probe_A / self.spacing.min())) + 1
        shifts = np.array(list(product(range(-reach, reach + 1), repeat=3))) @ self.lattice
        out = {}
        for region in sorted(set(self.membership.values())):
            vertices = np.array([node_xyz[n] for n, r in self.membership.items()
                                 if r == region and self.node_power[n] > TOL])
            vertices = (vertices @ inverse % 1.0) @ self.lattice
            center_mask = grid == region
            if not center_mask.any():
                frac = vertices @ inverse
                index = np.rint(frac * np.asarray(grid.shape)).astype(int) % np.asarray(grid.shape)
                center_mask = center_mask.copy()
                center_mask[tuple(index.T)] = True
            if probe_A <= 0:
                out[region] = (center_mask, center_mask)
                continue
            centers = np.vstack([xyz[center_mask.ravel()], vertices])
            tree = cKDTree((centers[None] + shifts[:, None]).reshape(-1, 3))
            dist, _ = tree.query(xyz, distance_upper_bound=probe_A + 1e-9)
            out[region] = (center_mask, np.isfinite(dist).reshape(grid.shape) | center_mask)
        return out


def run(structure_path: Path, output: Path, *, probe_A: float = 1.0,
        radius_scheme: str = "none", custom_radii: dict[str, float] | None = None,
        export_spacing_A: float | None = None) -> dict[str, Any]:
    """Label every candidate void with an exact geometric dimension; optionally write XSF."""
    structure_path = structure_path.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    if not np.isfinite(probe_A) or probe_A < 0:
        raise ValueError("probe radius must be finite and non-negative")
    graph = build_graph_record(structure_path)
    from pymatgen.core import Element
    symbols = [Element.from_Z(int(z)).symbol for z in graph["atoms"]["atomic_numbers"]]
    radii_by_element, notes = element_radii(symbols, radius_scheme, custom_radii)
    radii = np.array([radii_by_element[s] + probe_A for s in symbols])
    structure_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
    settings = {"source_sha256": structure_hash, "method": "power_diagram",
                "probe_radius_angstrom": float(probe_A), "radius_scheme": radius_scheme,
                "element_radii_angstrom": radii_by_element}
    analysis_id = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    space = PowerSpace(np.asarray(graph["lattice_angstrom"]), np.asarray(graph["atoms"]["fractional_coords"]), radii)
    analysis = space.explore()
    void_rows, assigned = [], {}
    for void_id, frac in zip(graph["voids"]["void_id"], graph["voids"]["fractional_coords"]):
        node, status = space.locate(np.asarray(frac))
        region_id = analysis["membership"].get(node) if node is not None else None
        if region_id is not None:
            assigned.setdefault(region_id, []).append(void_id)
        void_rows.append({"void_id": void_id, "fractional_coords": list(map(float, frac)),
                          "region_id": region_id, "assignment_status": status})
    region_rows = []
    for region in analysis["regions"]:
        key = f"region-{region['region_id']:03d}"
        region_rows.append(region | {"region_key": key, "region_uid": f"{analysis_id}:{key}",
                                     "connectivity_label": f"{region['dimension']}D",
                                     "label_status": "exact",
                                     "void_ids": assigned.get(region["region_id"], [])})
    for row in void_rows:
        region = region_rows[row["region_id"]] if row["region_id"] is not None else None
        for field in ("region_uid", "connectivity_label", "label_status", "dimension"):
            row[field] = region[field] if region else None
    report = {"source_structure": str(structure_path), "analysis_id": analysis_id, **settings,
              "element_radius_sources": notes,
              "regions": region_rows, "voids": void_rows,
              "interpretation": "Exact connectivity of space outside balls of radius element radius + probe; "
                                "geometry, not ELF or electronic dimensionality."}
    output.mkdir(parents=True, exist_ok=True)
    if export_spacing_A is not None:
        if not np.isfinite(export_spacing_A) or export_spacing_A <= 0:
            raise ValueError("export spacing must be finite and positive")
        grid, export = space.raster(export_spacing_A)
        reach = space.probe_reach(grid, probe_A)
        numbers = np.asarray(graph["atoms"]["atomic_numbers"], dtype=int)
        cart = np.asarray(graph["atoms"]["fractional_coords"], dtype=float) @ space.lattice
        # Each view is its own 0/1 grid so one isosurface level (0.5) fits all
        # files: -centers marks where a probe center fits, -filled marks all
        # space covered by a probe ball that fits.
        empty = np.zeros(grid.shape, dtype=bool)
        all_centers = np.any([c for c, _ in reach.values()], axis=0) if reach else empty
        all_filled = np.any([f for _, f in reach.values()], axis=0) if reach else empty
        write_xsf(output / "open_space-centers.xsf", space.lattice, numbers, cart, all_centers)
        write_xsf(output / "open_space-filled.xsf", space.lattice, numbers, cart, all_filled)
        for row in region_rows:
            centers, filled = reach[row["region_id"]]
            stem = f"{row['region_key']}-{row['connectivity_label']}"
            row["export_center_samples"] = int(centers.sum())
            row["export_filled_samples"] = int(filled.sum())
            row["xsf_centers_file"] = f"{stem}-centers.xsf"
            row["xsf_filled_file"] = f"{stem}-filled.xsf"
            write_xsf(output / row["xsf_centers_file"], space.lattice, numbers, cart, centers)
            write_xsf(output / row["xsf_filled_file"], space.lattice, numbers, cart, filled)
        report.update(export | {"export_grid_spacing_angstrom": float(export_spacing_A),
                                "open_space_xsf": {"centers": "open_space-centers.xsf",
                                                   "filled": "open_space-filled.xsf"},
                                "display_in_vesta": "Open any .xsf file and set the isosurface level to 0.5. "
                                                    "-centers files show where a probe center can sit; -filled "
                                                    "files show all space a fitting probe ball covers."})
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("structure", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-radius", type=float, default=1.0,
                        help="Probe radius in Å, added to every element radius")
    parser.add_argument("--radius-scheme", choices=RADIUS_SCHEMES, default="none")
    parser.add_argument("--radii-json", type=Path, help="Element-to-radius table for --radius-scheme custom")
    parser.add_argument("--export-spacing", type=float,
                        help="Also write VESTA XSF grids at this approximate spacing in Å")
    args = parser.parse_args()
    custom = json.loads(args.radii_json.read_text()) if args.radii_json else None
    report = run(args.structure, args.output, probe_A=args.probe_radius,
                 radius_scheme=args.radius_scheme, custom_radii=custom,
                 export_spacing_A=args.export_spacing)
    labels = ", ".join(r["connectivity_label"] for r in report["regions"]) or "no open regions"
    print(f"Wrote {args.output.resolve() / 'report.json'}: {labels}")


if __name__ == "__main__":
    main()
