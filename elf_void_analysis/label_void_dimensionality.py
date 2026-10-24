#!/usr/bin/env python3
"""Label candidate voids by periodic accessible-space dimensionality.

This is a geometric, probe-dependent label. It thresholds nearest-atom
clearance on a periodic fractional grid, finds 6-neighbor connected regions,
and measures each region's independent periodic winding directions. It is not
an electronic/ELF dimensionality analysis.
"""
from __future__ import annotations

from collections import deque
from itertools import product
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


DIMENSION_CLASS_NAMES = ("0D", "1D", "2D", "3D")


def _grid_shape(lattice: np.ndarray, spacing_A: float) -> tuple[int, int, int]:
    # Approximate a requested Cartesian spacing along each lattice vector.
    lengths = np.linalg.norm(lattice, axis=1)
    shape = np.maximum(1, np.ceil(lengths / spacing_A).astype(int))
    return tuple(int(x) for x in shape)


def _periodic_tree(lattice: np.ndarray, atom_frac: np.ndarray, radius_A: float) -> cKDTree:
    """Tree over enough atom images to find every neighbor within radius."""
    inv_lattice = np.linalg.inv(lattice)
    # If a Cartesian displacement is <= radius, each fractional component is
    # bounded by radius times the corresponding inverse-lattice column norm.
    frac_bounds = radius_A * np.linalg.norm(inv_lattice, axis=0)
    limits = np.ceil(frac_bounds).astype(int) + 1
    shifts = np.asarray(list(product(*[
        range(-int(limit), int(limit) + 1) for limit in limits
    ])), dtype=np.int32)
    image_frac = (atom_frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    image_cart = image_frac @ lattice
    return cKDTree(image_cart)


def _accessible_voxels(
    lattice: np.ndarray,
    atom_frac: np.ndarray,
    shape: tuple[int, int, int],
    probe_radius_A: float,
    chunk_size: int,
) -> np.ndarray:
    tree = _periodic_tree(lattice, atom_frac, probe_radius_A)
    total = int(np.prod(shape))
    active = np.empty(total, dtype=bool)
    # Flattened indices are decoded in chunks to avoid allocating the full
    # Cartesian grid (and a grid-by-atoms distance tensor) at once.
    for first in range(0, total, chunk_size):
        flat = np.arange(first, min(first + chunk_size, total), dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(flat, shape))
        frac = ijk / np.asarray(shape, dtype=float)
        cart = frac @ lattice
        nearest, _ = tree.query(cart, k=1, workers=1)
        active[first:first + len(flat)] = nearest >= probe_radius_A - 1e-10
    return active.reshape(shape)


def _component_winding_rank(
    active: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Label active components and compute each component's winding rank."""
    shape = np.asarray(active.shape, dtype=np.int64)
    component_ids = np.full(active.shape, -1, dtype=np.int32)
    ranks: list[int] = []
    neighbor_steps = [
        np.asarray(step, dtype=np.int64)
        for axis in range(3)
        for step in (
            tuple(1 if k == axis else 0 for k in range(3)),
            tuple(-1 if k == axis else 0 for k in range(3)),
        )
    ]
    component_id = 0
    for seed_tuple in np.argwhere(active):
        seed = tuple(int(v) for v in seed_tuple)
        if component_ids[seed] >= 0:
            continue
        # lifted stores coordinates on the infinite cover of the periodic grid.
        lifted: dict[tuple[int, int, int], np.ndarray] = {seed: np.asarray(seed, dtype=np.int64)}
        component_ids[seed] = component_id
        queue: deque[tuple[int, int, int]] = deque([seed])
        cycles: list[np.ndarray] = []
        while queue:
            here = queue.popleft()
            here_arr = np.asarray(here, dtype=np.int64)
            for step in neighbor_steps:
                raw_next = here_arr + step
                nxt_arr = raw_next % shape
                nxt = tuple(int(v) for v in nxt_arr)
                if not active[nxt]:
                    continue
                proposed_lift = lifted[here] + step
                if component_ids[nxt] < 0:
                    component_ids[nxt] = component_id
                    lifted[nxt] = proposed_lift
                    queue.append(nxt)
                else:
                    discrepancy = proposed_lift - lifted[nxt]
                    # A closed path on the periodic grid differs by whole-cell
                    # translations. These integer vectors span the winding space.
                    winding = np.rint(discrepancy / shape).astype(np.int64)
                    if np.any(winding):
                        cycles.append(winding)
        rank = int(np.linalg.matrix_rank(np.asarray(cycles, dtype=float))) if cycles else 0
        ranks.append(rank)
        component_id += 1
    return component_ids, ranks


def label_void_dimensionality(
    lattice_angstrom: Any,
    atom_fractional_coords: Any,
    void_fractional_coords: Any,
    *,
    probe_radius_A: float = 1.0,
    grid_spacing_A: float = 0.5,
    chunk_size: int = 65536,
) -> dict[str, Any]:
    """Return per-void accessible-space dimensionality labels and metadata.

    Labels are ``0``/``1``/``2``/``3`` for components winding around that many
    independent periodic lattice directions. A center outside the active space
    receives ``-1`` and a false validity flag. Grid voxels sample fractional
    points at ``(i/n_x, j/n_y, k/n_z)``; 6-neighbor connectivity uses periodic
    wrapping. ``grid_spacing_A`` sets approximate spacing along cell vectors.
    """
    lattice = np.asarray(lattice_angstrom, dtype=float)
    atoms = np.asarray(atom_fractional_coords, dtype=float)
    voids = np.asarray(void_fractional_coords, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("lattice_angstrom must be a finite 3x3 array")
    if abs(np.linalg.det(lattice)) < 1e-10:
        raise ValueError("lattice_angstrom must be nonsingular")
    if atoms.ndim != 2 or atoms.shape[1:] != (3,) or not len(atoms) or not np.isfinite(atoms).all():
        raise ValueError("atom_fractional_coords must be a nonempty finite [N,3] array")
    if voids.ndim != 2 or voids.shape[1:] != (3,) or not np.isfinite(voids).all():
        raise ValueError("void_fractional_coords must be a finite [N,3] array")
    if not np.isfinite(probe_radius_A) or probe_radius_A <= 0:
        raise ValueError("probe_radius_A must be finite and positive")
    if not np.isfinite(grid_spacing_A) or grid_spacing_A <= 0:
        raise ValueError("grid_spacing_A must be finite and positive")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    shape = _grid_shape(lattice, float(grid_spacing_A))
    active = _accessible_voxels(
        lattice, atoms % 1.0, shape, float(probe_radius_A), int(chunk_size)
    )
    component_ids, component_ranks = _component_winding_rank(active)
    labels = np.full(len(voids), -1, dtype=np.int8)
    valid = np.zeros(len(voids), dtype=np.uint8)
    for i, frac in enumerate(voids % 1.0):
        voxel = tuple(np.floor(frac * np.asarray(shape)).astype(int) % np.asarray(shape))
        comp = int(component_ids[voxel])
        if comp >= 0:
            labels[i] = component_ranks[comp]
            valid[i] = 1
    return {
        "void_dimension": labels,
        "void_dimension_valid": valid,
        "metadata": {
            "definition": "periodic accessible free-space winding rank by nearest-atom clearance",
            "probe_radius_angstrom": float(probe_radius_A),
            "grid_spacing_angstrom": float(grid_spacing_A),
            "grid_shape": list(shape),
            "connectivity": "periodic 6-neighbor voxel connectivity",
            "dimension_class_names": list(DIMENSION_CLASS_NAMES),
            "invalid_label": -1,
            "active_voxel_rule": "nearest-atom distance >= probe radius",
            "voxel_sampling": "fractional grid nodes at index/shape",
            "component_count": len(component_ranks),
            "active_voxel_count": int(active.sum()),
            "caveat": (
                "grid- and probe-dependent geometric label; not an ELF/electronic "
                "classification; labels a component containing the candidate center"
            ),
        },
    }

