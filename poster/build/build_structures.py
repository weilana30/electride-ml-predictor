#!/usr/bin/env python3
"""Real crystal geometry and void surfaces for the Blender connectivity figures.

For each material this reads the repository CIF, builds the conventional cell,
and finds the probe-accessible space with the poster's setting: Shannon ionic
radius + 1.0 A probe, the same radii power_void_connectivity.py reports.
The surface is that space thickened by each material's "offset" (angstrom, less
than the 1.0 A probe) so it is visible; the regions and their connectivity are unchanged.

Surfaces are geometric void regions. They are not ELF or charge density.
For readability the 0D cages are drawn as spheres at their real centres, and the
3D panel is a schematic rod lattice rather than a crystal structure.

    python poster/build/build_structures.py OUTPUT_DIR
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy import ndimage, sparse
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]  # repository root
PROBE = 1.0
SPACING = 0.14          # grid spacing, angstrom

# label, cif, ionic radii, repeats of the conventional cell, origin (fractional), bonds
MATERIALS = {
    "0D": dict(name="Ca6Al7O16", cif="mp-721592", radii={"Ca": 1.14, "Al": 0.675, "O": 1.26},
               repeat=(1, 1, 1), origin=(0, 0, 0), bonds={}, keep="all", offset=0.45, smooth=2.6, whole=True, relax=40,
               show=("Ca", "Al")),
    "1D": dict(name="Y5Si3", cif="mp-2538", radii={"Y": 1.04, "Si": 1.1},
               repeat=(2, 2, 3), origin=(-0.5, -0.5, 0), bonds={}, keep="largest", offset=0.62, smooth=2.0, relax=15, rectangle=True),
    "2D": dict(name="Ca2N", cif="mp-2686", radii={"Ca": 1.14, "N": 1.32},
               repeat=(5, 5, 1), origin=(0, 0, 0), bonds={}, keep="all", offset=0.38, smooth=1.5),
}


def periodic_components(mask):
    """Label a boolean grid, joining components across periodic faces."""
    labels, count = ndimage.label(mask)
    parent = list(range(count + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for axis in range(3):
        a, b = np.take(labels, 0, axis), np.take(labels, -1, axis)
        for p, q in zip(a[(a > 0) & (b > 0)], b[(a > 0) & (b > 0)]):
            parent[find(int(p))] = find(int(q))
    roots = np.array([find(i) for i in range(count + 1)])
    merged = roots[labels]
    ids, sizes = np.unique(merged[merged > 0], return_counts=True)
    return merged, ids, sizes


def dimension(mask, size):
    """Dimensionality of the largest region: how much it grows in a 2x2x2 periodic tiling.

    A pocket stays one cell's size, a channel doubles, a sheet grows 4x, a network 8x.
    """
    _, _, sizes = periodic_components(np.tile(mask, (2, 2, 2)))
    return int(round(np.log2(sizes.max() / size)))


def relax(verts, faces, rounds):
    """Taubin smoothing: removes grid texture without shrinking the surface."""
    n = len(verts)
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = sparse.coo_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    a.data[:] = 1
    a = sparse.diags(1 / np.maximum(np.asarray(a.sum(1)).ravel(), 1)) @ a
    for _ in range(rounds):
        verts = verts + 0.5 * (a @ verts - verts)
        verts = verts - 0.53 * (a @ verts - verts)
    return verts


def build(label, spec, out):
    s = Structure.from_file(ROOT / "data" / "cifs" / f"{spec['cif']}.cif")
    conv = SpacegroupAnalyzer(s, 0.05).get_conventional_standard_structure()
    L = conv.lattice.matrix
    n = np.maximum(np.ceil(conv.lattice.abc / np.array(SPACING)).astype(int), 8)
    frac = np.stack(np.meshgrid(*[np.arange(k) / k for k in n], indexing="ij"), -1).reshape(-1, 3)
    cart = frac @ L

    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    clearance = np.full(len(cart), np.inf)
    for element, radius in spec["radii"].items():
        sites = np.array([x.frac_coords for x in conv if x.specie.symbol == element])
        images = ((sites[None] + shifts[:, None]).reshape(-1, 3)) @ L
        d, _ = cKDTree(images).query(cart, workers=-1)
        clearance = np.minimum(clearance, d - radius - PROBE)
    clearance = clearance.reshape(n)

    labels, ids, sizes = periodic_components(clearance > 0)
    big = int(sizes.max())
    found = dimension(clearance > 0, big)
    print(label, spec["name"], "grid", n.tolist(), "atoms", len(conv), "regions", len(ids),
          "with >100 voxels:", int((sizes > 100).sum()), "largest region dimension:", found, flush=True)
    assert f"{found}D" == label, f"{spec['name']}: expected {label}, found {found}D"

    keep = ids if spec["keep"] == "all" else ids[[int(np.argmax(sizes))]]
    region = np.isin(labels, keep)

    offset = spec["offset"]
    # Thicken the kept region by a true distance, so separate regions keep their identity
    images = (frac[region.reshape(-1)][None] + shifts[:, None]).reshape(-1, 3) @ L
    d, _ = cKDTree(images).query(cart, workers=-1, distance_upper_bound=offset + 0.6)
    d = np.where(np.isfinite(d), d, offset + 0.6).reshape(n)
    field = ndimage.gaussian_filter(offset - d, spec["smooth"], mode="wrap")

    # Tile into the displayed box and cap the surface flat on the box faces
    rep, org = np.array(spec["repeat"]), np.array(spec["origin"], float)
    rect = None
    if spec.get("rectangle"):
        # Rectangular box on the hexagonal lattice: two staggered rows of two whole channels
        a, c = conv.lattice.a, conv.lattice.c
        lo = np.array([0.25 * a, -0.433 * a, 0.0])
        hi = np.array([2.25 * a, 1.299 * a, rep[2] * c])
        axes = [np.arange(lo[k], hi[k] + SPACING, SPACING) for k in range(3)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        where = ((points @ np.linalg.inv(L)) * n).T
        box = ndimage.map_coordinates(field, where, order=1, mode="grid-wrap").reshape([len(v) for v in axes])
        for k in range(3):
            shape = [1, 1, 1]
            shape[k] = -1
            box = np.minimum(box, np.minimum(axes[k] - lo[k], hi[k] - axes[k]).reshape(shape) - 1e-3)
        rect = (lo, hi)
        start = None
    elif spec.get("whole"):
        # Draw each closed region whole, keeping those centred in the cell
        low = -np.round(0.3 * n).astype(int)
        idx = [np.arange(low[a], rep[a] * n[a] - low[a] + 1) % n[a] for a in range(3)]
        box = field[np.ix_(*idx)]
        parts, count = ndimage.label(box > 0)
        centres = np.array(ndimage.center_of_mass(box > 0, parts, range(1, count + 1)))
        centres = (centres + low) / n
        inside = np.flatnonzero(np.all((centres >= 0) & (centres < 1), axis=1)) + 1
        print("   whole regions drawn:", len(inside), flush=True)
        void_centres = (centres[inside - 1] @ L).tolist()
        box = np.where(np.isin(parts, inside), box, -1.0)
        start = low / n
    else:
        idx = [(np.arange(rep[a] * n[a] + 1) + int(round(org[a] * n[a]))) % n[a] for a in range(3)]
        box = field[np.ix_(*idx)]
        heights = 1 / np.linalg.norm(np.linalg.inv(L), axis=0)  # plane spacing along each axis
        for a in range(3):
            t = np.arange(box.shape[a]) / n[a]
            face = np.minimum(t, rep[a] - t) * heights[a]
            shape = [1, 1, 1]
            shape[a] = -1
            box = np.minimum(box, face.reshape(shape) - 1e-3)
        start = org
    verts, faces, *_ = marching_cubes(np.pad(box, 1, constant_values=-1), 0.0)
    verts = (verts - 1) * SPACING + rect[0] if rect else ((verts - 1) / n + start) @ L
    if spec.get("relax"):
        verts = relax(verts, faces, spec["relax"])

    # Atoms inside the box, including those on its faces
    atoms = []
    shown = spec.get("show")
    if rect:
        org, rep = np.array([-2.0, -2.0, 0.0]), np.array([6.0, 6.0, float(rep[2])])
    span = [range(int(np.floor(org[a])) - 1, int(np.ceil(org[a] + rep[a])) + 1) for a in range(3)]
    for site in conv:
        for i in span[0]:
            for j in span[1]:
                for k in span[2]:
                    f = site.frac_coords % 1 + (i, j, k)
                    wanted = shown is None or site.specie.symbol in shown
                    p = f @ L
                    fits = (np.all(p > rect[0] - 1e-3) and np.all(p < rect[1] + 1e-3)) if rect else (
                        np.all(f > org - 1e-3) and np.all(f < org + rep + 1e-3))
                    if wanted and fits:
                        atoms.append((site.specie.symbol, *(f @ L).tolist()))
    pos = np.array([a[1:] for a in atoms])
    bonds = []
    for (e1, e2), cutoff in spec["bonds"].items():
        for a, b in cKDTree(pos).query_pairs(cutoff):
            if {atoms[a][0], atoms[b][0]} == {e1, e2}:
                bonds.append((a, b))

    unit = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    corners = rect[0] + unit * (rect[1] - rect[0]) if rect else (org + rep * unit) @ L
    np.savez_compressed(out / f"{label}.npz", verts=verts.astype("f4"), faces=faces.astype("i4"))
    (out / f"{label}.json").write_text(json.dumps(dict(
        name=spec["name"], atoms=atoms, bonds=bonds, corners=corners.tolist(),
        cell=None, void_spheres=void_centres if spec.get("whole") else None)))
    print("   surface", len(verts), "verts;", len(atoms), "atoms;", len(bonds), "bonds", flush=True)


def schematic_network(out, size=12.0, radius=1.05, steps=132):
    """3D panel: a simple connected rod lattice. A teaching schematic, not a real structure."""
    t = np.linspace(0, size, steps)
    x, y, z = np.meshgrid(t, t, t, indexing="ij")
    lines = (size / 4, 3 * size / 4)

    def rods(u, v):
        return radius - np.min([np.hypot(u - p, v - q) for p in lines for q in lines], axis=0)

    k = 3.0  # smooth union, so junctions blend
    field = np.log(np.exp(k * rods(y, z)) + np.exp(k * rods(x, z)) + np.exp(k * rods(x, y))) / k
    face = np.minimum.reduce([x, size - x, y, size - y, z, size - z])
    box = np.minimum(field, face - 1e-3)
    verts, faces, *_ = marching_cubes(np.pad(box, 1, constant_values=-1), 0.0)
    verts = (verts - 1) * (t[1] - t[0])
    points = (0, size / 2, size)
    atoms = [("Ba" if (i + j + k2) % 2 == 0 else "N", points[i], points[j], points[k2])
             for i in range(3) for j in range(3) for k2 in range(3)]
    corners = [[a, b, c] for a in (0, size) for b in (0, size) for c in (0, size)]
    np.savez_compressed(out / "3D.npz", verts=verts.astype("f4"), faces=faces.astype("i4"))
    (out / "3D.json").write_text(json.dumps(dict(name="Schematic network", atoms=atoms, bonds=[],
                                                 corners=corners, cell=None, void_spheres=None)))
    print("3D schematic:", len(verts), "verts;", len(atoms), "atoms", flush=True)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    for key, value in MATERIALS.items():
        build(key, value, target)
    schematic_network(target)
