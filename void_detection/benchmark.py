#!/usr/bin/env python3
"""Small ELFNet void-detection benchmark. Requires only Python and NumPy."""
from __future__ import annotations

import itertools
import json
import time
import argparse
from pathlib import Path

import numpy as np


SAMPLE = Path("/private/tmp/elfnet_sample/dft-reference-elfs-50000-v1/2/BaH6_Imm2_200/Li_Ir")
OUT = Path(__file__).parent


def read_poscar(path):
    lines = path.read_text().splitlines()
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    mode_line = 7
    if lines[mode_line].strip().lower().startswith("s"):
        mode_line += 1
    direct = lines[mode_line].strip().lower().startswith("d")
    coords = np.array([[float(x) for x in lines[mode_line + 1 + i].split()[:3]] for i in range(sum(counts))])
    frac = coords if direct else coords @ np.linalg.inv(lattice)
    species = [s for s, n in zip(symbols, counts) for _ in range(n)]
    return lattice, frac % 1, species


def read_elfcar(path):
    lines = path.read_text().splitlines()
    i = 5
    species_or_counts = lines[i].split()
    try:
        counts = [int(x) for x in species_or_counts]
        i += 1
    except ValueError:
        counts = [int(x) for x in lines[i + 1].split()]
        i += 2
    if lines[i].strip().lower().startswith("s"):
        i += 1
    i += 1 + sum(counts)
    while not lines[i].strip():
        i += 1
    dims = tuple(int(x) for x in lines[i].split()[:3])
    nx, ny, nz = dims
    need = nx * ny * nz
    vals = np.fromstring(" ".join(lines[i + 1:]), sep=" ", count=need, dtype=float)
    if vals.size != need:
        raise ValueError(f"ELFCAR grid truncated: got {vals.size}, expected {need}")
    elf = vals.reshape((nz, ny, nx)).transpose((2, 1, 0)).astype(np.float32)
    return elf, dims


def frac_distance(a, b, lattice):
    d = np.asarray(a) - np.asarray(b)
    return min(np.linalg.norm((d + s) @ lattice) for s in itertools.product((-1, 0, 1), repeat=3))


def all_grid_frac(shape):
    axes = [np.arange(n, dtype=float) / n for n in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def clearance_grid(fracgrid, atoms, lattice):
    out = np.full(fracgrid.shape[:-1], np.inf)
    for atom in atoms:
        d = fracgrid - atom
        best2 = np.full(out.shape, np.inf)
        for shift in itertools.product((-1, 0, 1), repeat=3):
            cart = (d + shift) @ lattice
            best2 = np.minimum(best2, np.einsum("...i,...i->...", cart, cart))
        out = np.minimum(out, np.sqrt(best2))
    return out


def grid_peaks(field, lattice, min_value=-np.inf, connectivity=26, merge_A=0.35):
    mask = field >= min_value
    greater = np.zeros(field.shape, dtype=bool)
    neighbor_offsets = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
    if connectivity == 6:
        neighbor_offsets = [o for o in neighbor_offsets if sum(abs(x) for x in o) == 1]
    for off in neighbor_offsets:
        neigh = np.roll(field, shift=tuple(-x for x in off), axis=(0, 1, 2))
        mask &= field >= neigh
        greater |= field > neigh
    points = np.argwhere(mask & greater)
    if len(points) == 0:
        return []
    dims = np.array(field.shape)
    order = sorted(points, key=lambda p: (-float(field[tuple(p)]), tuple(p)))
    accepted = []
    for p in order:
        f = p / dims
        if all(frac_distance(f, q["frac"], lattice) >= merge_A for q in accepted):
            accepted.append({"frac": f, "value": float(field[tuple(p)]), "grid": tuple(int(x) for x in p)})
    return accepted


def empty_sphere_vertices(lattice, atoms, radius_cut=0.0):
    """Brute-force periodic Delaunay circumcenters for a small unit cell."""
    shifts = list(itertools.product((-1, 0, 1), repeat=3))
    pts = np.array([(a + s) @ lattice for a in atoms for s in shifts])
    centers = []
    for inds in itertools.combinations(range(len(pts)), 4):
        p = pts[list(inds)]
        mat = 2 * (p[1:] - p[0])
        try:
            c = np.linalg.solve(mat, np.sum(p[1:] ** 2, axis=1) - np.sum(p[0] ** 2))
        except np.linalg.LinAlgError:
            continue
        raw_frac = c @ np.linalg.inv(lattice)
        if np.any(raw_frac < -1e-8) or np.any(raw_frac >= 1.0 - 1e-8):
            continue
        f = raw_frac % 1.0
        d = np.linalg.norm(pts - c, axis=1)
        r = float(d.min())
        if r < radius_cut:
            continue
        # Retain true empty spheres; each relevant sphere has at least four
        # periodic image sites on its boundary. Tolerance handles roundoff.
        if np.any(d < r - 2e-5) or np.count_nonzero(d <= r + 2e-5) < 4:
            continue
        if not any(frac_distance(f, old["frac"], lattice) < 0.03 for old in centers):
            centers.append({"frac": f, "value": float(r), "grid": None})
    return sorted(centers, key=lambda c: -c["value"])


def clearance_one(frac, atoms, lattice):
    return min(frac_distance(frac, atom, lattice) for atom in atoms)


def refine_clearance_seeds(candidates, atoms, lattice, initial_step, merge_A=0.35):
    """Pattern-search continuous clearance maxima from sampled grid seeds."""
    dirs = [np.array(o, dtype=float) for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
    refined = []
    for c in candidates:
        f = np.array(c["frac"], dtype=float).copy()
        value = clearance_one(f, atoms, lattice)
        step = float(initial_step)
        for _ in range(100):
            best_f, best_value = f, value
            for direction in dirs:
                trial = (f + step * direction) % 1.0
                trial_value = clearance_one(trial, atoms, lattice)
                if trial_value > best_value + 1e-10:
                    best_f, best_value = trial, trial_value
            if best_value > value + 1e-10:
                f, value = best_f, best_value
            else:
                step *= 0.5
                if step * max(np.linalg.norm(v) for v in lattice) < 1e-3:
                    break
        if not any(frac_distance(f, old["frac"], lattice) < merge_A for old in refined):
            refined.append({"frac": f, "value": float(value), "grid": c.get("grid")})
    return sorted(refined, key=lambda c: -c["value"])


def evaluate(candidates, targets, lattice, tolerances=(0.3, 0.5, 0.75, 1.0), budgets=(5, 10, 20)):
    result = {"candidate_count": len(candidates)}

    def one_to_one_precision(items, tol):
        unmatched = set(range(len(targets)))
        matched = 0
        for c in items:
            nearest = min(unmatched, key=lambda j: frac_distance(c["frac"], targets[j]["frac"], lattice), default=None)
            if nearest is not None and frac_distance(c["frac"], targets[nearest]["frac"], lattice) <= tol:
                unmatched.remove(nearest)
                matched += 1
        return matched / len(items) if items else None

    for tol in tolerances:
        result[f"recall_all_at_{tol:.2f}A"] = (sum(any(frac_distance(c["frac"], t["frac"], lattice) <= tol for c in candidates) for t in targets) / len(targets)) if targets else None
        result[f"precision_all_at_{tol:.2f}A"] = one_to_one_precision(candidates, tol)
        for k in budgets:
            top = candidates[:k]
            result[f"recall_top{k}_at_{tol:.2f}A"] = (sum(any(frac_distance(c["frac"], t["frac"], lattice) <= tol for c in top) for t in targets) / len(targets)) if targets else None
            result[f"precision_top{k}_at_{tol:.2f}A"] = one_to_one_precision(top, tol)
    result["mean_candidate_nearest_target_A"] = (float(np.mean([min(frac_distance(c["frac"], t["frac"], lattice) for t in targets) for c in candidates])) if targets and candidates else None)
    return result


def synthetic_benchmarks():
    cases = [
        {"name": "simple_cubic", "lattice": np.eye(3) * 4.0, "atoms": np.array([[0.0, 0.0, 0.0]]), "targets": np.array([[0.5, 0.5, 0.5]])},
        {"name": "bcc", "lattice": np.eye(3) * 4.0, "atoms": np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]), "targets": np.array([[0.25, 0.5, 0.0], [0.75, 0.5, 0.0], [0.5, 0.25, 0.0], [0.5, 0.75, 0.0], [0.5, 0.0, 0.25], [0.5, 0.0, 0.75], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.25, 0.0, 0.5], [0.75, 0.0, 0.5], [0.0, 0.25, 0.5], [0.0, 0.75, 0.5]])},
        {"name": "fcc", "lattice": np.eye(3) * 4.0, "atoms": np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]), "targets": np.array([[0.5, 0.5, 0.5], *[[x, y, z] for x, y, z in itertools.product((0.25, 0.75), repeat=3)]])},
        {"name": "skewed_triclinic_one_atom", "lattice": np.array([[4.0, 0.0, 0.0], [1.2, 3.8, 0.0], [0.6, 0.4, 4.0]]), "atoms": np.array([[0.0, 0.0, 0.0]]), "targets": None},
    ]
    outputs = []
    for case in cases:
        lattice, atoms, targets = case["lattice"], case["atoms"], case["targets"]
        reference_vertices = None
        if targets is None:
            reference_vertices = empty_sphere_vertices(lattice, atoms, radius_cut=0.2)
            targets = np.array([v["frac"] for v in reference_vertices])
        grid = all_grid_frac((24, 24, 24))
        field = clearance_grid(grid, atoms, lattice)
        entry = {"name": case["name"], "atom_count": len(atoms), "known_target_count": len(targets), "methods": {}}
        for conn in (6, 26):
            for merge in (0.3, 0.55, 0.8):
                cand = grid_peaks(field, lattice, connectivity=conn, merge_A=merge)
                cand.sort(key=lambda c: -c["value"])
                entry["methods"][f"grid_{conn}conn_merge{merge:.2f}A"] = evaluate(cand, [{"frac": t} for t in targets], lattice)
        seed = grid_peaks(field, lattice, connectivity=26, merge_A=0.3)
        refined = refine_clearance_seeds(seed, atoms, lattice, initial_step=1 / 24, merge_A=0.3)
        entry["methods"]["grid26_then_continuous_clearance_refinement"] = evaluate(refined, [{"frac": t} for t in targets], lattice)
        if case["name"] in ("simple_cubic", "bcc"):
            t0 = time.perf_counter()
            vor = empty_sphere_vertices(lattice, atoms, radius_cut=0.2)
            dt = time.perf_counter() - t0
            entry["methods"]["empty_sphere_vertices"] = {**evaluate(vor, [{"frac": t} for t in targets], lattice), "runtime_s": dt}
        if reference_vertices is not None:
            entry["reference_method"] = "independently computed periodic empty-sphere vertices (shared Voronoi/Delaunay family; not an independent physical label)"
            entry["methods"]["empty_sphere_vertices"] = evaluate(reference_vertices, [{"frac": t} for t in targets], lattice)
        entry["resolution_sweep"] = []
        for n in (12, 16, 24, 32, 48):
            local_grid = all_grid_frac((n, n, n))
            local_clear = clearance_grid(local_grid, atoms, lattice)
            cand = grid_peaks(local_clear, lattice, connectivity=26, merge_A=0.3)
            cand.sort(key=lambda c: -c["value"])
            refined = refine_clearance_seeds(cand, atoms, lattice, initial_step=1 / n, merge_A=0.3)
            entry["resolution_sweep"].append({"grid": [n, n, n], **evaluate(cand, [{"frac": t} for t in targets], lattice, tolerances=(0.2, 0.3, 0.5), budgets=(5, 10)), "refined_candidate_count": len(refined), "refined_recall_all_at_0.20A": evaluate(refined, [{"frac": t} for t in targets], lattice, tolerances=(0.2,), budgets=())["recall_all_at_0.20A"], "refined_precision_all_at_0.20A": evaluate(refined, [{"frac": t} for t in targets], lattice, tolerances=(0.2,), budgets=())["precision_all_at_0.20A"]})
        outputs.append(entry)
    return outputs


def pressure_scale_sweep():
    """Check whether absolute clearance cutoffs survive isotropic compression."""
    lattice0 = np.eye(3) * 4.0
    atoms = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    targets = np.array([[0.25, 0.5, 0.0], [0.75, 0.5, 0.0], [0.5, 0.25, 0.0], [0.5, 0.75, 0.0], [0.5, 0.0, 0.25], [0.5, 0.0, 0.75], [0.0, 0.5, 0.25], [0.0, 0.5, 0.75], [0.25, 0.0, 0.5], [0.75, 0.0, 0.5], [0.0, 0.25, 0.5], [0.0, 0.75, 0.5]])
    rows = []
    for scale in (1.0, 0.75, 0.5, 0.375, 0.25):
        lattice = lattice0 * scale
        field = clearance_grid(all_grid_frac((24, 24, 24)), atoms, lattice)
        row = {"scale_from_4A": scale, "cell_edge_A": 4 * scale, "expected_bcc_tetrahedral_clearance_A": float(np.sqrt(5) * scale), "thresholds": []}
        for cutoff in (0.0, 0.5, 0.8, 1.0):
            cand = grid_peaks(field, lattice, min_value=cutoff, connectivity=26, merge_A=0.3)
            cand.sort(key=lambda c: -c["value"])
            row["thresholds"].append({"absolute_cutoff_A": cutoff, **evaluate(cand, [{"frac": t} for t in targets], lattice, tolerances=(0.3,), budgets=(5, 10))})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=SAMPLE, help="Directory containing POSCAR and ELFCAR")
    args = parser.parse_args()
    sample = args.sample
    lattice, atoms, species = read_poscar(sample / "POSCAR")
    elf, dims = read_elfcar(sample / "ELFCAR")
    grid = all_grid_frac(dims)
    clear = clearance_grid(grid, atoms, lattice)
    atom_exclusion = 0.8
    elf_threshold_default = 0.35
    elf_targets_all = grid_peaks(elf, lattice, min_value=elf_threshold_default, merge_A=0.35)
    elf_targets = [p for p in elf_targets_all if min(frac_distance(p["frac"], a, lattice) for a in atoms) >= atom_exclusion]
    methods = {}
    for conn in (6, 26):
        for cutoff in (0.0, 0.5, 0.8, 1.0):
            cand = grid_peaks(clear, lattice, min_value=cutoff, connectivity=conn, merge_A=0.35)
            cand.sort(key=lambda c: -c["value"])
            methods[f"clearance_grid_{conn}conn_r{cutoff:.1f}A"] = evaluate(cand, elf_targets, lattice)
    for merge in (0.35, 0.5, 0.75, 1.0):
        cand = grid_peaks(clear, lattice, min_value=0.5, connectivity=26, merge_A=merge)
        cand.sort(key=lambda c: -c["value"])
        methods[f"clearance_grid_26conn_merge{merge:.2f}A"] = evaluate(cand, elf_targets, lattice)
    refine_start = time.perf_counter()
    seed_candidates = grid_peaks(clear, lattice, min_value=0.5, connectivity=26, merge_A=0.35)
    refined_candidates = refine_clearance_seeds(seed_candidates, atoms, lattice, initial_step=1 / max(dims), merge_A=0.35)
    methods["clearance_grid_then_continuous_refinement"] = {**evaluate(refined_candidates, elf_targets, lattice), "runtime_s": time.perf_counter() - refine_start}
    t0 = time.perf_counter()
    vor = empty_sphere_vertices(lattice, atoms, radius_cut=0.5)
    vor_time = time.perf_counter() - t0
    methods["periodic_empty_sphere_vertices_r0.5A"] = evaluate(vor, elf_targets, lattice)
    methods["periodic_empty_sphere_vertices_r0.5A"]["runtime_s"] = vor_time
    elf_oracle = sorted(elf_targets, key=lambda p: -p["value"])
    methods["ELF_peak_oracle"] = evaluate(elf_oracle, elf_targets, lattice)

    # Sweep target definition to show how the apparent score depends on what
    # is called an interstitial ELF attractor.
    sensitivity = []
    for threshold in (0.30, 0.35, 0.40, 0.45, 0.50):
        for exclusion in (0.8, 1.0, 1.2):
            peaks = grid_peaks(elf, lattice, min_value=threshold, merge_A=0.35)
            targets = [p for p in peaks if min(frac_distance(p["frac"], a, lattice) for a in atoms) >= exclusion]
            one = grid_peaks(clear, lattice, min_value=0.5, connectivity=26, merge_A=0.35)
            one.sort(key=lambda c: -c["value"])
            sensitivity.append({"elf_threshold": threshold, "atom_exclusion_A": exclusion, "target_count": len(targets), **evaluate(one, targets, lattice)})

    report = {
        "sample": str(sample),
        "structure": {"species": species, "atom_count": len(atoms), "lattice_A": lattice.tolist(), "grid_shape": dims, "grid_spacing_A_approx": [float(np.linalg.norm(lattice[i]) / dims[i]) for i in range(3)]},
        "labels": {"definition": f"ELF grid local maxima >={elf_threshold_default:.2f}, excluding peaks within 0.8 A of any atom; these are ELF features, not excess-electron-density labels", "all_grid_elf_peak_count_ge_threshold": len(elf_targets_all), "elf_range": [float(elf.min()), float(elf.max())], "target_count": len(elf_targets), "all_peaks": [{"frac": p["frac"].tolist(), "elf": p["value"], "nearest_atom_A": min(frac_distance(p["frac"], a, lattice) for a in atoms)} for p in grid_peaks(elf, lattice, min_value=-1, merge_A=0.35)], "targets": [{"frac": p["frac"].tolist(), "elf": p["value"], "nearest_atom_A": min(frac_distance(p["frac"], a, lattice) for a in atoms)} for p in elf_targets]},
        "methods": methods,
        "target_sensitivity": sensitivity,
        "synthetic_geometry_benchmarks": synthetic_benchmarks(),
        "bcc_isotropic_compression_sweep": pressure_scale_sweep(),
        "elf_target_nearest_candidate": [{"target_frac": p["frac"].tolist(), "elf": p["value"], "nearest_clearance_candidate_A": min(frac_distance(p["frac"], c["frac"], lattice) for c in grid_peaks(clear, lattice, min_value=0.5, connectivity=26, merge_A=0.35)), "within_top5_A": min(frac_distance(p["frac"], c["frac"], lattice) for c in sorted(grid_peaks(clear, lattice, min_value=0.5, connectivity=26, merge_A=0.35), key=lambda c: -c["value"])[:5])} for p in elf_targets],
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Void detection experiments (exploratory)", "", "## Scope", "", "Single-structure prototype using the extracted ELFNet DFT-reference example `BaH6_Imm2_200/Li_Ir` (2 atoms and a 40 x 24 x 24 ELF grid), plus analytical periodic simple-cubic, BCC, FCC, and skewed triclinic cells. The small set makes this a smoke test, not a general performance estimate. ELF maxima are not equivalent to excess electron density; without charge density, labels only describe ELF features.", "", "## Methods and variations tested", "", "- Periodic nearest-atom clearance sampled on a grid, then 6- or 26-neighbor maxima detection; absolute clearance cutoffs 0.0/0.5/0.8/1.0 A; duplicate-merging radii 0.3/0.55/0.8 A; grid resolutions 12^3 to 48^3; and continuous pattern-search refinement from grid seeds.", "- Periodic empty-sphere centers from atom-image quadruplets (a brute-force Delaunay/Voronoi-family baseline) on small cells.", "- ELF-grid local maxima after nuclear exclusion, used as a reference/upper-bound method rather than a structure-only predictor.", "- Analytical SC/BCC/FCC targets; a skewed-cell stress test; and a BCC isotropic cell-scaling sweep to test whether absolute clearance cutoffs survive compression.", "", "## Targets and metrics", "", "For the ELFNet example, default reference targets are local ELF-grid maxima >=0.35, merged within 0.35 A, and farther than 0.8 A from nuclei. This low exploratory threshold captures its weak interstitial peaks; target count is swept over ELF thresholds 0.30–0.50 and nucleus exclusions 0.8/1.0/1.2 A. Synthetic targets are known high-symmetry void sites. Recall, one-to-one precision, candidate budgets/counts, periodic Cartesian match tolerances, and resolutions are saved in `results.json`.", "", "## Main observations", "", "- On the ideal SC and BCC cells, 26-neighbor clearance maxima recovered all specified sites with 1 and 12 candidates, respectively. FCC recall was complete but precision was 0.75 because 12 candidates covered 9 specified sites. At these high-symmetry settings, 6-neighbor detection produced many extra maxima (e.g. 212 vs 12 on BCC).", "- On the skewed triclinic cell, grid candidate recall varied non-monotonically with resolution and missed some reference vertices. Pattern-search refinement helped at one resolution but did not behave monotonically, so the grid method still needs better skew-cell testing and robust geometric refinement.", "- For the ELFNet sample, the single largest ELF peak is only 0.046 A from a nucleus. With the 0.35 threshold and 0.8 A nuclear exclusion there are five weak interstitial peaks (ELF 0.367–0.458). The 24-candidate clearance grid matched 4/5 within 1.0 A but none within 0.5 A; its top five matched 3/5 within 1.0 A. Wider merge tolerance reduced count to 6 but also reduced 1-A recall to 3/5.", "- Raising ELF threshold to 0.40 leaves two targets, both matched within 1 A; at 0.50 there are no interstitial targets. ELF threshold is therefore a label choice, not an innocuous setting.", "- In the BCC scaling sweep, a fixed 0.8 A cutoff removed all targets at a 1.0 A cell edge; a 1.0 A cutoff removed them at a 1.5 A edge. Preserve continuous clearance as a feature and avoid unvalidated absolute filters in compressed structures.", "- Empty-sphere enumeration found candidates on these tiny cells, but degenerate point sets produce duplicate/unstable sites. The project already has `dft/cp2k/find_voids.py`; it was not executed here because this runtime lacks its SciPy/spglib dependencies and the sample is POSCAR rather than CIF. Use that established implementation after setting up its environment for broader tests.", "", "## Limitations and next step", "", "This only tests one ELFNet structure and four idealized geometries. A matching ELF maximum is not proof of electride character. Expand to a diverse extracted subset, include low-symmetry and pressure-related distortions, and group evaluation by parent structure to avoid leakage. Use actual charge-density data if the target is integrated excess basin charge; ELF grids alone cannot provide it.", "", "## Reproduce", "", "Run `python benchmark.py --sample /path/to/ELFNet/sample` with NumPy installed. The default points to `/private/tmp/elfnet_sample/...`; the `--sample` option lets you select a stable extracted path. The benchmark writes `results.json` alongside this file.", ""]
    (OUT / "README.md").write_text("\n".join(lines))
    print(json.dumps({"targets": len(elf_targets), "methods": {k: {"candidates": v["candidate_count"], "recall_0.5A": v["recall_all_at_0.50A"], "recall_1.0A": v["recall_all_at_1.00A"]} for k, v in methods.items()}, "voronoi_s": vor_time}, indent=2))


if __name__ == "__main__":
    main()
