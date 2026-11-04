#!/usr/bin/env python3
"""Score power-diagram void dimensionality against literature electride labels.

Each registry electride with one literature dimensionality (0D-3D), a
Materials Project CIF and a resolved structure match is run through
``PowerSpace`` over a sweep of probe radii, for ionic radii and for the
uniform atom-center cutoff. The prediction is the highest dimension of any
open region (-1 when nothing is open). Literature labels are electronic
(mostly DFT ELF screens); these predictions are geometric.

The three case studies used to choose the default probes are excluded from
scoring. Balanced accuracy is the mean recall over 0D, 1D and 2D; the 3D class
is too small to average in but still counts toward exact accuracy.
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .power_void_connectivity import PowerSpace, element_radii


CALIBRATION_MP_IDS = ("mp-1019788", "mp-12905", "mp-2686")  # K4Al3(SiO4)3, Ba3CrN3, Ca2N
SCHEMES = {
    "ionic": {"probes": np.round(np.arange(0.40, 1.801, 0.05), 2), "default": 1.0},
    "none": {"probes": np.round(np.arange(1.40, 3.201, 0.05), 2), "default": 2.0},
}
BALANCED_CLASSES = (0, 1, 2)
LABELS = ("0D", "1D", "2D", "3D")


def load_test_set(registry_csv: Path, cif_dir: Path) -> pd.DataFrame:
    """One row per Materials Project structure with its literature labels."""
    registry = pd.read_csv(registry_csv)
    rows = registry[registry["electride_dimensionality"].isin(LABELS)
                    & registry["mp_id"].astype(str).str.startswith("mp-")
                    & (registry["structure_identity_status"] != "unresolved")]
    rows = rows[[(cif_dir / f"{mp_id}.cif").exists() for mp_id in rows["mp_id"]]]
    table = rows.groupby("mp_id").agg(
        formula=("normalized_formula", "first"),
        literature_labels=("electride_dimensionality", lambda s: ";".join(sorted(set(s)))),
        electride_status=("electride_status", lambda s: ";".join(sorted(set(s)))),
        primary_reference=("primary_reference", "first"),
        registry_ids=("material_id", lambda s: ";".join(sorted(set(s)))),
    ).reset_index()
    single = ~table["literature_labels"].str.contains(";")
    table["literature_dimension"] = np.where(single, table["literature_labels"].str[0], "")
    table["literature_dimension"] = pd.to_numeric(table["literature_dimension"], errors="coerce").astype("Int64")
    table["calibration"] = table["mp_id"].isin(CALIBRATION_MP_IDS)
    table["scored"] = single & ~table["calibration"]
    return table


def highest_dimensions(cif: Path, scheme: str, probes: np.ndarray) -> tuple[list[int], bool]:
    """Highest open-region dimension at each probe, and whether a radius fell back to atomic."""
    from pymatgen.core import Structure
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = Structure.from_file(str(cif))
    structure.remove_oxidation_states()
    symbols = [site.specie.symbol for site in structure]
    radii, notes = element_radii(symbols, scheme)
    base = np.array([radii[s] for s in symbols])
    dims = []
    for probe in probes:
        space = PowerSpace(structure.lattice.matrix, structure.frac_coords, base + probe)
        dims.append(max((r["dimension"] for r in space.explore()["regions"]), default=-1))
    return dims, any("empirical" in note for note in notes.values())


def _sweep_one(task: tuple[str, Path]) -> list[dict[str, Any]]:
    mp_id, cif = task
    rows = []
    for scheme, spec in SCHEMES.items():
        dims, fallback = highest_dimensions(cif, scheme, spec["probes"])
        rows += [{"mp_id": mp_id, "radius_scheme": scheme, "probe_radius_angstrom": float(p),
                  "highest_dimension": d, "atomic_radius_fallback": fallback}
                 for p, d in zip(spec["probes"], dims)]
    return rows


def score(literature: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """Exact accuracy, per-class recall and balanced accuracy over 0D-2D."""
    literature, predicted = np.asarray(literature, dtype=int), np.asarray(predicted, dtype=int)
    recall = {f"{k}D": float(np.mean(predicted[literature == k] == k))
              for k in range(4) if np.any(literature == k)}
    return {"n": int(len(literature)),
            "exact_accuracy": float(np.mean(predicted == literature)),
            "balanced_accuracy": float(np.mean([recall[f"{k}D"] for k in BALANCED_CLASSES])),
            "recall": recall}


def half_split_tuning(literature: np.ndarray, sweep: np.ndarray, repeats: int = 200,
                      seed: int = 0) -> dict[str, float]:
    """Pick the best probe on a random half, score it on the other half."""
    rng = np.random.default_rng(seed)
    held_out = []
    for _ in range(repeats):
        order = rng.permutation(len(literature))
        tune, test = order[: len(order) // 2], order[len(order) // 2:]
        best = int(np.argmax((sweep[tune] == literature[tune, None]).mean(0)))
        held_out.append(float(np.mean(sweep[test, best] == literature[test])))
    return {"repeats": repeats, "held_out_exact_accuracy_mean": float(np.mean(held_out)),
            "held_out_exact_accuracy_std": float(np.std(held_out))}


def summarize(table: pd.DataFrame, sweep: pd.DataFrame) -> dict[str, Any]:
    scored = table[table["scored"]].set_index("mp_id")
    literature = scored["literature_dimension"].to_numpy(dtype=int)
    summary: dict[str, Any] = {
        "structures": int(len(table)), "scored": int(len(scored)),
        "excluded_conflicting_labels": int(table["literature_labels"].str.contains(";").sum()),
        "excluded_calibration": sorted(table.loc[table["calibration"], "mp_id"]),
        "literature_counts": {f"{k}D": int(np.sum(literature == k)) for k in range(4)},
        "experimentally_confirmed": int(scored["electride_status"].str.contains("experimentally_confirmed").sum()),
        "all_0D_baseline_exact_accuracy": float(np.mean(literature == 0)),
        "chance_balanced_accuracy": 1 / len(BALANCED_CLASSES),
        "schemes": {},
    }
    for scheme, spec in SCHEMES.items():
        wide = (sweep[sweep["radius_scheme"] == scheme]
                .pivot(index="mp_id", columns="probe_radius_angstrom", values="highest_dimension")
                .loc[scored.index])
        probes = [float(p) for p in wide.columns]
        matrix = wide.to_numpy(dtype=int)
        curve = [score(literature, matrix[:, j]) for j in range(len(probes))]
        default = probes.index(spec["default"])
        best = int(np.argmax([c["exact_accuracy"] for c in curve]))
        predicted = matrix[:, default]
        confusion = pd.crosstab(pd.Series(literature, name="literature"),
                                pd.Series(predicted, name="predicted")).reindex(
            index=range(4), columns=range(-1, 4), fill_value=0)
        summary["schemes"][scheme] = {
            "default_probe_angstrom": spec["default"], "default": curve[default],
            "best_exact_probe_angstrom": probes[best], "best_exact": curve[best],
            "literature_dimension_reached_at_some_probe": float(np.mean((matrix == literature[:, None]).any(1))),
            "half_split_tuning": half_split_tuning(literature, matrix),
            "confusion_at_default": {"rows_literature": list(LABELS), "columns_predicted": ["none", *LABELS],
                                     "counts": confusion.to_numpy().tolist()},
            "curve": [{"probe_angstrom": p, **c} for p, c in zip(probes, curve)],
        }
    return summary


def run(registry_csv: Path, cif_dir: Path, output: Path, workers: int = 8,
        figures: bool = True) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    table = load_test_set(registry_csv, cif_dir)
    tasks = [(m, cif_dir / f"{m}.cif") for m in table["mp_id"]]
    with Pool(workers) as pool:
        sweep = pd.DataFrame([row for rows in pool.map(_sweep_one, tasks) for row in rows])
    fallback = sweep[sweep["radius_scheme"] == "ionic"].groupby("mp_id")["atomic_radius_fallback"].first()
    table["ionic_atomic_radius_fallback"] = table["mp_id"].map(fallback)
    for scheme, spec in SCHEMES.items():
        at_default = sweep[(sweep["radius_scheme"] == scheme) & (sweep["probe_radius_angstrom"] == spec["default"])]
        table[f"predicted_{scheme}_{spec['default']:.2f}"] = table["mp_id"].map(
            at_default.set_index("mp_id")["highest_dimension"])
    summary = summarize(table, sweep)
    summary["settings"] = {"registry": str(registry_csv), "cif_dir": str(cif_dir),
                           "probes": {s: [float(p) for p in spec["probes"]] for s, spec in SCHEMES.items()}}
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "structures.csv", index=False)
    sweep.drop(columns="atomic_radius_fallback").to_csv(output / "probe_sweep.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if figures:
        from .registry_dimensionality_figures import plot_all
        plot_all(summary, output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--registry", type=Path,
                        default=Path("data/electrides/final_dataset/inorganic_electride_registry_matched.csv"))
    parser.add_argument("--cif-dir", type=Path, default=Path("data/cifs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    summary = run(args.registry, args.cif_dir, args.output, workers=args.workers, figures=not args.no_figures)
    for scheme, result in summary["schemes"].items():
        d = result["default"]
        print(f"{scheme} @ {result['default_probe_angstrom']} Å: exact {d['exact_accuracy']:.1%}, "
              f"balanced {d['balanced_accuracy']:.1%} (n={d['n']})")


if __name__ == "__main__":
    main()
