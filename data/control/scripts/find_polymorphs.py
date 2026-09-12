"""Find same-composition structures in Materials Project.

This script reads the inorganic electride registry, queries Materials Project
once for each unique normalized formula, groups equivalent returned structures
with pymatgen StructureMatcher, and writes the results to:

    data/control/polymorphs/polymorphs.csv
    data/control/polymorphs/polymorph_failures.json

Fill in MP_API_KEY before running. A separate three-minute timeout is applied
to each formula so one slow query or comparison does not stop the full run.
"""

from __future__ import annotations

print("[startup] Starting polymorph search script...", flush=True)

import csv
import json
import os
import signal
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher

print("[startup] Materials-science libraries imported.", flush=True)


TIMEOUT_SECONDS = 3 * 60

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
REGISTRY_PATH = DATA_DIR / "electrides" / "inorganic_electride_registry.csv"
OUTPUT_DIR = DATA_DIR / "control" / "polymorphs"
OUTPUT_CSV = OUTPUT_DIR / "polymorphs.csv"
OUTPUT_FAILURES = OUTPUT_DIR / "polymorph_failures.json"

load_dotenv(PROJECT_ROOT / ".env")
MP_API_KEY = os.getenv("MP_API_KEY", "").strip()


class FormulaTimeout(Exception):
    """Raised when processing one formula exceeds the time limit."""


def timeout_handler(signum: int, frame: Any) -> None:
    raise FormulaTimeout(f"Exceeded {TIMEOUT_SECONDS} seconds")


def load_registry() -> tuple[list[str], dict[str, dict[str, set[str]]]]:
    """Return unique formulas and registry provenance grouped by formula."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    formulas: list[str] = []
    seen_formulas: set[str] = set()
    provenance: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            formula = (row.get("normalized_formula") or "").strip()
            if not formula:
                continue

            if formula not in seen_formulas:
                formulas.append(formula)
                seen_formulas.add(formula)

            status = (row.get("electride_status") or "not reported").strip()
            material_id = (row.get("material_id") or "not reported").strip()
            provenance[formula]["statuses"].add(status)
            provenance[formula]["material_ids"].add(material_id)

    return formulas, provenance


def structure_groups(
    structures: list[Any],
    matcher: StructureMatcher,
    formula: str,
) -> list[int]:
    """Group equivalent structures using pairwise StructureMatcher comparisons."""
    parent = list(range(len(structures)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(index_a: int, index_b: int) -> None:
        root_a = find(index_a)
        root_b = find(index_b)
        if root_a != root_b:
            parent[root_b] = root_a

    total_comparisons = len(structures) * (len(structures) - 1) // 2
    comparison_number = 0
    comparison_start = time.perf_counter()

    for i in range(len(structures)):
        for j in range(i + 1, len(structures)):
            comparison_number += 1

            if matcher.fit(structures[i], structures[j]):
                union(i, j)

            if (
                comparison_number == 1
                or comparison_number % 100 == 0
                or comparison_number == total_comparisons
            ):
                elapsed = time.perf_counter() - comparison_start
                print(
                    f"  {formula}: structure comparison "
                    f"{comparison_number}/{total_comparisons} "
                    f"({elapsed:.1f} s)",
                    flush=True,
                )

    root_to_group: dict[int, int] = {}
    groups: list[int] = []

    for i in range(len(structures)):
        root = find(i)
        if root not in root_to_group:
            root_to_group[root] = len(root_to_group) + 1
        groups.append(root_to_group[root])

    return groups


def process_formula(
    formula: str,
    provenance: dict[str, dict[str, set[str]]],
    matcher: StructureMatcher,
) -> list[dict[str, Any]]:
    """Retrieve and group all Materials Project structures for one formula."""
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "database_IDs",
        "volume",
        "nsites",
    ]

    with MPRester(api_key=MP_API_KEY, mute_progress_bars=True) as mpr:
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=fields,
        )

    if not docs:
        return []

    structures = [doc.structure for doc in docs]
    groups = structure_groups(structures, matcher, formula)
    group_sizes = defaultdict(int)
    for group in groups:
        group_sizes[group] += 1

    statuses = "|".join(sorted(provenance[formula]["statuses"]))
    source_ids = "|".join(sorted(provenance[formula]["material_ids"]))
    output_rows: list[dict[str, Any]] = []

    for index, doc in enumerate(docs):
        symmetry = doc.symmetry
        group = groups[index]

        output_rows.append({
            "source_formula": formula,
            "source_registry_statuses": statuses,
            "source_registry_material_ids": source_ids,
            "mp_material_id": str(doc.material_id),
            "mp_formula": doc.formula_pretty,
            "space_group": symmetry.symbol if symmetry else "not reported",
            "space_group_number": symmetry.number if symmetry else "not reported",
            "crystal_system": symmetry.crystal_system if symmetry else "not reported",
            "energy_above_hull_eV_per_atom": doc.energy_above_hull,
            "formation_energy_per_atom_eV": doc.formation_energy_per_atom,
            "volume_A3": doc.volume,
            "number_of_sites": doc.nsites,
            "structure_group": group,
            "structures_in_group": group_sizes[group],
            "control_status": "control_candidate",
        })

    return output_rows


def main() -> None:
    if not MP_API_KEY.strip():
        raise ValueError(
            "MP_API_KEY is empty. Add your Materials Project API key to the script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    formulas, provenance = load_registry()
    print(
        f"[startup] Loaded {len(formulas)} unique formulas from {REGISTRY_PATH}",
        flush=True,
    )

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )

    print("[startup] Structure matcher ready.", flush=True)

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        for number, formula in enumerate(formulas, start=1):
            print(f"[{number}/{len(formulas)}] Processing {formula}", flush=True)
            start = time.perf_counter()
            signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SECONDS)

            try:
                rows = process_formula(formula, provenance, matcher)
                all_rows.extend(rows)
                elapsed = time.perf_counter() - start
                print(
                    f"  completed in {elapsed:.1f} s; "
                    f"{len(rows)} Materials Project entries",
                    flush=True,
                )
            except FormulaTimeout as exc:
                elapsed = time.perf_counter() - start
                failure = {
                    "formula": formula,
                    "reason": "timeout",
                    "message": str(exc),
                    "elapsed_seconds": round(elapsed, 2),
                }
                failures.append(failure)
                print(f"  FAILED: {formula} exceeded 3 minutes", flush=True)
            except Exception as exc:  # Keep the full batch running.
                elapsed = time.perf_counter() - start
                failure = {
                    "formula": formula,
                    "reason": "error",
                    "message": repr(exc),
                    "elapsed_seconds": round(elapsed, 2),
                }
                failures.append(failure)
                print(f"  FAILED: {formula}: {exc}", flush=True)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)

    output_fields = [
        "source_formula",
        "source_registry_statuses",
        "source_registry_material_ids",
        "mp_material_id",
        "mp_formula",
        "space_group",
        "space_group_number",
        "crystal_system",
        "energy_above_hull_eV_per_atom",
        "formation_energy_per_atom_eV",
        "volume_A3",
        "number_of_sites",
        "structure_group",
        "structures_in_group",
        "control_status",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    failure_report = {
        "timeout_seconds_per_formula": TIMEOUT_SECONDS,
        "total_unique_formulas": len(formulas),
        "completed_formula_count": len(formulas) - len(failures),
        "failure_count": len(failures),
        "failures": failures,
    }

    with OUTPUT_FAILURES.open("w", encoding="utf-8") as handle:
        json.dump(failure_report, handle, indent=2)
        handle.write("\n")

    print(f"\nWrote {len(all_rows)} rows to {OUTPUT_CSV}")
    print(f"Wrote failure report to {OUTPUT_FAILURES}")


if __name__ == "__main__":
    main()
