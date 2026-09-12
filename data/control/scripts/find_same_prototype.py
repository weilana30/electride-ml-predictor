"""Find chemically different same-prototype control candidates.

The script reads the inorganic electride registry, retrieves candidate seed
structures and related structures from Materials Project, compares structures
with anonymous StructureMatcher matching, and writes one deduplicated row per
Materials Project candidate:

    data/control/same_prototype/same_prototype.csv
    data/control/same_prototype/same_prototype_failures.json

The API key is loaded from the project-level .env file. A three-minute timeout
is applied independently to each registry seed group.
"""

from __future__ import annotations

print("[startup] Starting same-prototype search...", flush=True)

import csv
import json
import os
import re
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
OUTPUT_DIR = DATA_DIR / "control" / "same_prototype"
OUTPUT_CSV = OUTPUT_DIR / "same_prototype.csv"
OUTPUT_FAILURES = OUTPUT_DIR / "same_prototype_failures.json"

load_dotenv(PROJECT_ROOT / ".env")
MP_API_KEY = os.getenv("MP_API_KEY", "").strip()


class SeedTimeout(Exception):
    """Raised when processing one registry seed exceeds the time limit."""


def timeout_handler(signum: int, frame: Any) -> None:
    raise SeedTimeout(f"Exceeded {TIMEOUT_SECONDS} seconds")


def clean(value: Any, default: str = "not reported") -> str:
    """Convert missing values to consistent strings."""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def load_seed_groups() -> tuple[list[dict[str, Any]], set[str]]:
    """Load unique formula/space-group/prototype seed groups."""
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    all_formulas: set[str] = set()

    with REGISTRY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            formula = clean(row.get("normalized_formula"), default="")
            if not formula:
                continue

            space_group = clean(row.get("space_group"))
            prototype = clean(row.get("crystal_structure_or_prototype"))
            key = (formula, space_group, prototype)
            all_formulas.add(formula)

            if key not in grouped:
                grouped[key] = {
                    "seed_formula": formula,
                    "seed_space_group": space_group,
                    "seed_prototype": prototype,
                "registry_material_ids": set(),
                "registry_statuses": set(),
                "registry_mp_ids": set(),
                }

            grouped[key]["registry_material_ids"].add(
                clean(row.get("material_id"))
            )
            grouped[key]["registry_statuses"].add(
                clean(row.get("electride_status"))
            )
            grouped[key]["registry_mp_ids"].update(
                re.findall(
                    r"mp-\d+",
                    clean(row.get("structure_source_or_database_id"), default=""),
                    flags=re.IGNORECASE,
                )
            )

    return list(grouped.values()), all_formulas


def doc_space_group(doc: Any) -> str:
    return clean(doc.symmetry.symbol if doc.symmetry else None)


def choose_seed_docs(seed_group: dict[str, Any], docs: list[Any]) -> list[Any]:
    """Select MP structures corresponding to the registry seed phase."""
    requested_space_group = seed_group["seed_space_group"]

    if requested_space_group != "not reported":
        matches = [
            doc for doc in docs if doc_space_group(doc) == requested_space_group
        ]
        if matches:
            return matches

    # If the registry has no usable space group, use the lowest-energy entry.
    # This is a fallback and is recorded through the source metadata.
    if not docs:
        return []

    lowest = min(
        docs,
        key=lambda doc: (
            doc.energy_above_hull
            if doc.energy_above_hull is not None
            else float("inf")
        ),
    )
    return [lowest]


def query_formula(formula: str, mpr: MPRester) -> list[Any]:
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "volume",
        "nsites",
    ]
    return list(
        mpr.materials.summary.search(
            formula=formula,
            fields=fields,
        )
    )


def query_material_ids(material_ids: list[str], mpr: MPRester) -> list[Any]:
    """Retrieve specific MP structures cited by the registry."""
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "volume",
        "nsites",
    ]
    return list(
        mpr.materials.summary.search(
            material_ids=material_ids,
            fields=fields,
        )
    )


def query_coarse_candidates(
    space_group_number: int,
    number_of_sites: int,
    mpr: MPRester,
) -> list[Any]:
    fields = [
        "material_id",
        "formula_pretty",
        "structure",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "volume",
        "nsites",
    ]
    return list(
        mpr.materials.summary.search(
            spacegroup_number=space_group_number,
            num_sites=number_of_sites,
            fields=fields,
        )
    )


def same_prototype(
    seed_structure: Any,
    candidate_structure: Any,
    matcher: StructureMatcher,
) -> bool:
    """Compare atomic arrangements while allowing element identities to differ."""
    return bool(matcher.fit_anonymous(seed_structure, candidate_structure))


def main() -> None:
    if not MP_API_KEY:
        raise ValueError(
            "MP_API_KEY was not found. Add it to the project-level .env file."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seed_groups, electride_formulas = load_seed_groups()
    print(
        f"[startup] Loaded {len(seed_groups)} unique seed groups "
        f"from {REGISTRY_PATH}",
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

    # Formula and coarse-query caches prevent repeated API work when several
    # electride seeds belong to the same chemical or structural family.
    formula_cache: dict[str, list[Any]] = {}
    coarse_cache: dict[tuple[int, int], list[Any]] = {}

    # Candidate MP IDs are the global deduplication key. If one control matches
    # several electride seeds, its seed provenance is aggregated into one row.
    candidate_records: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)

    try:
        with MPRester(api_key=MP_API_KEY, mute_progress_bars=True) as mpr:
            for seed_number, seed_group in enumerate(seed_groups, start=1):
                formula = seed_group["seed_formula"]
                label = (
                    f"{formula} | {seed_group['seed_space_group']} | "
                    f"{seed_group['seed_prototype']}"
                )
                print(
                    f"[{seed_number}/{len(seed_groups)}] Processing seed: {label}",
                    flush=True,
                )
                start = time.perf_counter()
                signal.setitimer(signal.ITIMER_REAL, TIMEOUT_SECONDS)

                try:
                    registry_mp_ids = sorted(seed_group["registry_mp_ids"])

                    if registry_mp_ids:
                        print(
                            f"  Querying {len(registry_mp_ids)} registry MP ID(s) "
                            f"for {formula}",
                            flush=True,
                        )
                        seed_query_docs = query_material_ids(registry_mp_ids, mpr)
                    else:
                        if formula not in formula_cache:
                            print(
                                f"  Querying seed structures for {formula}",
                                flush=True,
                            )
                            formula_cache[formula] = query_formula(formula, mpr)
                        seed_query_docs = formula_cache[formula]

                    # If a cited MP ID is stale or unavailable, fall back to
                    # the composition search before declaring the seed absent.
                    if not seed_query_docs and formula not in formula_cache:
                        print(
                            f"  Registry MP ID lookup empty; falling back to "
                            f"formula search for {formula}",
                            flush=True,
                        )
                        formula_cache[formula] = query_formula(formula, mpr)
                        seed_query_docs = formula_cache[formula]

                    seed_docs = choose_seed_docs(
                        seed_group,
                        seed_query_docs,
                    )

                    if not seed_docs:
                        failures.append({
                            "seed_formula": formula,
                            "seed_space_group": seed_group["seed_space_group"],
                            "seed_prototype": seed_group["seed_prototype"],
                            "reason": "no_seed_structure_found",
                            "message": (
                                "Materials Project returned no structure from "
                                "the registry MP IDs or formula search."
                            ),
                            "elapsed_seconds": round(
                                time.perf_counter() - start,
                                2,
                            ),
                        })
                        print(
                            f"  FAILED: no seed structure found for {label}",
                            flush=True,
                        )
                        continue

                    for seed_doc_number, seed_doc in enumerate(seed_docs, start=1):
                        seed_structure = seed_doc.structure
                        seed_reduced_formula = (
                            seed_structure.composition.reduced_formula
                        )
                        seed_symmetry = seed_doc.symmetry

                        if seed_symmetry is None:
                            print(
                                f"  Skipping {seed_doc.material_id}: no space group",
                                flush=True,
                            )
                            continue

                        coarse_key = (seed_symmetry.number, seed_doc.nsites)
                        if coarse_key not in coarse_cache:
                            print(
                                f"  Querying coarse candidates for "
                                f"SG {seed_symmetry.number}, "
                                f"{seed_doc.nsites} sites",
                                flush=True,
                            )
                            coarse_cache[coarse_key] = query_coarse_candidates(
                                seed_symmetry.number,
                                seed_doc.nsites,
                                mpr,
                            )

                        candidates = coarse_cache[coarse_key]
                        comparisons = 0
                        matches = 0
                        candidate_count = len(candidates)

                        for candidate_doc in candidates:
                            candidate_reduced_formula = (
                                candidate_doc.structure.composition.reduced_formula
                            )

                            # Same-composition phases belong to the separate
                            # polymorph workflow, not this control workflow.
                            if candidate_reduced_formula in electride_formulas:
                                continue

                            comparisons += 1
                            is_match = same_prototype(
                                seed_structure,
                                candidate_doc.structure,
                                matcher,
                            )

                            if is_match:
                                matches += 1
                                candidate_id = str(candidate_doc.material_id)
                                record = candidate_records.setdefault(
                                    candidate_id,
                                    {
                                        "candidate_mp_material_id": candidate_id,
                                        "candidate_formula": candidate_reduced_formula,
                                        "candidate_space_group": doc_space_group(candidate_doc),
                                        "candidate_space_group_number": candidate_doc.symmetry.number if candidate_doc.symmetry else "not reported",
                                        "candidate_crystal_system": candidate_doc.symmetry.crystal_system if candidate_doc.symmetry else "not reported",
                                        "candidate_energy_above_hull_eV_per_atom": candidate_doc.energy_above_hull,
                                        "candidate_formation_energy_per_atom_eV": candidate_doc.formation_energy_per_atom,
                                        "candidate_volume_A3": candidate_doc.volume,
                                        "candidate_number_of_sites": candidate_doc.nsites,
                                        "matched_seed_formulas": set(),
                                        "matched_seed_material_ids": set(),
                                        "matched_seed_space_groups": set(),
                                        "matched_seed_prototypes": set(),
                                    },
                                )
                                record["matched_seed_formulas"].add(formula)
                                record["matched_seed_material_ids"].update(
                                    seed_group["registry_material_ids"]
                                )
                                record["matched_seed_space_groups"].add(
                                    seed_group["seed_space_group"]
                                )
                                record["matched_seed_prototypes"].add(
                                    seed_group["seed_prototype"]
                                )

                            if (
                                comparisons == 1
                                or comparisons % 100 == 0
                                or comparisons == candidate_count
                            ):
                                print(
                                    f"  Seed {seed_doc.material_id}: compared "
                                    f"{comparisons}/{candidate_count}; "
                                    f"matches={matches}",
                                    flush=True,
                                )

                    elapsed = time.perf_counter() - start
                    print(
                        f"  completed in {elapsed:.1f} s; "
                        f"unique controls collected={len(candidate_records)}",
                        flush=True,
                    )
                except SeedTimeout as exc:
                    elapsed = time.perf_counter() - start
                    failure = {
                        "seed_formula": formula,
                        "seed_space_group": seed_group["seed_space_group"],
                        "seed_prototype": seed_group["seed_prototype"],
                        "reason": "timeout",
                        "message": str(exc),
                        "elapsed_seconds": round(elapsed, 2),
                    }
                    failures.append(failure)
                    print(
                        f"  FAILED: {label} exceeded 3 minutes",
                        flush=True,
                    )
                except Exception as exc:  # Keep the batch running.
                    elapsed = time.perf_counter() - start
                    failure = {
                        "seed_formula": formula,
                        "seed_space_group": seed_group["seed_space_group"],
                        "seed_prototype": seed_group["seed_prototype"],
                        "reason": "error",
                        "message": repr(exc),
                        "elapsed_seconds": round(elapsed, 2),
                    }
                    failures.append(failure)
                    print(f"  FAILED: {label}: {exc}", flush=True)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, previous_handler)

    output_fields = [
        "candidate_mp_material_id",
        "candidate_formula",
        "candidate_space_group",
        "candidate_space_group_number",
        "candidate_crystal_system",
        "candidate_energy_above_hull_eV_per_atom",
        "candidate_formation_energy_per_atom_eV",
        "candidate_volume_A3",
        "candidate_number_of_sites",
        "matched_seed_formulas",
        "matched_seed_material_ids",
        "matched_seed_space_groups",
        "matched_seed_prototypes",
        "control_status",
    ]

    output_rows = []
    for record in candidate_records.values():
        output_rows.append({
            "candidate_mp_material_id": record["candidate_mp_material_id"],
            "candidate_formula": record["candidate_formula"],
            "candidate_space_group": record["candidate_space_group"],
            "candidate_space_group_number": record["candidate_space_group_number"],
            "candidate_crystal_system": record["candidate_crystal_system"],
            "candidate_energy_above_hull_eV_per_atom": record["candidate_energy_above_hull_eV_per_atom"],
            "candidate_formation_energy_per_atom_eV": record["candidate_formation_energy_per_atom_eV"],
            "candidate_volume_A3": record["candidate_volume_A3"],
            "candidate_number_of_sites": record["candidate_number_of_sites"],
            "matched_seed_formulas": "|".join(sorted(record["matched_seed_formulas"])),
            "matched_seed_material_ids": "|".join(sorted(record["matched_seed_material_ids"])),
            "matched_seed_space_groups": "|".join(sorted(record["matched_seed_space_groups"])),
            "matched_seed_prototypes": "|".join(sorted(record["matched_seed_prototypes"])),
            "control_status": "control_candidate",
        })

    output_rows.sort(
        key=lambda row: (
            row["candidate_formula"],
            row["candidate_mp_material_id"],
        )
    )

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(output_rows)

    failure_report = {
        "timeout_seconds_per_seed_group": TIMEOUT_SECONDS,
        "total_seed_groups": len(seed_groups),
        "completed_or_skipped_seed_groups": len(seed_groups) - len(failures),
        "failure_count": len(failures),
        "deduplicated_candidate_count": len(output_rows),
        "failures": failures,
    }

    with OUTPUT_FAILURES.open("w", encoding="utf-8") as handle:
        json.dump(failure_report, handle, indent=2)
        handle.write("\n")

    print(f"\nWrote {len(output_rows)} deduplicated candidates to {OUTPUT_CSV}")
    print(f"Wrote failure report to {OUTPUT_FAILURES}")


if __name__ == "__main__":
    main()
