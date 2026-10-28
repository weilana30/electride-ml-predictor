#!/usr/bin/env python3
"""Generate fixed-geometry CP2K convergence inputs from the QE source inputs."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np

from find_voids import Atom, find_candidates

ROOT = Path(__file__).resolve().parents[2]
CP2K_ROOT = Path(__file__).resolve().parent
MATERIALS = ("Ba2As", "Ba3CrN3", "BaSn3", "Ca2N", "Cu2SiSe3", "MgS")
PSEUDO = {"Ba": ("DZVP-MOLOPT-SR-GTH-q10", "GTH-PBE-q10"),
          "As": ("DZVP-MOLOPT-SR-GTH-q5", "GTH-PBE-q5"),
          "Cr": ("DZVP-MOLOPT-SR-GTH-q14", "GTH-PBE-q14"),
          "N": ("DZVP-MOLOPT-SR-GTH-q5", "GTH-PBE-q5"),
          "Sn": ("DZVP-MOLOPT-SR-GTH-q4", "GTH-PBE-q4"),
          "Ca": ("DZVP-MOLOPT-SR-GTH-q10", "GTH-PBE-q10"),
          "Cu": ("DZVP-MOLOPT-SR-GTH-q11", "GTH-PBE-q11"),
          "Si": ("DZVP-MOLOPT-SR-GTH-q4", "GTH-PBE-q4"),
          "Se": ("DZVP-MOLOPT-SR-GTH-q6", "GTH-PBE-q6"),
          "Mg": ("DZVP-MOLOPT-SR-GTH-q10", "GTH-PBE-q10"),
          "S": ("DZVP-MOLOPT-SR-GTH-q6", "GTH-PBE-q6")}


def slurm_account() -> str:
    account = os.environ.get("SLURM_ACCOUNT")
    if not account:
        raise RuntimeError("Set SLURM_ACCOUNT before generating Slurm files")
    return account


def read_qe_structure(path: Path):
    text = path.read_text(encoding="utf-8")
    cell_match = re.search(r"CELL_PARAMETERS\s*(?:\([^)]*\)|\w+)?\s*\n(.*?)(?=\n\s*(?:ATOMIC_POSITIONS|K_POINTS))", text, re.I | re.S)
    pos_match = re.search(r"ATOMIC_POSITIONS\s*(?:\(([^)]*)\)|(\w+))\s*\n(.*?)(?=\n\s*(?:K_POINTS|CELL_PARAMETERS|$))", text, re.I | re.S)
    if not cell_match or not pos_match:
        raise ValueError(f"Could not parse QE cell and positions from {path}")
    cell_rows = [[float(v.replace("D", "E")) for v in line.split()[:3]] for line in cell_match.group(1).splitlines() if line.strip()]
    if len(cell_rows) != 3:
        raise ValueError(f"Expected three cell vectors in {path}")
    unit = (cell_match.group(0).splitlines()[0].split()[-1].strip("() ").lower())
    if "bohr" in unit:
        cell_rows = (np.asarray(cell_rows) * 0.529177210903).tolist()
    unit = (pos_match.group(1) or pos_match.group(2) or "angstrom").lower()
    atoms = []
    for line in pos_match.group(3).splitlines():
        fields = line.split()
        element = re.match(r"^([A-Z][a-z]?)", fields[0]) if fields else None
        if len(fields) < 4 or not element:
            continue
        symbol = element.group(1)
        coords = np.array([float(v.replace("D", "E")) for v in fields[1:4]])
        atoms.append((symbol, coords))
    if unit.startswith("crystal"):
        cart = [(symbol, coords @ np.asarray(cell_rows)) for symbol, coords in atoms]
    else:
        factor = 0.529177210903 if "bohr" in unit else 1.0
        cart = [(symbol, coords * factor) for symbol, coords in atoms]
    if not cart:
        raise ValueError(f"No atomic positions found in {path}")
    return np.asarray(cell_rows, dtype=float), cart


def frac_positions(lattice, cart_atoms):
    inverse = np.linalg.inv(lattice)
    return [Atom(symbol, tuple((xyz @ inverse) % 1.0)) for symbol, xyz in cart_atoms]


def make_cp2k_input(material, atom_rows, lattice, cutoff, rel_cutoff, mesh, project, basis_level="DZVP"):
    symbols = sorted({symbol for symbol, _ in atom_rows})
    unknown = sorted(set(symbols) - PSEUDO.keys())
    if unknown:
        raise ValueError(f"No CP2K basis/pseudopotential selection for: {unknown}")
    cell = "\n".join(f"      {axis} " + " ".join(f"{x: .12f}" for x in vector) for axis, vector in zip("ABC", lattice))
    coords = "\n".join(f"      {symbol:<2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}" for symbol, xyz in atom_rows)
    kinds = "\n".join(f"      &KIND {symbol}\n        BASIS_SET {basis_level}-MOLOPT-SR-GTH-q{PSEUDO[symbol][1].rsplit('q', 1)[1]}\n        POTENTIAL {PSEUDO[symbol][1]}\n      &END KIND" for symbol in symbols)
    return f"""&GLOBAL
  PROJECT {project}
  RUN_TYPE ENERGY
  PRINT_LEVEL MEDIUM
&END GLOBAL

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    BASIS_SET_FILE_NAME BASIS_MOLOPT_UCL
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    &MGRID
      CUTOFF {cutoff}
      REL_CUTOFF {rel_cutoff}
      NGRIDS 5
    &END MGRID
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &END QS
    &SCF
      SCF_GUESS ATOMIC
      EPS_SCF 1.0E-7
      MAX_SCF 200
      &DIAGONALIZATION ON\n      &END DIAGONALIZATION
      &MIXING
        METHOD BROYDEN_MIXING
        ALPHA 0.05
        NBROYDEN 8
      &END MIXING
    &END SCF
    &KPOINTS
      SCHEME MONKHORST-PACK {mesh[0]} {mesh[1]} {mesh[2]}
    &END KPOINTS
    &XC
      &XC_FUNCTIONAL PBE
      &END XC_FUNCTIONAL
    &END XC
    &POISSON
      PERIODIC XYZ
    &END POISSON
    &PRINT
      &ELF_CUBE
        STRIDE 1 1 1
      &END ELF_CUBE
    &END PRINT
  &END DFT
  &SUBSYS
    &CELL
{cell}
      PERIODIC XYZ
    &END CELL
    &COORD
{coords}
    &END COORD
{kinds}
  &END SUBSYS
&END FORCE_EVAL
"""


def write_submit(case_dir: Path, material: str, group: str, case: str, memory_gb: int):
    name = f"{material}_{group}_{case}"[:60]
    content = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --account={slurm_account()}
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem={memory_gb}G
#SBATCH --partition=cpu

set -euo pipefail
module load cp2k/2022.1
exe=${{CP2K_EXECUTABLE:-cp2k.psmp}}
command -v "$exe" >/dev/null || {{ echo "CP2K executable not found: $exe" >&2; exit 127; }}
case_dir=$(pwd -P)
scratch_base="${{RCAC_SCRATCH:-/scratch/negishi/$USER}}/electride-ml-predictor/cp2k"
scratch_dir="$scratch_base/${{SLURM_JOB_NAME}}_${{SLURM_JOB_ID}}"
workflow_dir=$(cd "$case_dir/../../.." && pwd -P)
candidate_file="$case_dir/../../structure/void_candidates.csv"
mkdir -p "$scratch_dir"
cp cp2k.inp "$scratch_dir/"
finish() {{
  status=$?
  if [[ -f "$scratch_dir/cp2k.out" ]]; then cp "$scratch_dir/cp2k.out" "$case_dir/"; fi
  for report in "$scratch_dir"/*.void_elf.csv; do
    if [[ -f "$report" ]]; then cp "$report" "$case_dir/"; fi
  done
  echo "CP2K scratch retained at $scratch_dir" >&2
  exit "$status"
}}
trap finish EXIT
cd "$scratch_dir"
srun "$exe" -i cp2k.inp -o cp2k.out
found_cube=0
for cube in "$scratch_dir"/*-ELF*.cube; do
  [[ -f "$cube" ]] || continue
  found_cube=1
  report="$scratch_dir/$(basename "$cube").void_elf.csv"
  python3 "$workflow_dir/analyze_void_elf.py" "$cube" "$candidate_file" --output "$report"
done
if [[ "$found_cube" -eq 0 ]]; then
  echo "CP2K completed without producing an ELF cube in $scratch_dir" >&2
  exit 1
fi
"""
    path = case_dir / "submit.slurm"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def main():
    for material in MATERIALS:
        source = ROOT / "dft" / material / "PBE" / "relax" / "pbe_vc-relax.in"
        lattice, cart_atoms = read_qe_structure(source)
        outdir = CP2K_ROOT / material
        structure_dir = outdir / "structure"
        structure_dir.mkdir(parents=True, exist_ok=True)
        (structure_dir / "structure.xyz").write_text(
            f"{len(cart_atoms)}\nQE pbe_vc-relax input structure; fixed geometry\n" +
            "".join(f"{symbol:<2s} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}\n" for symbol, xyz in cart_atoms),
            encoding="utf-8")
        atoms = frac_positions(lattice, cart_atoms)
        sites = find_candidates(lattice, atoms)
        with (structure_dir / "void_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["site", "frac_x", "frac_y", "frac_z", "cart_x_A", "cart_y_A", "cart_z_A", "clearance_radius_A", "nearest_species", "nearest_distance_A", "nearest_shell", "multiplicity"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for site in sites:
                row = {key: site[key] for key in ("site", "clearance_radius_A", "nearest_species", "nearest_distance_A", "nearest_shell", "multiplicity")}
                row.update(dict(zip(("frac_x", "frac_y", "frac_z"), site["frac"])))
                row.update(dict(zip(("cart_x_A", "cart_y_A", "cart_z_A"), site["cart"])))
                writer.writerow(row)

        cases = []
        for cutoff, rel_cutoff, memory_gb in ((300, 40, 16), (400, 50, 16), (500, 60, 20), (600, 70, 24)):
            cases.append(("cutoff", str(cutoff), cutoff, rel_cutoff, (5, 5, 5), memory_gb))
        for mesh, memory_gb in ((3, 16), (5, 20), (7, 24), (9, 32)):
            cases.append(("kpoints", str(mesh), 500, 60, (mesh, mesh, mesh), memory_gb))
        cases.append(("basis", "tzvp", 500, 60, (5, 5, 5), 24))
        for group, case, cutoff, rel_cutoff, mesh, memory_gb in cases:
            case_dir = outdir / group / case
            case_dir.mkdir(parents=True, exist_ok=True)
            project = re.sub(r"[^A-Za-z0-9_-]", "_", f"{material}_{group}_{case}")
            basis_level = "TZVP" if group == "basis" else "DZVP"
            (case_dir / "cp2k.inp").write_text(make_cp2k_input(material, cart_atoms, lattice, cutoff, rel_cutoff, mesh, project, basis_level), encoding="utf-8")
            write_submit(case_dir, material, group, case, memory_gb)
        all_submit = outdir / "submit_all.slurm"
        all_submit.write_text(
            "#!/bin/bash\n#SBATCH --job-name=" + material
            + f"_cp2k_submit\n#SBATCH --account={slurm_account()}\n"
            "#SBATCH --time=00:05:00\n#SBATCH --nodes=1\n"
            "#SBATCH --ntasks-per-node=1\n#SBATCH --mem=1G\n#SBATCH --partition=cpu\n"
            "set -euo pipefail\nfor case in cutoff/* kpoints/* basis/*; do "
            "(cd \"$case\" && sbatch submit.slurm); done\n",
            encoding="utf-8",
        )
        all_submit.chmod(0o755)
        print(f"{material}: {len(cart_atoms)} atoms, {len(sites)} structural void candidates, {len(cases)} CP2K cases -> {outdir}")


if __name__ == "__main__":
    main()
