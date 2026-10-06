#!/usr/bin/env python3
"""Generate the PBE-only SCF benchmark workflow."""

from __future__ import annotations

import re
import shutil
from pathlib import Path


ROOT = Path(__file__).parent


def set_value(text: str, key: str, value: str) -> str:
    pattern = rf"(\b{re.escape(key)}\s*=\s*)(?:'[^']*'|[^,\n/]+)"
    updated, count = re.subn(pattern, rf"\g<1>{value}", text, count=1)
    if count != 1:
        raise RuntimeError(f"Could not set {key!r}")
    return updated


def remove_namelist(text: str, section: str) -> str:
    return re.sub(
        rf"\n&{section}\n.*?^/\n",
        "\n",
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


def normalize_relax(text: str) -> str:
    text = text.replace(
        "Pre-production settings: converge cutoffs and k mesh before comparing functionals.",
        "Pre-production settings: converge cutoffs and k mesh before production calculations.",
    )
    text = set_value(text, "pseudo_dir", "'../../../pseudo/PBE_PAW'")
    return set_value(text, "outdir", "'./qe_data'")


def normalize_scf(base: str, label: str, parameter: str, value: str) -> str:
    text = re.sub(r"calculation\s*=\s*'vc-relax'", "calculation = 'scf'", base, count=1)
    text = text.replace(
        "Pre-production settings: converge cutoffs and k mesh before comparing functionals.",
        "Pre-production settings: converge cutoffs and k mesh before production calculations.",
    )
    text = remove_namelist(remove_namelist(text, "IONS"), "CELL")
    text = re.sub(
        r"^\s*(nstep|etot_conv_thr|forc_conv_thr)\s*=.*\n",
        "",
        text,
        flags=re.MULTILINE,
    )
    prefix_match = re.search(r"prefix\s*=\s*'([^']+)'", text)
    if prefix_match is None:
        raise RuntimeError("Could not find prefix")
    text = set_value(text, "prefix", f"'{prefix_match.group(1)}_{parameter}_{label}'")
    text = set_value(text, "outdir", "'./qe_data'")
    text = set_value(text, "pseudo_dir", "'../../../../pseudo/PBE_PAW'")
    text = set_value(text, "tstress", ".true.")
    text = set_value(text, "tprnfor", ".true.")

    if parameter == "ecutwfc":
        ecutwfc, ecutrho = value.split(":")
        text = set_value(text, "ecutwfc", ecutwfc)
        text = set_value(text, "ecutrho", ecutrho)
    elif parameter == "kpoints":
        text, count = re.subn(
            r"(?m)^\s*\d+\s+\d+\s+\d+\s+0\s+0\s+0\s*$",
            value,
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError("Could not set k-point mesh")
    elif parameter == "degauss":
        text = set_value(text, "degauss", value)
    else:
        raise ValueError(parameter)

    return (
        f"! PBE SCF benchmark: {parameter} = {value}\n"
        "! QE data are staged in scratch by the case submission script.\n"
        + text
    )


def case_submit(material: str, parameter: str, label: str, levels_to_root: int, relax: bool = False) -> str:
    job_name = f"{material}_pbe_{'relax' if relax else parameter}_{label}"
    time_limit = "02:00:00" if relax else "01:00:00"
    relative_root = "/".join(".." for _ in range(levels_to_root))
    input_name = "pbe_vc-relax.in" if relax else "scf.in"
    parameter_name = "relax" if relax else parameter
    return f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=9
#SBATCH --mem=16G
#SBATCH --partition=cpu

set -euo pipefail
case_dir=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
project_root=$(cd "$case_dir/{relative_root}" && pwd)
exec bash "$project_root/dft/run_scf_case.sh" "$project_root" "$case_dir" "{input_name}" "{material}" "{parameter_name}" "{label}"
'''


def submit_children(material: str, parameter: str, labels: list[str]) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={material}_pbe_{parameter}_submit",
        "#SBATCH --account=andre176",
        "#SBATCH --output=slurm-%x-%j.out",
        "#SBATCH --error=slurm-%x-%j.err",
        "#SBATCH --time=00:05:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --mem=1G",
        "#SBATCH --partition=cpu",
        "",
        "set -euo pipefail",
        'parent_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)',
        "",
        f'echo "Submitting {material} PBE {parameter} cases"',
    ]
    for label in labels:
        lines.append(
            f'sbatch --chdir="$parent_dir/{label}" '
            f'--output="$parent_dir/{label}/slurm-%x-%j.out" '
            f'--error="$parent_dir/{label}/slurm-%x-%j.err" '
            f'"$parent_dir/{label}/submit.slurm"'
        )
    return "\n".join(lines) + "\n"


def submit_relax(material: str) -> str:
    return f'''#!/bin/bash
#SBATCH --job-name={material}_pbe_relax_submit
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

set -euo pipefail
parent_dir=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
sbatch --chdir="$parent_dir/relax" \\
  --output="$parent_dir/relax/slurm-%x-%j.out" \\
  --error="$parent_dir/relax/slurm-%x-%j.err" \\
  "$parent_dir/relax/submit.slurm"
'''


def material_submit(material: str) -> str:
    return f'''#!/bin/bash
#SBATCH --job-name={material}_pbe_parameter_tests
#SBATCH --account=andre176
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --partition=cpu

set -euo pipefail
material_dir=$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)
for parameter in ecutwfc kpoints smearing_or_conv_thr; do
  bash "$material_dir/PBE/$parameter/submit.slurm"
done
'''


def main() -> None:
    sources = {}
    for material_dir in sorted(ROOT.iterdir()):
        if not material_dir.is_dir():
            continue
        source = material_dir / "PBE" / "pbe_vc-relax.in"
        if not source.exists():
            source = material_dir / "PBE" / "relax" / "pbe_vc-relax.in"
        if source.exists():
            sources[material_dir.name] = source.read_text()

    for material, base in sources.items():
        material_dir = ROOT / material
        pbe_dir = material_dir / "PBE"
        for old_dir in ("ecutwfc", "kpoints", "smearing_or_conv_thr"):
            shutil.rmtree(pbe_dir / old_dir, ignore_errors=True)
        shutil.rmtree(material_dir / "R2SCAN", ignore_errors=True)
        for old_file in ("pbe_vc-relax.in", "submit.slurm"):
            (pbe_dir / old_file).unlink(missing_ok=True)

        relax_dir = pbe_dir / "relax"
        relax_dir.mkdir(parents=True, exist_ok=True)
        (relax_dir / "pbe_vc-relax.in").write_text(normalize_relax(base))
        (relax_dir / "submit.slurm").write_text(case_submit(material, "relax", "final", 4, True))
        (pbe_dir / "submit.slurm").write_text(submit_relax(material))

        ecut = int(re.search(r"ecutwfc\s*=\s*(\d+)", base).group(1))
        rho = int(re.search(r"ecutrho\s*=\s*(\d+)", base).group(1))
        k_match = re.search(r"(?m)^\s*(\d+)\s+(\d+)\s+(\d+)\s+0\s+0\s+0\s*$", base)
        if k_match is None:
            raise RuntimeError(f"Could not find k-point mesh for {material}")
        kpoints = tuple(int(k_match.group(i)) for i in range(1, 4))

        sweeps = {
            "ecutwfc": [
                ("low", f"{round(ecut * 0.8)}:{round(rho * 0.8)}"),
                ("base", f"{ecut}:{rho}"),
                ("high", f"{round(ecut * 1.2)}:{round(rho * 1.2)}"),
            ],
            "kpoints": [
                ("low", "  %d %d %d 0 0 0" % tuple(max(1, round(k * 0.75)) for k in kpoints)),
                ("base", "  %d %d %d 0 0 0" % kpoints),
                ("high", "  %d %d %d 0 0 0" % tuple(max(1, round(k * 1.25)) for k in kpoints)),
            ],
            "smearing_or_conv_thr": [
                ("degauss_0p01", "0.01"),
                ("degauss_0p02", "0.02"),
                ("degauss_0p04", "0.04"),
            ],
        }

        for parameter, cases in sweeps.items():
            sweep_dir = pbe_dir / parameter
            sweep_dir.mkdir(parents=True, exist_ok=True)
            labels = []
            for label, value in cases:
                case_dir = sweep_dir / label
                case_dir.mkdir(parents=True, exist_ok=True)
                labels.append(label)
                actual_parameter = "degauss" if parameter == "smearing_or_conv_thr" else parameter
                (case_dir / "scf.in").write_text(normalize_scf(base, label, actual_parameter, value))
                (case_dir / "submit.slurm").write_text(case_submit(material, parameter, label, 5))
            (sweep_dir / "submit.slurm").write_text(submit_children(material, parameter, labels))

        (material_dir / "submit_parameter_tests.slurm").write_text(material_submit(material))


if __name__ == "__main__":
    main()
