#!/usr/bin/env python3
"""Generate SCF parameter-sweep directories from the benchmark relaxations."""

from pathlib import Path
import re


ROOT = Path(__file__).parent


def replace_one(text: str, pattern: str, replacement: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"Could not replace {pattern!r}")
    return updated


def scf_input(base: str, prefix_suffix: str, outdir: str, parameter: str, value: str) -> str:
    text = replace_one(base, r"calculation\s*=\s*'vc-relax'", "calculation = 'scf'")
    text = re.sub(r"\n&IONS\n.*?^/\n", "\n", text, count=1, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"\n&CELL\n.*?^/\n", "\n", text, count=1, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"^\s*nstep\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^\s*etot_conv_thr\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    text = re.sub(r"^\s*forc_conv_thr\s*=.*\n", "", text, count=1, flags=re.MULTILINE)
    text = replace_one(text, r"prefix\s*=\s*'([^']+)'", f"prefix = '\\1_{prefix_suffix}'")
    text = replace_one(text, r"outdir\s*=\s*'[^']+'", f"outdir = './{outdir}'")
    text = text.replace("pseudo_dir = '../../pseudo/", "pseudo_dir = '../../../pseudo/")

    if parameter == "ecutwfc":
        ecut, rho = value.split(":")
        text = replace_one(text, r"ecutwfc\s*=\s*\d+", f"ecutwfc = {ecut}")
        text = replace_one(text, r"ecutrho\s*=\s*\d+", f"ecutrho = {rho}")
    elif parameter == "kpoints":
        text = replace_one(text, r"(?m)^\s*\d+\s+\d+\s+\d+\s+0\s+0\s+0\s*$", value)
    elif parameter == "smearing":
        text = replace_one(text, r"degauss\s*=\s*[0-9.]+", f"degauss = {value}")
    elif parameter == "conv_thr":
        text = replace_one(text, r"conv_thr\s*=\s*[0-9.]+d[-+]?\d+", f"conv_thr = {value}")
    else:
        raise ValueError(parameter)

    return (
        f"! SCF parameter sweep: {parameter} = {value}\n"
        "! Same structure, functional, pseudopotential, and magnetic seed as the parent vc-relax input.\n"
        + text
    )


def elapsed_block() -> str:
    return r'''format_elapsed() {
  local total_seconds="$1"
  printf '%02d:%02d:%02d' \
    "$((total_seconds / 3600))" \
    "$(((total_seconds % 3600) / 60))" \
    "$((total_seconds % 60))"
}
'''


def sweep_submit(material: str, functional: str, parameter: str) -> str:
    job = f"{material}_{functional.lower()}_{parameter}_scf"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job}",
        "#SBATCH --account=andre176",
        "#SBATCH --output=slurm-%x-%j.out",
        "#SBATCH --error=slurm-%x-%j.err",
        "#SBATCH --time=00:30:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=9",
        "#SBATCH --mem=4G",
        "#SBATCH --partition=cpu",
        "",
        "set -euo pipefail",
        "",
        "module load quantum-espresso",
        'script_dir="${SLURM_SUBMIT_DIR:-$PWD}"',
        'cd "$script_dir"',
        "",
        elapsed_block().rstrip(),
        'job_start=$(date +%s)',
        f'echo "Starting {material} {functional} {parameter} SCF sweep: $(date)"',
        "",
        "for input in scf_*.in; do",
        '  stem="${input%.in}"',
        '  output="${stem}.out"',
        '  case_start=$(date +%s)',
        '  echo "[$(date)] Running ${input}"',
        '  srun pw.x -in "$input" > "$output"',
        '  case_elapsed=$(( $(date +%s) - case_start ))',
        '  echo "[$(date)] ${input} elapsed time: $(format_elapsed "$case_elapsed")"',
        "done",
        "",
        'job_elapsed=$(( $(date +%s) - job_start ))',
        'echo "Completed sweep at $(date)"',
        'echo "Total job elapsed time: $(format_elapsed "$job_elapsed")"',
        "",
    ]
    return "\n".join(lines)


def material_submit(material: str) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={material}_parameter_tests",
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
        'script_dir="${SLURM_SUBMIT_DIR:-$PWD}"',
        'cd "$script_dir"',
        "",
        'echo "Submitting SCF parameter sweeps for ' + material + '"',
        "for functional in PBE PBEsol R2SCAN; do",
        "  for parameter in ecutwfc kpoints smearing_or_conv_thr; do",
        '    sweep_dir="${script_dir}/${functional}/${parameter}"',
        '    sbatch --chdir="$sweep_dir" "$sweep_dir/submit.slurm"',
        "  done",
        "done",
        "",
    ]
    return "\n".join(lines)


base_paths = []
for functional in ("PBE", "PBEsol", "R2SCAN"):
    base_paths.extend(ROOT.glob(f"*/{functional}/*_vc-relax.in"))

for base_path in sorted(base_paths):
    material = base_path.parts[-3]
    functional = base_path.parts[-2]
    base = base_path.read_text()
    ecut = int(re.search(r"ecutwfc\s*=\s*(\d+)", base).group(1))
    rho = int(re.search(r"ecutrho\s*=\s*(\d+)", base).group(1))
    k_match = re.search(r"(?m)^\s*(\d+)\s+(\d+)\s+(\d+)\s+0\s+0\s+0\s*$", base)
    kpoints = tuple(int(k_match.group(i)) for i in range(1, 4))
    occupations_smearing = "occupations = 'smearing'" in base

    cutoff_values = []
    for label, factor in (("low", 0.8), ("base", 1.0), ("high", 1.2)):
        cutoff_values.append((f"ecutwfc_{label}", f"{round(ecut * factor)}:{round(rho * factor)}"))

    kpoint_values = []
    for label, factor in (("low", 0.75), ("base", 1.0), ("high", 1.25)):
        mesh = tuple(max(1, round(k * factor)) for k in kpoints)
        kpoint_values.append((f"kpoints_{label}", "  %d %d %d 0 0 0" % mesh))

    if occupations_smearing:
        final_parameter = "smearing"
        final_values = [(f"degauss_{x.replace('.', 'p')}", x) for x in ("0.01", "0.02", "0.04")]
    else:
        final_parameter = "conv_thr"
        final_values = [(f"conv_thr_{x.replace('.', 'd').replace('-', 'm')}", x) for x in ("1.0d-8", "1.0d-9", "1.0d-10")]

    sweeps = [("ecutwfc", cutoff_values), ("kpoints", kpoint_values), ("smearing_or_conv_thr", final_values)]
    for directory_name, values in sweeps:
        sweep_parameter = "conv_thr" if directory_name == "smearing_or_conv_thr" and final_parameter == "conv_thr" else directory_name
        if directory_name == "smearing_or_conv_thr" and final_parameter == "smearing":
            sweep_parameter = "smearing"
        sweep_dir = base_path.parent / directory_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for label, value in values:
            input_name = f"scf_{label}.in"
            outdir = f"out_{label}"
            (sweep_dir / input_name).write_text(scf_input(base, label, outdir, sweep_parameter, value))
        (sweep_dir / "submit.slurm").write_text(sweep_submit(material, functional, directory_name))

    (base_path.parent.parent / "submit_parameter_tests.slurm").write_text(material_submit(material))
