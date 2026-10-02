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
    text = text.replace("pseudo_dir = '../../pseudo/PBE_PAW'", "pseudo_dir = '../../../pseudo/PBE_PAW'")

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


def case_submit(material: str, functional: str, input_name: str) -> str:
    stem = input_name.removesuffix(".in")
    job = f"{material}_{functional.lower()}_{stem}"
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job}",
        "#SBATCH --account=andre176",
        "#SBATCH --output=slurm-%x-%j.out",
        "#SBATCH --error=slurm-%x-%j.err",
        "#SBATCH --time=01:00:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=9",
        "#SBATCH --mem=16G",
        "#SBATCH --partition=cpu",
        "",
        "set -euo pipefail",
        "",
        "module load quantum-espresso",
        "export OMPI_MCA_pml=ob1",
        "export OMPI_MCA_btl=self,tcp",
        'script_dir="$PWD"',
        "",
        elapsed_block().rstrip(),
        'job_start=$(date +%s)',
        f'echo "Starting {material} {functional} {stem}: $(date)"',
        f'echo "Running {input_name}"',
        f'srun pw.x -in "{input_name}" > "{stem}.out"',
        'job_elapsed=$(( $(date +%s) - job_start ))',
        'echo "Completed calculation at $(date)"',
        'echo "Total job elapsed time: $(format_elapsed "$job_elapsed")"',
        "",
    ]
    return "\n".join(lines)


def sweep_submit(material: str, functional: str, parameter: str) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={material}_{functional.lower()}_{parameter}_submit",
        "#SBATCH --account=andre176",
        "#SBATCH --output=slurm-%x-%j.out",
        "#SBATCH --error=slurm-%x-%j.err",
        "#SBATCH --time=00:05:00",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --mem=16G",
        "#SBATCH --partition=cpu",
        "",
        "set -euo pipefail",
        'script_dir="$PWD"',
        "",
        f'echo "Submitting {material} {functional} {parameter} SCF cases"',
        "for job in submit_scf_*.slurm; do",
        '  sbatch --chdir="$script_dir" "$script_dir/$job"',
        "done",
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
        "#SBATCH --mem=16G",
        "#SBATCH --partition=cpu",
        "",
        "set -euo pipefail",
        'script_dir="$PWD"',
        "",
        'echo "Submitting SCF parameter cases for ' + material + '"',
        "for functional in PBE R2SCAN; do",
        "  for parameter in ecutwfc kpoints smearing_or_conv_thr; do",
        '    sweep_dir="${script_dir}/${functional}/${parameter}"',
        '    for job in "$sweep_dir"/submit_scf_*.slurm; do',
        '      sbatch --chdir="$sweep_dir" "$job"',
        "    done",
        "  done",
        "done",
        "",
    ]
    return "\n".join(lines)


base_paths = []
PAW_FILES = {
    "As": "As.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ca": "Ca.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cr1": "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cr2": "Cr.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Mg": "Mg.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Se": "Se.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Sn": "Sn.pbe-dn-kjpaw_psl.1.0.0.UPF",
}


def normalize_base(text: str) -> str:
    def set_value(text: str, section: str, key: str, value: str) -> str:
        section_match = re.search(rf"(?ms)^&{section}\s*\n(.*?^/)", text)
        if section_match is None:
            raise RuntimeError(f"Could not find &{section} section")
        block = re.sub(rf"\b{key}\s*=\s*[^,\n/]+,?", "", section_match.group(0))
        block = block.replace("\n/", f"\n  {key} = {value},\n/", 1)
        return text[:section_match.start()] + block + text[section_match.end():]

    text = set_value(text, "SYSTEM", "ecutwfc", "60")
    text = set_value(text, "SYSTEM", "ecutrho", "480")
    text = set_value(text, "SYSTEM", "occupations", "'smearing'")
    text = set_value(text, "SYSTEM", "smearing", "'methfessel-paxton'")
    text = set_value(text, "SYSTEM", "degauss", "0.02")
    text = set_value(text, "ELECTRONS", "conv_thr", "1.0d-8")
    text = set_value(text, "ELECTRONS", "mixing_beta", "0.7")
    text = replace_one(text, r"(?m)^\s*\d+\s+\d+\s+\d+\s+0\s+0\s+0\s*$", "  17 17 17  0 0 0")
    text = replace_one(text, r"pseudo_dir\s*=\s*'[^']+'", "pseudo_dir = '../../pseudo/PBE_PAW'")

    species = re.search(r"(?ms)(^ATOMIC_SPECIES\s*\n)(.*?)(?=^\s*$)", text)
    if species is None:
        raise RuntimeError("Could not find ATOMIC_SPECIES block")

    def replace_species(match: re.Match[str]) -> str:
        element = match.group(2)
        if element not in PAW_FILES:
            raise RuntimeError(f"No downloaded PBE PAW file for {element}")
        return f"{match.group(1)}{element}{match.group(3)}{PAW_FILES[element]}"

    species_block = re.sub(
        r"(?m)^(\s*)([A-Z][a-z]?\d*)(\s+\S+\s+)\S+\s*$",
        replace_species,
        species.group(2),
    )
    return text[:species.start(2)] + species_block + text[species.end(2):]


for functional in ("PBE", "R2SCAN"):
    base_paths.extend(ROOT.glob(f"*/{functional}/*_vc-relax.in"))

for base_path in sorted(base_paths):
    material = base_path.parts[-3]
    functional = base_path.parts[-2]
    base = normalize_base(base_path.read_text())
    base_path.write_text(base)
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
        final_values = [(f"conv_thr_{x.replace('.', 'd').replace('-', 'm')}", x) for x in ("1.0d-7", "1.0d-8", "1.0d-9")]

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
            (sweep_dir / f"submit_{input_name.removesuffix('.in')}.slurm").write_text(
                case_submit(material, functional, input_name)
            )
        (sweep_dir / "submit.slurm").write_text(sweep_submit(material, functional, directory_name))

    (base_path.parent.parent / "submit_parameter_tests.slurm").write_text(material_submit(material))
