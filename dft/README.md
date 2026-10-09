# PBE DFT workflow

The separate CP2K fixed-geometry convergence and structure-only void workflow is
documented in [`cp2k/README.md`](cp2k/README.md).

This directory contains PBE calculations for three electrides and three control materials:

| Role | Structure | Dataset ID |
| --- | --- | --- |
| Electride | Ba2As | ELD-0003 / mp-2768 |
| Electride | Ba3CrN3 | ELD-0014 / mp-12905 |
| Electride | Ca2N | ELD-0049 / mp-2686 |
| Control | MgS | mp-1315 |
| Control | BaSn3 | mp-7759 |
| Control | Cu2SiSe3 | mp-15896 |

R2SCAN has been removed. The PBE PAW pseudopotentials are in `pseudo/PBE_PAW/`.

## Layout

Each material has the following structure:

```text
<material>/
├── PBE/
│   ├── relax/
│   │   ├── pbe_vc-relax.in
│   │   └── submit.slurm
│   ├── ecutwfc/
│   │   ├── low/
│   │   ├── base/
│   │   ├── high/
│   │   └── submit.slurm
│   ├── kpoints/
│   └── smearing_or_conv_thr/
└── submit_parameter_tests.slurm
```

Every benchmark value has its own directory. Its `scf.in` and `submit.slurm` are kept with the value-specific outputs.

## Scratch and copied-back files

`dft/run_scf_case.sh` stages each calculation in:

```text
$RCAC_SCRATCH/electride-ml-predictor/<material>_<test>_<value>_<jobid>/
```

The QE save data, ELF cube, and density cube remain in scratch while the job runs. After post-processing, the cube files and QE save data are removed from scratch on success. Each case directory receives only:

```text
scf.out
scf.err
pp_elf.out
pp_elf.err
pp_density.out
pp_density.err
summary.csv
slurm-*.out
slurm-*.err
```

`summary.csv` contains:

```text
material,parameter,case,total_energy_ry,max_force_ry_bohr,elf_max,density_at_elf_max
```

The maximum force is the largest per-atom force magnitude, not QE's aggregate `Total force` value. The density is sampled at the ELF maximum using periodic interpolation.

## Submitting jobs

From a material directory, submit all nine parameter cases with:

```bash
sbatch submit_parameter_tests.slurm
```

To submit only one sweep:

```bash
cd Ca2N/PBE/ecutwfc
sbatch submit.slurm
```

To submit the PBE relaxation:

```bash
cd Ca2N/PBE
sbatch submit.slurm
```

The relaxation also runs in scratch and receives a `summary.csv`. The parameter sweeps are fixed-geometry SCF calculations based on the original structure, not chained relaxations.

## Regenerating the workflow

If the parent PBE relaxation inputs or benchmark factors change, regenerate the case directories with:

```bash
python3 dft/generate_parameter_benchmarks.py
```

The generator removes and recreates the three PBE sweep trees and removes any material-level R2SCAN tree.
