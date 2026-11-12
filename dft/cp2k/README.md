# CP2K convergence and void ELF workflow

This is a separate PBE/GTH calculation set for the six structures in `../README.md`,
intended to establish an internally converged CP2K result set. The cases use fixed
geometry and are not parameter-matched to the QE PAW data. The inputs use integer
occupations without a smearing section. Every material uses the same resource
request for a given test value, with larger grids/bases assigned more memory;
all calculation jobs request 3 hours.

## Structure-only void finder

`find_voids.py` uses only fractional atomic positions, lattice vectors, periodic
Voronoi tessellation, and space-group symmetry. It does not read charge density,
ELF, or any other electronic output. It reports symmetry-distinct Voronoi
interstitial candidates, their fractional and Cartesian coordinates, nearest
species/distances, and clearance radii. The default 0.7 Å cutoff removes
near-atom Voronoi vertices; use `--min-radius 0` to inspect every vertex.

Install the small Python requirements in an environment, then run, for example:

```bash
python3 -m pip install -r requirements.txt
python find_voids.py ../structures/mp-2686.cif --output structures/Ca2N/voids.csv
```

The candidates are geometric cavities. The finder does not decide which cavity
contains electride electrons. That is measured later from the CP2K ELF cube.

## Calculation generation

```bash
python generate_workflow.py
```

This reads the existing QE `pbe_vc-relax.in` atomic positions and cell for each
material, preserving the fixed geometry used by the QE sweeps, and creates a
separate `dft/cp2k/<material>/` tree. Cases are:

* cutoff: 300, 400, 500, and 600 Ry (relative cutoffs 40, 50, 60, and 70 Ry), with 16, 16, 20, and 24 GB
* k mesh: 3x3x3, 5x5x5, 7x7x7, and 9x9x9, Gamma centered, with 16, 20, 24, and 32 GB
* basis: one TZVP-MOLOPT-SR-GTH case at 500/60 Ry and 5x5x5, with 24 GB; compare with the matching DZVP baseline

The full ELF cube is requested for each case. After CP2K finishes, the job script
runs `analyze_void_elf.py` against every structural candidate and writes the
local ELF maximum within 1.5 Å of each site. The per-cube CSV report is copied
back to that case directory. ELF cubes remain in the job's scratch directory.

Generated inputs use CP2K's standard `BASIS_MOLOPT`, `BASIS_MOLOPT_UCL`, and `GTH_POTENTIALS`
libraries, which must be available to the CP2K installation on Negishi. The
pseudopotential names are PBE GTH with standard small-core valence choices;
these are a separate valence treatment from the QE PAW set. The generator writes
inputs and submission scripts; it does not launch CP2K or submit jobs.

The electride structures may be metallic. With the requested no-smearing setup,
they use conventional diagonalization and conservative density mixing. SCF
convergence on those structures still needs a real Negishi run to establish.

## Running on Negishi

The generated Slurm scripts load `cp2k/2022.1` and run `cp2k.psmp` with `srun`.
Enter a case directory and submit:

```bash
sbatch submit.slurm
```

Submit all convergence cases for a material from its CP2K directory with:

```bash
sbatch submit_all.slurm
```

Each calculation requests 03:00:00. Jobs stage inputs and output under
`$RCAC_SCRATCH/electride-ml-predictor/cp2k/`, retain the scratch directory and
ELF cube, and copy the CP2K log plus per-void ELF CSV report back to the case
directory. This requires `python3` with NumPy available on the compute node.
Each material has nine DFT cases (four cutoff, four k-point, one basis comparison).
Automated job analysis checks every row in the material's
`structure/void_candidates.csv`; it does not feed ELF values back into the
structure-only site finder.

For example, from a case directory:

```bash
python ../../../analyze_void_elf.py CP2K_ELF_CUBE ../../structure/void_candidates.csv --site 1
```
