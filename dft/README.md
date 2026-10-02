# Geometry-optimization benchmark

This directory contains 18 `pw.x` input files: three geometry relaxations for each of three electrides and three controls.

## Selected structures

| Role | Structure | Dataset ID | Why it is included |
| --- | --- | --- | --- |
| Electride | Ba2As (0D) | ELD-0003 / mp-2768 | Binary, nonmagnetic 0D baseline. |
| Electride | Ba3CrN3 (1D) | ELD-0014 / mp-12905 | 1D transition-metal nitride; its inputs seed antiferromagnetism. |
| Electride | Ca2N (2D) | ELD-0049 / mp-2686 | Canonical layered 2D electride. |
| Control | MgS | mp-1315 | Simple ionic binary. |
| Control | BaSn3 | mp-7759 | Metallic Ba-containing intermetallic. |
| Control | Cu2SiSe3 | mp-15896 | Ternary covalent/chalcogenide semiconductor. |

The CIFs are P1 primitive-cell exports. Inputs intentionally retain those primitive cells through `ibrav=0`, explicit `CELL_PARAMETERS`, and fractional positions. Do not replace them with conventional cells mid-benchmark.

## Pseudopotentials: downloaded

Scalar-relativistic, norm-conserving ONCV pseudopotentials are installed under:

```text
pseudo/PBE/{As,Ba,Ca,Cr,Cu,Mg,N,S,Se,Si,Sn}.upf
pseudo/PBEsol/{As,Ba,Ca,Cr,Cu,Mg,N,S,Se,Si,Sn}.upf
```

They were downloaded from the curated KSPP mirror of the PseudoDojo `nc-sr-04` standard UPF tables. The QE pseudopotential site lists PseudoDojo as an available tested library, and its meta-GGA documentation requires norm-conserving pseudopotentials. Each required UPF was checked for the expected element, exchange-correlation label, and `pseudo_type="NC"` header.

- PBE: `norm-conserving/nc-sr-04/PBE/standard/upf/<Element>.upf`
- PBEsol: `norm-conserving/nc-sr-04/PBEsol/standard/upf/<Element>.upf`

For r2SCAN, these inputs intentionally reuse the PBE **norm-conserving** table. This is the standard practical choice when r2SCAN-specific PPs are unavailable, and it keeps the PBE/r2SCAN comparison internally controlled. Do not substitute SSSP PAW or ultrasoft PPs into only the r2SCAN jobs: QE's meta-GGA support requires norm-conserving PPs.

## Quantum ESPRESSO requirement

Use a current QE build with r2SCAN/LibXC support. The r2SCAN inputs use:

```text
input_dft = 'r2scan'
```

Test this first with the 3-atom Ca2N input. If QE reports that `r2scan` is unknown, the executable lacks the required support and must be rebuilt or replaced with a LibXC-enabled build. Do not silently change the r2SCAN inputs to PBE.

## Initial numerical settings

- All jobs are `vc-relax`, use BFGS ionic and cell updates, and relax to 0 kbar.
- The initial k mesh targets no more than about 0.15 1/angstrom reciprocal-space spacing.
- PBE/PBEsol begin at 100/400 Ry wavefunction/charge-density cutoffs.
- r2SCAN begins at 100/800 Ry to provide a denser grid for the kinetic-energy-density term.
- Metallic and electride candidates use Marzari-Vanderbilt smearing, 0.02 Ry. The two closed-shell controls (MgS and Cu2SiSe3) use fixed occupations. Keep this choice fixed across the three functionals for a given structure.

These are intentionally conservative **starting** settings, not a convergence claim. Before comparing optimized geometries, converge `ecutwfc`, `ecutrho`, the k-point mesh, and smearing for representative metallic and insulating structures. Use the most demanding converged values consistently across PBE, PBEsol, and r2SCAN for a given material.

## Run

From any structure folder, for example:

```bash
mkdir -p out
pw.x -in pbe_vc-relax.in > pbe_vc-relax.out
pw.x -in pbesol_vc-relax.in > pbesol_vc-relax.out
pw.x -in r2scan_vc-relax.in > r2scan_vc-relax.out
```

Run the three relaxations from the same CIF-derived starting structure. Do not chain the r2SCAN job from the PBE-relaxed geometry during this benchmark.

## SCF parameter sweeps

Each material/functional directory also contains three SCF sweep directories:

- `ecutwfc/`: low, baseline, and high wavefunction cutoffs, with `ecutrho` scaled by the same factor.
- `kpoints/`: low, baseline, and high automatic Monkhorst-Pack meshes.
- `smearing_or_conv_thr/`: three `degauss` values for metallic/smearing inputs, or three `conv_thr` values for fixed-occupation inputs.

Each sweep directory has one `submit.slurm` launcher that runs its three SCF inputs sequentially and prints per-case and total elapsed wall time. The material-level `submit_parameter_tests.slurm` launcher submits all nine sweep jobs (three functionals × three parameter directories). Run it from the material directory or submit it with `sbatch`.

The sweep inputs are independent SCF calculations at the original CIF-derived geometry; they do not perform ionic or cell relaxation. They use the same functional, pseudopotentials, magnetic seed, and structure as their parent input.

## Magnetic caveat: Ba3CrN3

Ba3CrN3 uses two labels, `Cr1` and `Cr2`, with identical Cr pseudopotentials and opposite initial moments. That is an antiferromagnetic seed, not proof that AFM is its ground state. After this three-functional geometry comparison, run a small magnetic-ordering check (AFM seed versus FM seed) at the chosen production level. DFT+U is a separate sensitivity study and is not folded into this PBE/PBEsol/r2SCAN geometry benchmark.

## Sources consulted

- QE `pw.x` input reference: https://www.quantum-espresso.org/Doc/INPUT_PW.html
- QE pseudopotential guidance: https://www.quantum-espresso.org/pseudopotentials/
- PseudoDojo/KSPP NC-SR 0.4.1 tables: https://github.com/Quantum-MultiScale/KSPP/tree/main/norm-conserving/nc-sr-04
- QE users discussion of r2SCAN/LibXC: https://lists.quantum-espresso.org/pipermail/users/2023-June/050558.html
