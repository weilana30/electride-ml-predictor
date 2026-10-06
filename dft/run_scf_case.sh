#!/bin/bash
set -euo pipefail

project_root=$1
case_dir=$2
input_name=$3
material=$4
parameter=$5
label=$6

module load quantum-espresso
export OMPI_MCA_pml=ob1
export OMPI_MCA_btl=self,tcp

scratch_base="${RCAC_SCRATCH:-/scratch/negishi/$USER}/electride-ml-predictor"
scratch_dir="$scratch_base/${material}_${parameter}_${label}_${SLURM_JOB_ID}"
mkdir -p "$scratch_dir"

copy_outputs() {
  for output in scf.out scf.err pp_elf.out pp_elf.err pp_density.out pp_density.err summary.csv; do
    if [[ -f "$scratch_dir/$output" ]]; then
      cp "$scratch_dir/$output" "$case_dir/$output"
    fi
  done
}

finish() {
  status=$?
  copy_outputs
  if [[ "$status" -eq 0 ]]; then
    rm -rf "$scratch_dir"
  else
    echo "Scratch files retained at $scratch_dir" >&2
  fi
  exit "$status"
}
trap finish EXIT

pseudo_dir="$project_root/dft/pseudo/PBE_PAW"
sed -E \
  -e "s#outdir[[:space:]]*=[[:space:]]*'[^']*'#outdir = './qe_data'#g" \
  -e "s#pseudo_dir[[:space:]]*=[[:space:]]*'[^']*'#pseudo_dir = '$pseudo_dir'#g" \
  "$case_dir/$input_name" > "$scratch_dir/scf.in"

cd "$scratch_dir"
srun pw.x -in scf.in > scf.out 2> scf.err

prefix=$(sed -n "s/.*prefix[[:space:]]*=[[:space:]]*'\\([^']*\\)'.*/\\1/p" scf.in | head -1)
if [[ -z "$prefix" ]]; then
  echo "Could not determine QE prefix from scf.in" >&2
  exit 1
fi

cat > pp_elf.in <<EOF
&INPUTPP
  prefix = '$prefix',
  outdir = './qe_data',
  plot_num = 8,
/
&PLOT
  iflag = 3,
  output_format = 6,
  fileout = 'elf.cube',
/
EOF

cat > pp_density.in <<EOF
&INPUTPP
  prefix = '$prefix',
  outdir = './qe_data',
  plot_num = 0,
/
&PLOT
  iflag = 3,
  output_format = 6,
  fileout = 'density.cube',
/
EOF

srun pp.x -in pp_elf.in > pp_elf.out 2> pp_elf.err
srun pp.x -in pp_density.in > pp_density.out 2> pp_density.err

python3 "$project_root/dft/analyze_elf_density.py" \
  elf.cube density.cube \
  --label "$label" \
  --output cube_analysis.csv

total_energy=$(awk '/!.*total energy/ {energy=$5} END {print energy}' scf.out)
max_force=$(awk '
/Forces acting on atoms/ {if (seen) last=current; seen=1; current=0; next}
seen && /^[[:space:]]*atom[[:space:]]+[0-9]+.*force =/ {
  magnitude=sqrt($7*$7+$8*$8+$9*$9)
  if (magnitude > current) current=magnitude
}
END {if (seen) {last=current; printf "%.12g", last}}
' scf.out)
elf_max=$(awk -F, 'END {print $4}' cube_analysis.csv)
density_at_elf_max=$(awk -F, 'END {print $5}' cube_analysis.csv)

if [[ -z "$total_energy" || -z "$max_force" || -z "$elf_max" || -z "$density_at_elf_max" ]]; then
  echo "Could not extract all summary values" >&2
  exit 1
fi

cat > summary.csv <<EOF
material,parameter,case,total_energy_ry,max_force_ry_bohr,elf_max,density_at_elf_max
$material,$parameter,$label,$total_energy,$max_force,$elf_max,$density_at_elf_max
EOF
