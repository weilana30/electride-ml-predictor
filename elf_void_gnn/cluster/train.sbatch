#!/bin/bash
#SBATCH --account=andre176
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=elfnet-v1
#SBATCH --output=/home/weilana/elfnet-runs/elf_void_gnn/logs/train-%j.out
#SBATCH --error=/home/weilana/elfnet-runs/elf_void_gnn/logs/train-%j.err
set -euo pipefail
module load rocm/6.3.4
cd /home/weilana/electride-ml-predictor
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
/home/weilana/.conda/envs/elfnet-train/bin/python -m elf_void_gnn.train --require-gpu --config elf_void_gnn/configs/elf_center_v1.yaml
