#!/bin/bash
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 24:00:00
#SBATCH --qos=long
#SBATCH -J RN18
#SBATCH -o logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs runs/baseline/ResNet18_v1
source .venv/bin/activate

{ time python src/main_baseline.py; } 2>&1 | tee runs/baseline/ResNet18_v1/runtime.txt
