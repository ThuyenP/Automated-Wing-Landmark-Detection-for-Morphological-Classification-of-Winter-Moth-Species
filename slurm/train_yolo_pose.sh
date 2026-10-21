#!/bin/bash
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=30G
#SBATCH -t 24:00:00
#SBATCH --qos=long
#SBATCH -J 26x_U
#SBATCH -o logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate
python src/main_yolo_pose.py