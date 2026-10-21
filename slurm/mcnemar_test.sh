#!/bin/bash
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J mcnemar
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source .venv/bin/activate
hostname; date
which python && python --version
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Edit these to point at your validation images and labels
IMG_DIR="images (bleached)/val"
LABEL_DIR="labels (bleached)/val"

# YOLO11x vs true species
python src/mcnemar_test.py --model_type "YOLO (v3)" --model_name yolo11x_200 \
    --img_dir "$IMG_DIR" --label_dir "$LABEL_DIR"

# YOLO26x vs true species, plus paired YOLO26x vs YOLO11x
python src/mcnemar_test.py --model_type "YOLO (v3)" --model_name yolo26x_200 \
    --img_dir "$IMG_DIR" --label_dir "$LABEL_DIR" \
    --compare_csv "runs/pose/YOLO (v3)/yolo11x_200/mcnemar/yolo_predictions.csv" --compare_name YOLO11x