# Automated-Wing-Landmark-Detection-for-Morphological-Classification-of-Winter-Moth-Species

Automatically identify the species of a wing (**BS** or **WM**) from a microscope image, and locate 24 wing landmarks.

This repository contains the code for the honors thesis *<thesis title>* (<your name>, UMass Amherst, 2026). The model is a YOLO26x-Pose network trained on 1,218 hand-landmarked wing images. On 305 held-out validation wings it classifies species with **94.1% accuracy**, compared with **90.2%** for a classical geometric morphometrics (GM) baseline.

> **BS** = *O. bruceata* · **WM** = *O. brumata*

---

## Contents

1. [Predict species for your own images](#1-predict-species-for-your-own-images) ← start here if you just want results
2. [Understanding the output](#2-understanding-the-output)
3. [Results](#3-results)
4. [Reproducing the thesis](#4-reproducing-the-thesis)
5. [Repository layout](#5-repository-layout)
6. [Citation, license and contact](#6-citation-license-and-contact)

---

## 1. Predict species for your own images

### What your images need to look like

The model was trained on images with these properties. Images that look very different may get wrong predictions.

- **Format:** PNG or JPG
- **Size:** 3584 × 2748 pixels (other sizes work, but accuracy was only measured at this size)
- **Content:** one wing per image, photographed the same way as the training images
- **No pre-processing needed:** the prediction script applies the same contrast enhancement (CLAHE) that was used during training. Do **not** enhance the images yourself.

### Option A — Google Colab (no installation)

1. Open the notebook: **<Colab link>**
2. Click **Runtime → Run all**.
3. When asked, upload your wing images.
4. When it finishes, download `predictions.csv`.

### Option B — On your own computer or on Unity

You need Python 3.12. Run these commands once to set things up:

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then, every time you want predictions:

```bash
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python predict.py --images path/to/your/wing/folder
```

The trained model (`best.pt`, 128 MB) downloads automatically the first time. You can also download it by hand from the [Releases page](https://github.com/<your-username>/<repo-name>/releases) and put it in `models/`.

> **On Unity:** run predictions inside a job or an interactive session (`salloc`), not on the login node.

---

## 2. Understanding the output

`predictions.csv` has one row per image:

| Column | Meaning |
|---|---|
| `image` | File name |
| `species` | Predicted species: `BS` or `WM` |
| `confidence` | How sure the model is, from 0 to 1 |

A folder of annotated images is also saved, showing the detected wing and its 24 landmarks so you can check each prediction by eye.

**How much to trust it:** on the validation set the model was wrong on about 6 of every 100 wings. Treat low-confidence predictions (for example below 0.5) as "needs a human check".

---

## 3. Results

All numbers are on the same 305 validation wings (147 BS, 158 WM).

| Model | Accuracy | Macro F1 | BS recall | WM recall |
|---|---|---|---|---|
| **YOLO26x-Pose** | **0.941** | **0.941** | 0.952 | 0.930 |
| YOLO11x-Pose | 0.934 | 0.934 | 0.966 | 0.905 |
| GM baseline (Procrustes + PCA + LDA) | 0.902 | 0.902 | 0.905 | 0.899 |

**McNemar's tests** (paired, same 305 wings):

| Comparison | χ² | p | Conclusion |
|---|---|---|---|
| YOLO26x vs GM | 5.04 | 0.025 | YOLO26x significantly more accurate |
| YOLO11x vs GM | 3.38 | 0.066 | No significant difference |
| YOLO26x vs YOLO11x | 0.06 | 0.80 | No significant difference |
| YOLO11x errors (BS→WM vs WM→BS) | 4.05 | 0.044 | YOLO11x leans toward predicting BS |
| YOLO26x errors (BS→WM vs WM→BS) | 0.50 | 0.48 | Errors balanced between species |

**Notes**

- The validation set was also used to pick the best training checkpoint, so these are validation (not fully independent test) results.
- The GM baseline classifies from the hand-placed landmarks; YOLO works directly from the image.
- Full outputs are in [`results/`](results/).

---

## 4. Reproducing the thesis

### 4.1 Get the data

The images (~21 GB) are too large for GitHub and are stored at: **<data link>**

Put them in this layout at the top of the repository (or edit the paths in the scripts):

```
images (bleached)/train/   1,218 training images (.png)
images (bleached)/val/       305 validation images (.png)
labels (bleached)/train/   matching YOLO label files (.txt)
labels (bleached)/val/
General Data/Wing IDs - Label.csv   species of every wing (1 = BS, 2 = WM)
```

For **training** YOLO (step 2), the folders must be named `images/` and `labels/` instead of `images (bleached)/` and `labels (bleached)/`, because that is what [`configs/dataset.yaml`](configs/dataset.yaml) expects. Rename them (or make a copy), then set `path:` in `configs/dataset.yaml` to the folder that holds them. The evaluation and GM scripts (steps 3–5) use the `(bleached)` names by default.

> **On Unity:** keep the images in a scratch workspace, not in `/project` or your home folder. See the [HPC workspace documentation](https://unityhpc.org/documentation/managing-files/hpc-workspace/).

### 4.2 Install

Same as in [Option B](#option-b--on-your-own-computer-or-on-unity) above. If you want to train on a GPU, install PyTorch for your CUDA version from [pytorch.org](https://pytorch.org) **before** running `pip install -r requirements.txt`. The thesis used `torch 2.7.1+cu118`.

### 4.3 Run the pipeline

Run every command from the top of the repository. On Unity, submit the `slurm/` scripts with `sbatch` instead of running Python directly.

| Step | What it does | Command |
|---|---|---|
| 1. Prepare data | Convert landmark files to YOLO format (`src/COCO_converter.py`), split 80/20 into train/val (`random_state=42`), apply CLAHE (`src/preprocessing.py`) | Enable the steps you need in `preprocess()` in `src/main_yolo_pose.py`, then run it |
| 2. Train YOLO | Fine-tune `yolo26x-pose.pt` for up to 200 epochs (early stopping after 50 without improvement) | `sbatch slurm/train_yolo_pose.sh` |
| 3. Evaluate YOLO | Predict species for each validation wing; McNemar's tests | `sbatch slurm/mcnemar_test.sh` |
| 4. Metrics | Precision, recall, F1 per species | `python src/yolo_metrics.py` |
| 5. GM baseline | Procrustes alignment + PCA + LDA trained on train wings, tested on val wings; McNemar vs YOLO | `python src/gm_holdout.py` |
| 6. ResNet baseline | Fine-tune ResNet18 on the images | `sbatch slurm/train_baseline.sh` |

Step 3 expects the trained weights at `runs/pose/YOLO (v3)/<model>/weights/best.pt`. Either train them in step 2 or download them from the Releases page.

**Training settings used in the thesis:** batch size 8, image augmentation limited to horizontal flip (p = 0.3), scale ±0.1 and translate ±0.05, with no mosaic, colour or vertical-flip augmentation (see `src/YOLO_Pose_train.py`).

---

## 5. Repository layout

```
├── predict.py           Predict species for new images (start here)
├── notebooks/           Google Colab version of predict.py
├── src/                 All thesis code
│   ├── preprocessing.py     CLAHE contrast enhancement, file renaming
│   ├── COCO_converter.py    Landmark files → YOLO label format
│   ├── main_yolo_pose.py    Data split + training entry point
│   ├── YOLO_Pose_train.py   YOLO training settings
│   ├── evaluation.py        Training metrics and prediction helpers
│   ├── mcnemar_test.py      Per-wing YOLO predictions + McNemar's test
│   ├── yolo_metrics.py      Precision / recall / F1
│   ├── GM.py                Geometric morphometrics functions
│   ├── gm_holdout.py        GM baseline on the validation wings
│   ├── resNet.py            ResNet baseline
│   ├── baseline_model.py    Baseline wrappers
│   └── main_baseline.py     Baseline entry point
├── slurm/               Job scripts for the Unity cluster
├── configs/             dataset.yaml for YOLO
├── models/              Trained weights go here (download from Releases)
├── results/             Final metrics and McNemar outputs
└── docs/                Extra notes
```

---

## 6. Citation, license and contact

If you use this code or model, please cite:

```
Thuyen Pham. (2026). Automated Wing Landmark Detection for Morphological Classification of Winter Moth Species. Honors thesis, University of Massachusetts Amherst.
```

Questions or problems: open an issue on this repository, or contact Thuyen Pham at thuyenpham@umass.edu.

Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [morphops](https://github.com/vaipatel/morphops).
