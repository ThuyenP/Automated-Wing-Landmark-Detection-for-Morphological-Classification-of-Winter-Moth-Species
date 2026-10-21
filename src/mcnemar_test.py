import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from ultralytics import YOLO

general_path = "runs/pose"
class_names = {0: "BS", 1: "WM"}

# Read the true species of each image from the YOLO label txt (first token = class id)
def load_true_labels(label_dir):
    true_map = {}
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(label_dir, fname), "r") as f:
            line = f.readline().strip().split()
        if line:
            true_map[os.path.splitext(fname)[0]] = int(line[0])
    return true_map

# Predict one species per image: keep the highest-confidence detection
# conf is set low so (almost) every wing gets a prediction; no detection -> pred = -1
def yolo_predictions(model_type, model_name, img_dir, label_dir, conf=0.001):
    model_path = os.path.join(general_path, model_type, model_name, "weights/best.pt")
    model = YOLO(model_path)
    true_map = load_true_labels(label_dir)

    rows = []
    for res in model.predict(source=img_dir, conf=conf, stream=True, verbose=False):
        stem = os.path.splitext(os.path.basename(res.path))[0]
        if stem not in true_map:
            print(f"Skipping {stem}: no label txt found")
            continue

        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            best = int(boxes.conf.argmax())
            pred, score = int(boxes.cls[best]), float(boxes.conf[best])
        else:
            print(f"  No detections in: {res.path}")
            pred, score = -1, 0.0

        rows.append({"image": stem, "true": true_map[stem], "pred": pred, "confidence": score})

    df = pd.DataFrame(rows).sort_values("image").reset_index(drop=True)
    df["correct"] = (df["true"] == df["pred"]).astype(int)
    return df

# McNemar test on the two discordant counts b and c
# Exact binomial p-value (recommended when b + c < 25) and chi-square with continuity correction
def mcnemar(b, c):
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_chi2": 1.0, "p_exact": 1.0}
    stat = (abs(b - c) - 1) ** 2 / n
    return {
        "b": b,
        "c": c,
        "chi2": stat,
        "p_chi2": float(chi2.sf(stat, df=1)),
        "p_exact": float(binomtest(b, n, 0.5).pvalue),
    }

# Test 1: YOLO prediction vs true species (2x2 confusion matrix)
# H0: P(BS predicted as WM) == P(WM predicted as BS), i.e. the errors are symmetric
def mcnemar_vs_truth(df):
    detected = df[df["pred"] != -1]
    table = pd.crosstab(
        detected["true"].map(class_names), detected["pred"].map(class_names),
        rownames=["True"], colnames=["Predicted"]
    ).reindex(index=["BS", "WM"], columns=["BS", "WM"], fill_value=0)

    b = int(table.loc["BS", "WM"])  # BS misclassified as WM
    c = int(table.loc["WM", "BS"])  # WM misclassified as BS
    return table, mcnemar(b, c)

# Test 2: YOLO vs another model on the same images (correct / incorrect table)
# other_csv needs columns: image, pred (0 = BS, 1 = WM)
# H0: both models have the same error rate
def mcnemar_vs_model(df, other_csv, other_name="other"):
    other = pd.read_csv(other_csv)
    other["image"] = other["image"].astype(str).str.replace(r"\.(png|jpg|jpeg)$", "", regex=True)
    merged = df.merge(other[["image", "pred"]], on="image", suffixes=("", f"_{other_name}"))
    other_correct = (merged["true"] == merged[f"pred_{other_name}"]).astype(int)

    table = pd.crosstab(
        merged["correct"].map({1: "YOLO correct", 0: "YOLO wrong"}),
        other_correct.map({1: f"{other_name} correct", 0: f"{other_name} wrong"})
    )
    b = int(((merged["correct"] == 1) & (other_correct == 0)).sum())  # only YOLO correct
    c = int(((merged["correct"] == 0) & (other_correct == 1)).sum())  # only other correct
    return len(merged), table, mcnemar(b, c)

def report(title, table, res, alpha=0.05):
    lines = [
        f"\n{'='*50}", f"  {title}", f"{'='*50}",
        table.to_string(), "",
        f"  Discordant pairs: b = {res['b']}, c = {res['c']}",
        f"  Chi-square (continuity corrected) = {res['chi2']:.4f}, p = {res['p_chi2']:.4f}",
        f"  Exact binomial p = {res['p_exact']:.4f}",
    ]
    p = res["p_exact"] if res["b"] + res["c"] < 25 else res["p_chi2"]
    lines.append(f"  -> {'Reject' if p < alpha else 'Fail to reject'} H0 at alpha = {alpha} (p = {p:.4f})")
    text = "\n".join(lines)
    print(text)
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="YOLO (v3)")
    parser.add_argument("--model_name", default="yolo26x_200")
    parser.add_argument("--img_dir", default="images (bleached)/val")
    parser.add_argument("--label_dir", default="labels (bleached)/val")
    parser.add_argument("--compare_csv", default=None, help="Other model predictions (image, pred)")
    parser.add_argument("--compare_name", default="other")
    args = parser.parse_args()

    out_dir = os.path.join(general_path, args.model_type, args.model_name, "mcnemar")
    os.makedirs(out_dir, exist_ok=True)

    df = yolo_predictions(args.model_type, args.model_name, args.img_dir, args.label_dir)
    df.to_csv(os.path.join(out_dir, "yolo_predictions.csv"), index=False)
    print(f"Images: {len(df)}, no detection: {(df['pred'] == -1).sum()}, accuracy: {df['correct'].mean():.4f}")

    table, res = mcnemar_vs_truth(df)
    text = report("McNemar: YOLO prediction vs true species", table, res)

    if args.compare_csv:
        n, table2, res2 = mcnemar_vs_model(df, args.compare_csv, args.compare_name)
        text += report(f"McNemar: YOLO vs {args.compare_name} ({n} paired images)", table2, res2)

    with open(os.path.join(out_dir, "mcnemar_results.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nSaved results to {out_dir}")
