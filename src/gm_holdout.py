import os
import numpy as np
import pandas as pd
import morphops as mops
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix

from GM import load_yolo_landmarks, class_balance, align_shape
from mcnemar_test import mcnemar_vs_model, report

class_names = ["BS", "WM"]

# Align new specimens to the training mean shape the same way gpa does:
# remove position, scale to unit centroid size, then rotate (reflection allowed) onto the mean
def align_to_mean(landmarks, mean_shape):
    X0 = mops.remove_position(landmarks)
    X0 = mops.remove_scale(X0)
    return np.array([mops.rotate(x, mean_shape)["aligned"] for x in X0])

# GM baseline fit on the train wings only, evaluated on the held-out val wings
# GPA, PCA and LDA are all fit on train; val is only transformed
def gm_holdout(train_dir, test_dir, wing_ID):
    train_lmk, train_files = load_yolo_landmarks(train_dir, img_w=3584, img_h=2748)
    test_lmk, test_files = load_yolo_landmarks(test_dir, img_w=3584, img_h=2748)
    y_train = class_balance(wing_ID, train_files)
    y_test = class_balance(wing_ID, test_files)

    train_aligned, mean_shape = align_shape(train_lmk)
    test_aligned = align_to_mean(test_lmk, mean_shape)

    pca = PCA(n_components=0.95, whiten=False)
    train_scores = pca.fit_transform(train_aligned.reshape(len(train_aligned), -1))
    test_scores = pca.transform(test_aligned.reshape(len(test_aligned), -1))
    print(f"PCA: {train_scores.shape[1]} components, variance explained {pca.explained_variance_ratio_.sum():.3f}")

    lda = LinearDiscriminantAnalysis().fit(train_scores, y_train)
    preds = lda.predict(test_scores)

    # CSV Encode: 1 = BS, 2 = WM  ->  YOLO class id: 0 = BS, 1 = WM
    return pd.DataFrame({
        "image": [f.replace(".txt", "") for f in test_files],
        "true": y_test - 1,
        "pred": preds - 1,
    })

if __name__ == "__main__":
    out_dir = "runs/baseline/GM_holdout"
    os.makedirs(out_dir, exist_ok=True)

    gm = gm_holdout("labels (bleached)/train", "labels (bleached)/val", "General Data/Wing IDs - Label.csv")
    gm.to_csv(os.path.join(out_dir, "gm_predictions.csv"), index=False)

    report_df = pd.DataFrame(classification_report(gm["true"], gm["pred"], labels=[0, 1], target_names=class_names,
                                                   output_dict=True, zero_division=0)).transpose()
    report_df.to_csv(os.path.join(out_dir, "classification_metrics.csv"))
    print(f"\nGM held-out ({len(gm)} wings)")
    print(pd.DataFrame(confusion_matrix(gm["true"], gm["pred"], labels=[0, 1]),
                       index=["True BS", "True WM"], columns=["Pred BS", "Pred WM"]).to_string())
    print()
    print(report_df.round(4).to_string())

    text = ""
    for name in ["yolo26x_200", "yolo11x_200"]:
        yolo = pd.read_csv(os.path.join("runs/pose/YOLO (v3)", name, "mcnemar/yolo_predictions.csv"))
        n, table, res = mcnemar_vs_model(yolo, os.path.join(out_dir, "gm_predictions.csv"), "GM")
        text += report(f"McNemar: {name} vs GM held-out ({n} paired images)", table, res)
    with open(os.path.join(out_dir, "mcnemar_yolo_vs_gm.txt"), "w") as f:
        f.write(text + "\n")
