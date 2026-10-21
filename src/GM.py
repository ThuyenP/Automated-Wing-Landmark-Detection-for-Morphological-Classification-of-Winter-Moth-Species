import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import morphops as mops
import seaborn as sns
import cv2 as cv
import os

from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

def get_width_height(img_path):
    img = cv.imread(img_path)
    img_h, img_w = img.shape[:2]
    return img_w, img_h

def load_yolo_landmarks(label_dir, img_w, img_h): # Width = 3584, Height = 2748
    all_landmarks = []
    all_files = []
    
    for fname in sorted(os.listdir(label_dir)):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(label_dir, fname)
        with open(fpath, 'r') as f:
            line = f.readline().strip().split()
        
        # YOLO format: class cx cy w h kp1x kp1y kp1v kp2x kp2y kp2v ...
        kp_data = line[5:]  # skip class + bbox
        coords = []
        for i in range(0, len(kp_data), 3):
            x = float(kp_data[i])   * img_w  # denormalize
            y = float(kp_data[i+1]) * img_h
            coords.append([x, y])
        
        all_landmarks.append(coords)
        all_files.append(fname)
    
    return np.array(all_landmarks), all_files  # shape: (n, # landmarks = 24, 2)

def class_balance(id_path, files_name): # id_path = "General Data/Wing IDs - Label.csv"
    df = pd.read_csv(id_path)
    label_map = dict(zip(df['image_name'], df['Encode']))
    labels = np.array([label_map[f.replace('.txt', '')] for f in files_name])
    return labels

def align_shape(landmarks):
    temp_result = mops.gpa(landmarks)
    aligned = temp_result['aligned']
    mean_shape = temp_result['mean']
    return aligned, mean_shape

def PCA_helper(aligned):
    n = aligned.shape[0]
    flat = aligned.reshape(n, -1)  # (n_specimens, n_landmarks * 2)
    pca = PCA(n_components=0.95, whiten=False)
    pca_scores = pca.fit_transform(flat)  
    return flat, pca, pca_scores

def LDA_helper(pca_scores, labels, species_list, output_path="lda_confusion.png"):
    print("Run LDA")
    lda = LinearDiscriminantAnalysis()
    cv_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(lda, pca_scores, labels, cv=cv_split)
    
    # Metrics
    report = classification_report(labels, preds, target_names=species_list.astype(str), output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df.to_string())
    report_df.to_csv("lda_metrics.csv")
    print("Metrics saved to lda_metrics.csv")

    # Confusion matrix
    cm_result = confusion_matrix(labels, preds, labels=species_list)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_result, annot=True, fmt='d',
                xticklabels=species_list,
                yticklabels=species_list,
                cmap='Blues')
    plt.title("Classical GM confusion matrix")
    plt.ylabel("True species")
    plt.xlabel("Predicted species")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    return preds

def eval(preds, labels, species_list, output_dir):
    report = classification_report(labels, preds, 
                                   target_names=species_list.astype(str), 
                                   output_dict=True)
    
    # print to terminal
    print(classification_report(labels, preds, target_names=species_list.astype(str)))
    
    # extract key summary metrics
    acc = report['accuracy']
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # Save the value
    summary_df = pd.DataFrame([{
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }])
    summary_path = os.path.join(output_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    
    return acc, precision, recall, f1

def GM_helper(label_dir, wing_ID, output_dir):
    landmarks, filenames = load_yolo_landmarks(
        label_dir=label_dir,
        img_w=3584,
        img_h=2748
    )

    labels = class_balance(wing_ID, filenames)
    aligned, mean_shape = align_shape(landmarks)
    print("Mean shape:", mean_shape)
    flat, pca, pca_scores = PCA_helper(aligned)
    print(f"Reduced from {flat.shape[1]} to {pca_scores.shape[1]} dimensions")
    print(f"Variance explained: {pca.explained_variance_ratio_.cumsum()[-1]:.3f}")    

    species_list = np.unique(labels)
    colors = cm.tab10(np.linspace(0, 1, len(species_list)))

    plt.figure(figsize=(8, 6))
    for sp, col in zip(species_list, colors):
        mask = labels == sp
        plt.scatter(pca_scores[mask, 0], pca_scores[mask, 1],
                    label=sp, color=col, alpha=0.7, s=40)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("PCA of wing shape — Procrustes coordinates")
    plt.legend(fontsize=8)
    plt.tight_layout()
    pca_path = os.path.join(output_dir, "pca_plot.png")
    plt.savefig(pca_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved to {pca_path}")

    preds = LDA_helper(pca_scores, labels, species_list)

    acc, precision, recall, f1 = eval(preds, labels, species_list, output_dir)

    return acc, precision, recall, f1

# Run GM.py on 305 testing vals
# import os
# import numpy as np
# import pandas as pd
# import morphops as mops
# from sklearn.decomposition import PCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.metrics import classification_report, confusion_matrix

# from GM import load_yolo_landmarks, class_balance, align_shape
# from mcnemar_test import mcnemar_vs_model, report

# class_names = ["BS", "WM"]

# # Align new specimens to the training mean shape the same way gpa does:
# # remove position, scale to unit centroid size, then rotate (reflection allowed) onto the mean
# def align_to_mean(landmarks, mean_shape):
#     X0 = mops.remove_position(landmarks)
#     X0 = mops.remove_scale(X0)
#     return np.array([mops.rotate(x, mean_shape)["aligned"] for x in X0])

# # GM baseline fit on the train wings only, evaluated on the held-out val wings
# # GPA, PCA and LDA are all fit on train; val is only transformed
# def gm_holdout(train_dir, test_dir, wing_ID):
#     train_lmk, train_files = load_yolo_landmarks(train_dir, img_w=3584, img_h=2748)
#     test_lmk, test_files = load_yolo_landmarks(test_dir, img_w=3584, img_h=2748)
#     y_train = class_balance(wing_ID, train_files)
#     y_test = class_balance(wing_ID, test_files)

#     train_aligned, mean_shape = align_shape(train_lmk)
#     test_aligned = align_to_mean(test_lmk, mean_shape)

#     pca = PCA(n_components=0.95, whiten=False)
#     train_scores = pca.fit_transform(train_aligned.reshape(len(train_aligned), -1))
#     test_scores = pca.transform(test_aligned.reshape(len(test_aligned), -1))
#     print(f"PCA: {train_scores.shape[1]} components, variance explained {pca.explained_variance_ratio_.sum():.3f}")

#     lda = LinearDiscriminantAnalysis().fit(train_scores, y_train)
#     preds = lda.predict(test_scores)

#     # CSV Encode: 1 = BS, 2 = WM  ->  YOLO class id: 0 = BS, 1 = WM
#     return pd.DataFrame({
#         "image": [f.replace(".txt", "") for f in test_files],
#         "true": y_test - 1,
#         "pred": preds - 1,
#     })

# if __name__ == "__main__":
#     out_dir = "runs/baseline/GM_holdout"
#     os.makedirs(out_dir, exist_ok=True)

#     gm = gm_holdout("labels (bleached)/train", "labels (bleached)/val", "General Data/Wing IDs - Label.csv")
#     gm.to_csv(os.path.join(out_dir, "gm_predictions.csv"), index=False)

#     report_df = pd.DataFrame(classification_report(gm["true"], gm["pred"], labels=[0, 1], target_names=class_names,
#                                                    output_dict=True, zero_division=0)).transpose()
#     report_df.to_csv(os.path.join(out_dir, "classification_metrics.csv"))
#     print(f"\nGM held-out ({len(gm)} wings)")
#     print(pd.DataFrame(confusion_matrix(gm["true"], gm["pred"], labels=[0, 1]),
#                        index=["True BS", "True WM"], columns=["Pred BS", "Pred WM"]).to_string())
#     print()
#     print(report_df.round(4).to_string())

#     text = ""
#     for name in ["yolo26x_200", "yolo11x_200"]:
#         yolo = pd.read_csv(os.path.join("runs/pose/YOLO (v3)", name, "mcnemar/yolo_predictions.csv"))
#         n, table, res = mcnemar_vs_model(yolo, os.path.join(out_dir, "gm_predictions.csv"), "GM")
#         text += report(f"McNemar: {name} vs GM held-out ({n} paired images)", table, res)
#     with open(os.path.join(out_dir, "mcnemar_yolo_vs_gm.txt"), "w") as f:
#         f.write(text + "\n")
