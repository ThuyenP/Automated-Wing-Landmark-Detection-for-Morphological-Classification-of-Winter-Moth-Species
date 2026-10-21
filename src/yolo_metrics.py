import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

general_path = "runs/pose"
class_names = ["BS", "WM"]

# Per-species precision / recall / F1 from the per-image predictions saved by mcnemar_test.py
# Images with no detection (pred = -1) count as wrong
def yolo_metrics(model_type, model_name):
    pred_dir = os.path.join(general_path, model_type, model_name, "mcnemar")
    df = pd.read_csv(os.path.join(pred_dir, "yolo_predictions.csv"))

    report = classification_report(df["true"], df["pred"], labels=[0, 1],
                                   target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(pred_dir, "classification_metrics.csv"))

    print(f"\n{'='*50}\n  {model_name}  ({len(df)} images, no detection: {(df['pred'] == -1).sum()})\n{'='*50}")
    print(pd.DataFrame(confusion_matrix(df["true"], df["pred"], labels=[0, 1]),
                       index=["True BS", "True WM"], columns=["Pred BS", "Pred WM"]).to_string())
    print()
    print(report_df.round(4).to_string())
    return report_df

if __name__ == "__main__":
    for name in ["yolo11x_200", "yolo26x_200"]:
        yolo_metrics("YOLO (v3)", name)
