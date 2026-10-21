import os
from ultralytics import YOLO
import pandas as pd 

general_path = "runs/pose"

# Max recall
def mRecall(model_type, model_name, isBox=True): 
    results_path = os.path.join(general_path, model_type, model_name, "train/results.csv")
    results = pd.read_csv(results_path)
    return results["metrics/recall(B)"].max() if isBox else results["metrics/recall(P)"].max()

# Max precision
def mPrecision(model_type, model_name, isBox=True):
    results_path = os.path.join(general_path, model_type, model_name, "train/results.csv")
    results = pd.read_csv(results_path)
    return results["metrics/precision(B)"].max() if isBox else results["metrics/precision(P)"].max()

# Max mAP50
def mAP50(model_type, model_name, isBox=True):
    results_path = os.path.join(general_path, model_type, model_name, "train/results.csv")
    results = pd.read_csv(results_path)
    return results["metrics/mAP50(B)"].max() if isBox else results["metrics/mAP50(P)"].max()

# Max mAP50-95
def mAP50_95(model_type, model_name, isBox=True):
    results_path = os.path.join(general_path, model_type, model_name, "train/results.csv")
    results = pd.read_csv(results_path)
    return results["metrics/mAP50-95(B)"].max() if isBox else results["metrics/mAP50-95(P)"].max()

# Predict the image labels and points coords given img_source (either a single img or a img folder)
# Return: an array of individiual img prediction, including class_id, class_name, and confidence
def predict_helper(model_type, model_name, img_source):
    result = []
    
    model_path = os.path.join(general_path, model_type, model_name, "weights/best.pt")
    project_path = os.path.join(general_path, model_type, model_name)  # Fix: remove extra "predict" nesting
    
    model = YOLO(model_path)
    predict_res = model.predict(
        source=img_source,
        save=True,
        show=False,        # Fix: set False since no display available
        conf=0.5,          # Fix: lower threshold to see if detections appear
        project=project_path,
        name="predict"
    )
    
    for res in predict_res:
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]

                result.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence
                })
        else:
            print(f"  No detections in: {res.path}")

    # Print results
    if result:
        print(f"\n{'='*40}")
        print(f"  Classification Results")
        print(f"{'='*40}")
        for i, r in enumerate(result):
            print(f"  [{i+1}] Class:      {r['class_name']} (id={r['class_id']})")
            print(f"       Confidence: {r['confidence']:.2%}")
            print(f"  {'-'*38}")
    else:
        print("\n No detections found. Try lowering the conf threshold further.")

    return result