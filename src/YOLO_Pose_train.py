# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# # Train the model
# results = model.train(data="pre-train/dataset.yaml", epochs=2)

from ultralytics import YOLO
import multiprocessing
import torch
torch.backends.cuda.preferred_linalg_library("magma")
epoch = 200

def yolo_train_helper(model_name, dataset_path, save_dir, sub_dir, epochs=epoch):
    model = YOLO(model_name)
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        patience = int(1/4 * epochs),
        batch = 8, 
        workers=0, 
        mosaic=0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        flipud=0.0, fliplr=0.3,
        scale=0.1, translate=0.05, 
        project=save_dir, 
        name=sub_dir
    )


def yolo_train(model_name, dataset_path, save_dir, sub_dir, epochs=epoch):
    multiprocessing.freeze_support()
    yolo_train_helper(model_name, dataset_path, save_dir, sub_dir, epochs)
