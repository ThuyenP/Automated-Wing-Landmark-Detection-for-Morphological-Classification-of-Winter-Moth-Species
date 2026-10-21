from sklearn.model_selection import train_test_split
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import shutil
import multiprocessing

from preprocessing import clahe_transform, sync_name
from COCO_converter import COCO_convert, fill_missing, decrement_first_number_in_files
from YOLO_Pose_train import yolo_train

# Source - https://stackoverflow.com/a/66642969
# Posted by Datexland, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-03, License - CC BY-SA 4.0

# Helper method to split the image into train and train folder
def split_img_label(data_train, data_test, folder_train, folder_test):
    
    os.makedirs(folder_train, exist_ok=True)
    os.makedirs(folder_test, exist_ok=True) 
    
    train_ind=list(data_train.index)
    test_ind=list(data_test.index)
    
    # Train folder
    for i in tqdm(range(len(train_ind))):
        src = str(data_train[train_ind[i]])
        dst = os.path.join(folder_train, os.path.basename(src))
        shutil.copy2(src, dst)

    # Test folder
    for j in tqdm(range(len(test_ind))):
        src = str(data_test[test_ind[j]])
        dst = os.path.join(folder_test, os.path.basename(src))
        shutil.copy2(src, dst)

# Split the coco format accordingly
def split_coco(src_coco, train_dir, test_dir, folder_train, folder_test):

    os.makedirs(folder_train, exist_ok=True)
    os.makedirs(folder_test, exist_ok=True) 

    for fname in os.listdir(test_dir):
        stem = os.path.splitext(fname)[0]
        src_txt = os.path.join(src_coco, stem + '.txt')
        des_txt = os.path.join(folder_test, stem + '.txt')

        shutil.copy2(src_txt, des_txt)

    for fname in os.listdir(train_dir):
        stem = os.path.splitext(fname)[0]
        src_txt = os.path.join(src_coco, stem + '.txt')
        des_txt = os.path.join(folder_train, stem + '.txt')

        shutil.copy2(src_txt, des_txt)

# Split train and test dataset
def split(img_path, src_coco):
    print(len(src_coco))
    # Source - https://stackoverflow.com/a/66642969
    # Posted by Datexland, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-02-03, License - CC BY-SA 4.0        
    list_img=[img for img in os.listdir(img_path) if img.endswith('.png')==True]
    # list_txt=[img for img in os.listdir(PATH) if img.endswith('.txt')==True]

    path_img=[]

    for img_name in list_img:
        stem = os.path.splitext(img_name)[0]  
        txt_name = stem + ".txt"
        txt_path = os.path.join(src_coco, txt_name)

        # only keep images that have a matching txt in src_coco
        if os.path.exists(txt_path):
            path_img.append(os.path.join(img_path, img_name))
        
    df=pd.DataFrame(path_img)

    # split (with no validation)
    data_train, data_test, labels_train, labels_test = train_test_split(df[0], df.index, test_size=0.20, random_state=42)

    # Function split 
    split_img_label(data_train, data_test,'./images/train',"./images/val")
    split_coco(src_coco, './images/train', "./images/val", './labels/train', './labels/val')
    
# Preprocessing and splitting the data
def preprocess(raw_img, raw_label, coco_label, species_label):
    # # Syncing the data file name
    # print("Syncing the file name")
    # sync_name(raw_img)
    # sync_name(raw_label)

    # # COCO convert all the annotated file
    # print("Start filling out missing data points")
    # fill_missing(raw_label, raw_label)

    # print("Start COCO Conversion")
    # COCO_convert(raw_label, coco_label, species_label)

    # # Split train and test dataset
    print("Start splitting")
    split(raw_img, coco_label)

    # clahe_convert the images
    print("CLAHE Transformation")
    clahe_transform("images/train", "images/train")
    clahe_transform("images/val", "images/val")
    
def split_species(folder_BS, folder_WM, image_dir, label_csv):
    os.makedirs(folder_BS, exist_ok=True)
    os.makedirs(folder_WM, exist_ok=True)

    # Read CSV
    df = pd.read_csv(label_csv)

    # Optional: clean column names in case of extra spaces
    df.columns = df.columns.str.strip()

    # Build mapping: image_name -> Encode
    label_map = dict(zip(df["image_name"], df["Encode"]))

    image_files = [
        fname for fname in os.listdir(image_dir)
        if fname.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for fname in tqdm(image_files, desc="Splitting images"):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        stem = os.path.splitext(fname)[0]
        src = os.path.join(image_dir, fname)

        if stem not in label_map:
            print(f"Skipping {fname}: no label found in CSV")
            continue

        label_id = label_map[stem]

        if label_id == 1:
            dst = os.path.join(folder_BS, fname)
            shutil.move(src, dst)
        elif label_id == 2:
            dst = os.path.join(folder_WM, fname)
            shutil.move(src, dst)
        else:
            print(f"Skipping {fname}: unknown label_id = {label_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26x-pose.pt")
    parser.add_argument("--sub_dir", default="yolo26x_200")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--project", default="YOLO_unbleached")
    parser.add_argument("--data", default="configs/dataset.yaml")
    args = parser.parse_args()

    # Set path
    raw_img = "General Data/raw_img"
    coco_label = "General Data/coco_labels"
    raw_label = "General Data/raw_labels"
    species_label = "General Data/Wing IDs - Label.csv"

    unbleached_img = "General Data/unbleached_img"

    runTrain_dir = args.project
    sub_dir = args.sub_dir

    dataset_path = args.data

    # Preprocess the data
    # preprocess(unbleached_img, raw_label, coco_label, species_label)
    # sync_name(unbleached_img)
    

    # # Train the model
    model_name = args.model
    yolo_train(model_name, dataset_path, runTrain_dir, sub_dir, args.epochs)
    # split_species("ResNet labels/test/BS", "ResNet labels/test/WM", "ResNet labels/test", species_label)
    # print("Train data")
    # split_species("ResNet labels/train/BS", "ResNet labels/train/WM", "ResNet labels/train", species_label)
    