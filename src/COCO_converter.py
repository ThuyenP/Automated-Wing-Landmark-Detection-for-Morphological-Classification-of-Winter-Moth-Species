import os
import re
import cv2
from tqdm import tqdm
import pandas as pd

# Known since all the images I collected are saved in the same format and size
img_width = 3548
img_height = 2748

# Fill in missing points in the input file
# In_path: path of the original labeled .txt file
# Out_path: path of the fixed labeled .txt file
def helper_fill_missing(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        m = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*$", line)
        if m:
            idx, x, y = map(int, m.groups())
            rows.append((idx, x, y))

    # Map existing rows by id
    row_map = {idx: (x, y) for idx, x, y in rows}

    # Build full 1..25, inserting missing as (0,0)
    full = []
    for idx in range(1, 25):
        x, y = row_map.get(idx, (0, 0))
        full.append((idx, x, y))

    # Rewrite file: keep everything but replace the landmarks block content
    out_lines = []
    inside = False
    for line in lines:
        if "<landmarks.pixel" in line:
            inside = True
            out_lines.append(line)
            # write our full 1..25 block
            for idx, x, y in full:
                out_lines.append(f"\t\t{idx}\t{x}\t{y}\n")
            continue
        if inside and "</landmarks.pixel>" in line:
            inside = False
            out_lines.append(line)
            continue
        if not inside:
            out_lines.append(line)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

# Get the min and max for x and y from the file
# Filename: path of the labeled .txt file
def min_max(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    rows = re.findall(r'^\s*\d+\s+(\d+)\s+(\d+)\s*$', text, flags=re.M)

    xs = [int(x) for x, y in rows]
    ys = [int(y) for x, y in rows]

    return max(xs), min(xs), max(ys), min(ys)

# Get the pair points from the filename
# Filename: path of the labeled .txt file
def get_pair(filename):
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    rows = re.findall(r'^\s*\d+\s+(\d+)\s+(\d+)\s*$', text, flags=re.M)
    xs = [int(x) for x, y in rows]
    ys = [int(y) for x, y in rows]

    # Normalize the pairs
    norm_xs = [x / img_width for x in xs]
    norm_ys = [y / img_height for y in ys]

    return norm_xs, norm_ys

# Run this to know the image shape in case needed 
def img_width_height(imgname):
    im = cv2.imread(imgname)
    h, w, c = im.shape
    return w, h

# Get the normalized midx-midy coord, width and height
# Filename: path of the labeled .txt file
def mid_width_height(filename):
    max_x, min_x, max_y, min_y = min_max(filename)
    mid_x = (max_x + min_x)/2
    mid_y = (max_y + min_y)/2
    width = max_x - min_x
    height = max_y - min_y
    
    norm_midx = mid_x / img_width
    norm_midy = mid_y / img_height

    norm_width = width / img_width
    norm_height = height / img_height
    return norm_midx, norm_midy, norm_width, norm_height

# Fill missing points in each file within the original label dir
# original_label_dir: directory of the original label .txt folder
# fixed_label_dir: directory of the fixed label .txt folder
def fill_missing(original_label_dir, fixed_label_dir):
    os.makedirs(fixed_label_dir, exist_ok=True)

    filenames = [f for f in os.listdir(original_label_dir) if os.path.isfile(os.path.join(original_label_dir, f))]

    for filename in tqdm(filenames, desc="Fixing labels", unit="file"):
            name, _ = os.path.splitext(filename)
            fixed_path = os.path.join(fixed_label_dir, f"{name}.txt")
            # fixed_21_label_path = os.path.join(fixed_21_label_dir, f"{name}.txt")

            helper_fill_missing(os.path.join(original_label_dir, filename), fixed_path)
            # remove_21(os.path.join(original_label_dir, filename), fixed_21_label_path) # This was used in the beginning when the landmarks have 25 points, but I removed point 21 since it's a faulty point

# Get the species label
def get_label(species_df, id):
    row = species_df[species_df["image_name"] == id]
    if row.empty:
        return None
    
    return row["Encode"].iloc[0] - 1

# Create the coco format from the .txt files
# label_dir - the folder of the fixed labels
# coco_dir - the folder of the coco format data
def COCO_convert(label_dir, coco_dir, species_label): 
    os.makedirs(coco_dir, exist_ok=True)
    species_df = pd.read_csv(species_label)

    filenames = [f for f in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, f))]

    for filename in tqdm(filenames, desc="Fixing labels", unit="file"):
            name, _ = os.path.splitext(filename)
            label_path = os.path.join(coco_dir, f"{name}.txt")

            midx, midy, width, height = mid_width_height(os.path.join(label_dir, filename))
            xs, ys = get_pair(os.path.join(label_dir, filename))
            label = get_label(species_df, name)
            if label != 3: # Only use images with species specification
                with open(label_path, "w") as f:
                    f.write(f"{label} {midx} {midy} {width} {height} ")
                    for i in range(len(xs)):
                        f.write(f"{xs[i]} {ys[i]} 2 ")

def decrement_first_number_in_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                continue

            parts = content.split()

            # decrement the first number
            parts[0] = str(int(parts[0]) - 1)

            new_content = " ".join(parts)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"Updated: {filename}")
