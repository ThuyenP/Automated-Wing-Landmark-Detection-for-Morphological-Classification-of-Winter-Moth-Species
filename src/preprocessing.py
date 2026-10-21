import os
import cv2 as cv
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Remove the file_name "-bleach"
def sync_name(folder_path):
    count = 0
    
    for root, dirs, files in os.walk(folder_path):
        for filename in tqdm(files, desc="Reading image files"):
            old_path = os.path.join(root, filename)
            
            # Skip if path doesn't exist (e.g., broken links)
            if not os.path.exists(old_path):
                continue

            # Remove spaces
            new_filename = filename.replace("-unbleached", "")

            new_path = os.path.join(root, new_filename)

            # Only rename if the new name is different
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            count += 1

    print("Total number of files:", count)
    print("Done renaming all files.")

# Clahe transform helper
def clahe_enhance(path):
    img = cv.imread(path)

    # convert to LAB color space (best for contrast operations)
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

    l, a, b = cv.split(lab)

    # apply CLAHE only on L channel
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    # merge back
    enhanced_lab = cv.merge((cl, a, b))

    # convert back to BGR for saving
    enhanced_bgr = cv.cvtColor(enhanced_lab, cv.COLOR_LAB2BGR)

    return enhanced_bgr

# Clahe transform 
def clahe_transform(raw_dir, transformed_dir):

    for filename in tqdm(os.listdir(raw_dir), desc= 'Applying CLAHE'):
        if filename.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(raw_dir, filename)
            
            if cv.imread(img_path) is None:
                print('Skip unreadble file:', filename)
                continue

            output = clahe_enhance(img_path)

            cv.imwrite(os.path.join(transformed_dir, filename), output)
            print('Saved:', img_path)

import os
from tqdm import tqdm


def collect_stems(folder_path, ext):
    """Walk folder_path and return {filename_without_extension: full_path} for files with the given extension."""
    stems = {}

    for root, dirs, files in os.walk(folder_path):
        for filename in tqdm(files, desc=f"Reading {ext} files in {os.path.basename(folder_path)}"):
            if filename.lower().endswith(ext):
                stem = filename[: -len(ext)]
                stems[stem] = os.path.join(root, filename)

    return stems


def find_matching_names(labels_folder, images_folder, label_ext=".txt", image_ext=".png"):
    """Return only the files that have the same name in both folders."""
    label_stems = collect_stems(labels_folder, label_ext)
    image_stems = collect_stems(images_folder, image_ext)

    label_names = set(label_stems.keys())
    image_names = set(image_stems.keys())

    matched = sorted(label_names & image_names)
    unmatched_images = sorted(image_names - label_names)

    print(f"\nMatched files: {len(matched)}")
    for stem in matched:
        print(f"  {stem}  ->  {label_stems[stem]}  <-->  {image_stems[stem]}")

    print(f"\nUnmatched {image_ext} files in {images_folder} (no corresponding {label_ext}): {len(unmatched_images)}")
    for stem in unmatched_images:
        print(f"  {image_stems[stem]}")

    return matched, unmatched_images


def delete_unmatched_images(image_stems, unmatched_stems):
    """Delete image files whose stem is in unmatched_stems."""
    deleted = 0
    for stem in unmatched_stems:
        path = image_stems[stem]
        os.remove(path)
        print(f"Deleted: {path}")
        deleted += 1
    print(f"\nDeleted {deleted} unmatched image files.")


if __name__ == "__main__":
    RAW_LABELS_DIR = "General Data/coco_labels"
    IMAGES_DIR = "General Data/unbleached_img"

    label_stems = collect_stems(RAW_LABELS_DIR, ".txt")
    image_stems = collect_stems(IMAGES_DIR, ".png")

    label_names = set(label_stems.keys())
    image_names = set(image_stems.keys())

    matched = sorted(label_names & image_names)
    unmatched_images = sorted(image_names - label_names)

    print(f"\nMatched files: {len(matched)}")
    for stem in matched:
        print(f"  {stem}  ->  {label_stems[stem]}  <-->  {image_stems[stem]}")

    print(f"\nUnmatched .png files in {IMAGES_DIR} (no corresponding .txt): {len(unmatched_images)}")
    for stem in unmatched_images:
        print(f"  {image_stems[stem]}")

    # Confirm before deleting
    if unmatched_images:
        confirm = input(f"\nDelete these {len(unmatched_images)} unmatched files? (yes/no): ").strip().lower()
        if confirm == "yes":
            delete_unmatched_images(image_stems, unmatched_images)
        else:
            print("Skipped deletion.")