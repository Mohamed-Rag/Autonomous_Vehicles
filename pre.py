import os
import shutil
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from collections import Counter

# --- Configuration ---
# NOTE: The paths below are placeholders
# You MUST update these paths to your actual project structure before running the script.

# --- Phase 1: Merge/Copy Data ---
def merge_data(base_dirs, output_images, output_labels):
    """
    Copies images and corresponding JSON labels from multiple split directories
    (train, val, test) into a single merged directory, renaming them sequentially.
    """
    print("--- Phase 1: Merging Data ---")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    counter = 0
    for split_dir in base_dirs:
        images_dir = os.path.join(split_dir, "images")
        labels_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(images_dir):
            print(f"Warning: Image directory not found: {images_dir}")
            continue

        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue
            
            label_name = os.path.splitext(img_name)[0] + ".json"
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, label_name)

            if os.path.exists(label_path):  # Only move if label exists
                # Using a fixed extension for consistency, assuming all images are converted to JPG or are JPG
                new_img = os.path.join(output_images, f"{counter:06d}.jpg")
                new_lbl = os.path.join(output_labels, f"{counter:06d}.json")
                shutil.copy2(img_path, new_img)
                shutil.copy2(label_path, new_lbl)
                counter += 1

    print(f"✅ Done merging! Total files: {counter}")
    return output_images, output_labels

# --- Phase 2: Convert JSON labels to YOLO format (TXT) ---
def convert_json_to_yolo(input_json_dir, input_img_dir, output_yolo_dir):
    """
    Converts JSON bounding box labels to YOLO format (.txt files).
    It also dynamically creates a class map based on encountered categories.
    """
    print("\n--- Phase 2: Converting JSON to YOLO TXT ---")
    os.makedirs(output_yolo_dir, exist_ok=True)

    class_map = {}
    next_id = 0

    def get_class_id(category):
        nonlocal next_id
        if category not in class_map:
            class_map[category] = next_id
            next_id += 1
        return class_map[category]

    json_files = [f for f in os.listdir(input_json_dir) if f.endswith(".json")]
    
    for file in tqdm(json_files, desc="Converting labels"):
        json_path = os.path.join(input_json_dir, file)
        img_path = os.path.join(input_img_dir, file.replace(".json", ".jpg")) # Assuming images are .jpg

        if not os.path.exists(img_path):
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_path}")
            continue

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            print(f"Error opening image: {img_path}")
            continue

        objects = []
        # Assuming the structure is similar to BDD100K: data["frames"][0]["objects"]
        if "frames" in data and data["frames"]:
            for obj in data["frames"][0].get("objects", []):
                if "box2d" not in obj:
                    continue
                cat = obj["category"]
                box = obj["box2d"]

                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                # Normalize to YOLO format (center_x, center_y, width, height)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                cls_id = get_class_id(cat)
                objects.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if objects:
            out_path = os.path.join(output_yolo_dir, file.replace(".json", ".txt"))
            with open(out_path, "w") as f:
                f.write("\n".join(objects))

    print("✅ Conversion complete!")
    print(f"📋 Total classes: {len(class_map)}")
    print("\nClass mapping:")
    for k, v in sorted(class_map.items(), key=lambda item: item[1]):
        print(f"  {v}: {k}")
        
    return output_yolo_dir, class_map

# --- Phase 3: Data Distribution Analysis (Optional, can be removed if not needed) ---
def analyze_distribution(labels_dir):
    """Analyzes class distribution and bounding box sizes from YOLO labels."""
    print("\n--- Phase 3: Analyzing Data Distribution ---")
    class_counts = Counter()
    images_with_classes = []
    bbox_sizes = []

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    
    for label_file in tqdm(label_files, desc="Analyzing labels"):
        path = os.path.join(labels_dir, label_file)
        with open(path, "r") as f:
            lines = f.readlines()
        if not lines:
            continue
        
        classes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls, x, y, w, h = parts
                cls = int(cls)
                w, h = float(w), float(h)
                classes.append(cls)
                bbox_sizes.append(w * h)
                class_counts[cls] += 1
            except ValueError:
                continue # Skip malformed lines
        
        images_with_classes.append(set(classes))

    single_class_images = sum(1 for s in images_with_classes if len(s) == 1)

    print(f"Total images: {len(images_with_classes)}")
    print(f"Single-class images: {single_class_images}")
    print(f"\nClass distribution:")
    for c, count in class_counts.most_common():
        print(f"   Class {c}: {count}")

    if bbox_sizes:
        print(f"\n Avg bbox size (all classes): {np.mean(bbox_sizes):.5f}")
        print(f" Std bbox size: {np.std(bbox_sizes):.5f}")
    else:
        print("No bounding boxes found for size analysis.")

# --- Phase 4: Cleaning/Filtering Data ---
def clean_data(src_img_dir, src_lbl_dir, dst_img_dir, dst_lbl_dir, area_thresh=0.002, car_cls_id=2):
    """
    Removes bounding boxes smaller than area_thresh and removes images that
    only contain the 'car' class (ID 2).
    """
    print("\n--- Phase 4: Cleaning/Filtering Data ---")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    total_boxes = 0
    removed_boxes = 0
    images_kept = 0
    images_removed = 0

    label_files = [f for f in os.listdir(src_lbl_dir) if f.endswith(".txt")]

    for file in tqdm(label_files, desc="Cleaning labels"):
        label_path = os.path.join(src_lbl_dir, file)
        with open(label_path, "r") as f:
            lines = f.readlines()

        new_lines = []
        all_classes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            try:
                cls, xc, yc, w, h = map(float, parts)
                cls = int(cls)
            except ValueError:
                continue

            area = w * h
            total_boxes += 1
            
            if area >= area_thresh:
                new_lines.append(line)
                all_classes.append(cls)
            else:
                removed_boxes += 1

        # Check if the image should be kept
        # Keep if:
        # 1. There are remaining boxes (new_lines)
        # 2. AND not all remaining boxes are the 'car' class (ID 2)
        
        # Check for 'only car' condition on the *remaining* boxes
        has_non_car_class = any(cls != car_cls_id for cls in all_classes)
        
        if new_lines and (has_non_car_class or car_cls_id not in all_classes):
            # Save the cleaned label file
            new_label_path = os.path.join(dst_lbl_dir, file)
            with open(new_label_path, "w") as f:
                f.writelines(new_lines)

            # Copy the corresponding image
            img_name = os.path.splitext(file)[0] + ".jpg"
            src_img_path = os.path.join(src_img_dir, img_name)
            dst_img_path = os.path.join(dst_img_dir, img_name)

            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
                images_kept += 1
        else:
            images_removed += 1

    print("\n === CLEANING SUMMARY ===")
    print(f"Total boxes processed: {total_boxes}")
    if total_boxes > 0:
        print(f"Removed small boxes: {removed_boxes} ({(removed_boxes/total_boxes)*100:.2f}%)")
    print(f"Images kept: {images_kept}")
    print(f"Images removed: {images_removed}")
    print(f" Cleaned dataset saved in: {dst_img_dir} and {dst_lbl_dir}")
    
    return dst_img_dir, dst_lbl_dir

# --- Phase 5: Merging Cleaned Data with Support Data ---
def merge_cleaned_with_support(main_imgs, main_lbls, support_imgs, support_lbls, merged_imgs, merged_lbls, car_cls_id=2):
    """
    Merges the cleaned main dataset with a support dataset, excluding images
    from the main dataset that only contain the 'car' class.
    """
    print("\n--- Phase 5: Merging Cleaned Data with Support Data ---")
    os.makedirs(merged_imgs, exist_ok=True)
    os.makedirs(merged_lbls, exist_ok=True)

    def has_only_car(label_path):
        if not os.path.exists(label_path):
            return False
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        if not lines:
            return False
        try:
            classes = [int(line.split()[0]) for line in lines]
            return all(c == car_cls_id for c in classes)
        except ValueError:
            return False # Malformed line

    removed_count = 0
    kept_count = 0

    # 1. Copy/Filter Main Data
    print("Processing main data...")
    for lbl_file in tqdm(os.listdir(main_lbls), desc="Filtering main data"):
        if not lbl_file.endswith(".txt"):
            continue
        lbl_path = os.path.join(main_lbls, lbl_file)
        img_name = os.path.splitext(lbl_file)[0] + ".jpg"
        img_path = os.path.join(main_imgs, img_name)

        if has_only_car(lbl_path):
            removed_count += 1
            continue

        # Copy label and image
        shutil.copy(lbl_path, os.path.join(merged_lbls, lbl_file))
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(merged_imgs, img_name))
            kept_count += 1
        else:
            print(f"Warning: Image not found for label {lbl_file}")

    print(f"Copied images from main: {kept_count}")
    print(f"Excluded images (car only): {removed_count}")

    # 2. Copy Support Data
    print("\nProcessing support data...")
    support_count = 0
    for lbl_file in tqdm(os.listdir(support_lbls), desc="Copying support data"):
        if not lbl_file.endswith(".txt"):
            continue
        
        # Check for potential filename collision with main data (unlikely with sequential naming)
        # If collision is a concern, a renaming strategy should be implemented here.
        
        lbl_path = os.path.join(support_lbls, lbl_file)
        img_name = os.path.splitext(lbl_file)[0] + ".jpg"
        img_path = os.path.join(support_imgs, img_name)

        shutil.copy(lbl_path, os.path.join(merged_lbls, lbl_file))
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(merged_imgs, img_name))
            support_count += 1
        else:
            print(f"Warning: Image not found for support label {lbl_file}")

    print(f"Added support images: {support_count}")
    print(f"Final merged dataset saved in: {merged_imgs} and {merged_lbls}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Define all paths here ---
    # Phase 1 Paths
    BASE_DIRS = [
        r"D:\databackup\bdddddddddd\train",
        r"D:\databackup\bdddddddddd\val",
        r"D:\databackup\bdddddddddd\test"
    ]
    MERGED_IMAGES = "data/merged/images"
    MERGED_LABELS_JSON = "data/merged/labels_json"
    
    # Phase 2 Paths
    MERGED_LABELS_YOLO = "data/merged/labels_yolo"
    
    # Phase 4 Paths
    CLEANED_IMAGES = "data/cleaned/images"
    CLEANED_LABELS = "data/cleaned/labels"
    AREA_THRESHOLD = 0.002
    CAR_CLASS_ID = 2 # Based on the notebook's output: 2: car
    
    # Phase 5 Paths
    SUPPORT_IMAGES = r"D:\cleaned.v1-original_index.yolov8\support_cleaned\images"
    SUPPORT_LABELS = r"D:\cleaned.v1-original_index.yolov8\support_cleaned\labels"
    FINAL_IMAGES = "data/final/images"
    FINAL_LABELS = "data/final/labels"
    
    # 1. Merge Data (Optional, only if you need to re-run the initial merge)
    # merged_img_dir, merged_json_dir = merge_data(BASE_DIRS, MERGED_IMAGES, MERGED_LABELS_JSON)
    
    # Assuming Phase 1 is done and data is in MERGED_IMAGES and MERGED_LABELS_JSON
    merged_img_dir = MERGED_IMAGES
    merged_json_dir = MERGED_LABELS_JSON
    
    # 2. Convert JSON to YOLO
    yolo_dir, class_map = convert_json_to_yolo(merged_json_dir, merged_img_dir, MERGED_LABELS_YOLO)
    
    # 3. Analyze Distribution (Optional)
    analyze_distribution(yolo_dir)
    
    # 4. Clean Data
    cleaned_img_dir, cleaned_lbl_dir = clean_data(
        merged_img_dir, 
        yolo_dir, 
        CLEANED_IMAGES, 
        CLEANED_LABELS, 
        area_thresh=AREA_THRESHOLD, 
        car_cls_id=CAR_CLASS_ID
    )
    
    # 5. Analyze Distribution of Cleaned Data (Optional)
    analyze_distribution(cleaned_lbl_dir)
    
    # 6. Merge Cleaned Data with Support Data
    merge_cleaned_with_support(
        cleaned_img_dir, 
        cleaned_lbl_dir, 
        SUPPORT_IMAGES, 
        SUPPORT_LABELS, 
        FINAL_IMAGES, 
        FINAL_LABELS,
        car_cls_id=CAR_CLASS_ID
    )
    
    print("\n--- Preprocessing script finished. ---")
    print("Remember to update the hardcoded D:\\ paths in the configuration section!")
