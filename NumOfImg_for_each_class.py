import os
import matplotlib.pyplot as plt
from collections import Counter

# -------------------------------
# Configuration
# -------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(SCRIPT_DIR, 'images')
LABEL_DIR = os.path.join(SCRIPT_DIR, 'labels')
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


CLASS_NAMES = ['cardboard_bags', 'cardboard_boxes', 'cardboard_cups', 'cardboard_holders']

# -------------------------------
# Count class occurrences per bounding box
# -------------------------------
def count_classes_in_labels(label_dir, img_dir):
    class_counter = Counter()
    total_boxes = 0
    missing_images = 0
    valid_label_files = 0

    print("Scanning label files...\n")
    
    for label_file in os.listdir(label_dir):
        if not label_file.lower().endswith('.txt'):
            continue
        
        label_path = os.path.join(label_dir, label_file)
        img_name = os.path.splitext(label_file)[0]
        
        # Check if corresponding image exists
        img_found = False
        for ext in IMG_EXTENSIONS:
            if os.path.exists(os.path.join(img_dir, img_name + ext)):
                img_found = True
                break
        
        if not img_found:
            missing_images += 1
            continue  # Skip if no image

        valid_label_files += 1

        with open(label_path, 'r') as f:
            for line in f:
                values = line.strip().split()
                if len(values) >= 5:
                    try:
                        class_id = int(values[0])
                        class_counter[class_id] += 1
                        total_boxes += 1
                    except (ValueError, IndexError):
                        continue  # Skip malformed lines

    return class_counter, total_boxes, valid_label_files, missing_images

# Main
def main():
    # Validate directories
    if not os.path.exists(LABEL_DIR):
        print(f"Error: Label directory not found: {LABEL_DIR}")
        return
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Image directory not found: {IMAGE_DIR}")
        return

    # Count classes
    class_counter, total_boxes, valid_images, missing_images = count_classes_in_labels(LABEL_DIR, IMAGE_DIR)

    # Summary
    print("="*60)
    print("CLASS DISTRIBUTION SUMMARY")
    print("="*60)
    print(f"{'Class ID':<8} {'Class Name':<20} {'Count':<8} {'Percentage'}")
    print("-"*60)

    print(f"Total images with labels: {valid_images}")
    if missing_images > 0:
        print(f"Labels without images: {missing_images}")
    print(f"Total bounding boxes: {total_boxes}\n")

    # Print per-class stats
    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}"
        percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
        print(f"{class_id:<8} {name:<20} {count:<8} {percentage:.2f}%")

    print("-"*60)

    # Plot bar chart
    if class_counter:
        plt.figure(figsize=(10, 6))
        class_ids = sorted(class_counter.keys())
        labels = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}" for i in class_ids]
        counts = [class_counter[i] for i in class_ids]

        colors = plt.cm.Set3(range(len(counts)))  # Better color handling
        bars = plt.bar(labels, counts, color=colors)
        plt.title('Number of Objects per Class', fontsize=16, fontweight='bold')
        plt.xlabel('Class')
        plt.ylabel('Number of Bounding Boxes')
        plt.xticks(rotation=15, ha='right')

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.show()
    else:
        print("No valid bounding boxes found in labels.")


if __name__ == '__main__':
    main()