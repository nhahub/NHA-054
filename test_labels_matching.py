import os
import cv2
import random
import numpy as np

# -------------------------------
# Configuration
# -------------------------------
IMAGE_DIR = r'G:\split\imgs'     # Folder with images
LABEL_DIR = r'G:\split\labels'     # Folder with .txt label files (YOLO format)
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Class names (update with your actual classes)
CLASS_NAMES =  [
    # cardboard
    'cardboard_bags', 'cardboard_boxes', 'cardboard_cups', 'cardboard_holders',

    # glass
    'glass_bottle_medium', 'glass_bottle_small', 'glass_cup', 'glass_jar', 'glass_jug',

    # metal
    'AL_foil', 'chair_iron', 'door_handle_iron', 'fan_iron', 'kettle_AL',
    'large_can_AL', 'large_cooking_pan_AL', 'large_fork_AL', 'large_knife_AL',
    'large_spoon_AL', 'medium_fork_AL', 'medium_spoon_AL', 'refrigerator_iron',
    'small_can_AL', 'small_cooking_pan_AL', 'small_fork_AL', 'small_knife_AL',
    'small_spoon_AL', 'stove_iron', 'table_iron', 'tap_iron', 'washer_machine_iron',

    # paper
    'Cartoon_paper', 'Tissue_Paper', 'newspapers', 'paper',

    # plastic
    'plastic_bags', 'plastic_bottles_large', 'plastic_bottles_medium', 'plastic_bottles_small',
    'plastic_boxes', 'plastic_chairs', 'plastic_cups', 'plastic_utensils',

    # textile
    'child_chemise', 'child_dress', 'child_jacket', 'child_pullover',
    'child_short', 'child_skirt', 'child_t_shirt', 'child_trouser',
    'men_chemise', 'men_jacket', 'men_pullover', 'men_short',
    'men_t_shirt', 'men_trouser', 'women_blouse', 'women_chemise',
    'women_gloves', 'women_jacket', 'women_pullover', 'women_scarves',
    'women_skirt', 'women_socks', 'women_summer_dress',
    'women_trousers', 'women_winter_dress',

    # wood
    'King&Queen bed', 'Tall chair', 'closet', 'dining table',
    'full bed', 'seat', 'side table'
]
# -------------------------------

def get_random_image_and_label(img_dir, label_dir):
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTENSIONS)]
    if not img_files:
        raise ValueError("No images found in the image directory.")
    
    random_img = random.choice(img_files)
    img_path = os.path.join(img_dir, random_img)
    
    # Corresponding label file (same name, .txt)
    label_path = os.path.join(label_dir, os.path.splitext(random_img)[0] + '.txt')
    
    if not os.path.exists(label_path):
        print(f"Warning: Label not found for {random_img}, skipping...")
        return get_random_image_and_label(img_dir, label_dir)  # Recurse until valid pair
    
    return img_path, label_path, random_img

def draw_yolo_boxes(image, label_path, class_names):
    img_height, img_width = image.shape[:2]
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        values = line.strip().split()
        if len(values) < 5:
            continue
        class_id = int(values[0])
        x_center = float(values[1]) * img_width
        y_center = float(values[2]) * img_height
        w = float(values[3]) * img_width
        h = float(values[4]) * img_height
        
        # Convert to corner coordinates
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put label
        label = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image

# -------------------------------
# Main Execution
# -------------------------------
def main():
    img_path, label_path, img_name = get_random_image_and_label(IMAGE_DIR, LABEL_DIR)
    
    print(f"Selected Image: {img_name}")
    print(f"Label File: {os.path.basename(label_path)}")
    
    # Load image
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    # Draw boxes
    image_with_boxes = draw_yolo_boxes(image.copy(), label_path, CLASS_NAMES)
    
    # Show image
    cv2.imshow('Random Image with Bounding Boxes', image_with_boxes)
    print("Press any key on the image window to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()