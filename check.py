import cv2
import os
import random
import yaml  # We need this to read the data.yaml file
import numpy as np


SPLIT_DATASET_PATH = r"C:\\Users\\HP ZBOOK\\Downloads\\recycle_project\\splitted"

def yolo_to_cv2(yolo_box, img_width, img_height):
    class_id, x_center, y_center, w, h = yolo_box
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    x_min = int(x_center - (w / 2))
    y_min = int(y_center - (h / 2))
    x_max = int(x_center + (w / 2))
    y_max = int(y_center + (h / 2))
    return x_min, y_min, x_max, y_max


try:
    # 1. Define paths based on the new structure
    # We will check the 'train' split by default. 
    # You can change 'train' to 'val' here to check the validation set.
    data_yaml_file = os.path.join(SPLIT_DATASET_PATH, "data.yaml")
    images_dir = os.path.join(SPLIT_DATASET_PATH, "train", "images")
    labels_dir = os.path.join(SPLIT_DATASET_PATH, "train", "labels")

    # 2. Load class names from data.yaml
    with open(data_yaml_file, 'r') as f:
        data_config = yaml.safe_load(f)
        class_names = data_config['names']
    print(f"Loaded {len(class_names)} classes from data.yaml.")

    # 3. Get list of all images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Error: No images found in {images_dir}")
        exit()
        
    print(f"Found {len(image_files)} images in 'train' split. Starting visual check...")
    print("Press any key (except 'q') to see the next random image.")
    print("Press 'q' to quit.")

    # 4. Loop and show random images
    while True:
        # Pick a random image
        img_name = random.choice(image_files)
        img_path = os.path.join(images_dir, img_name)
        
        # Find corresponding label
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_name)
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img_height, img_width, _ = img.shape
        
        # Check if label exists
        if not os.path.exists(label_path):
            print(f"Warning: No label file found for {img_name}")
            cv2.imshow("Verification (No Label)", img)
        else:
            # Read labels and draw boxes
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                        
                    # Parse YOLO line
                    class_id = int(parts[0])
                    yolo_box = [float(p) for p in parts]
                    
                    # Get class name
                    if class_id < len(class_names):
                        label_text = class_names[class_id]
                    else:
                        label_text = f"INVALID_ID_{class_id}"
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = yolo_to_cv2(yolo_box, img_width, img_height)
                    
                    # Draw
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(f"Verification: {img_name}", img)

        # Wait for user key
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if key == ord('q') or key == 27: # 'q' or ESC key
            print("Quitting.")
            break

except FileNotFoundError:
    print(f"Error: Could not find dataset paths. Is SPLIT_DATASET_PATH correct?")
    print(f"Path was: {SPLIT_DATASET_PATH}")
except Exception as e:
    print(f"An error occurred: {e}")