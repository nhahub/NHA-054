from ultralytics import YOLO
import numpy as np 
from ultralytics.utils.plotting import Annotator  # Import the new tool
import cv2  # Import OpenCV
import math

# --- Step 1: Define your "Weight Lookup Table" ---
# (This is your dictionary, exactly as you provided it)
AVERAGE_WEIGHTS_G = {
    # Cardboard
    'cardboard_bags': 50,
    'cardboard_boxes': 400,
    'cardboard_cups': 10,
    'cardboard_holders': 50,
    
    # Glass
    'glass_bottle_medium': 300,
    'glass_bottle_small': 200,
    'glass_cup': 250,
    'glass_jar': 350,
    'glass_jug': 1200,
    
    # Metal (HIGHLY VARIABLE)
    'AL_foil': 10,
    'chair_iron': 7000,
    'door_handle_iron': 350,
    'fan_iron': 5000,
    'kettle_AL': 800,
    'large_can_AL': 18,
    'large_cooking_pan_AL': 1500,
    'large_fork_AL': 80,
    'large_knife_AL': 80,
    'large_spoon_AL': 80,
    'medium_fork_AL': 50,
    'medium_spoon_AL': 50,
    'refrigerator_iron': 85000,
    'small_can_AL': 15,
    'small_cooking_pan_AL': 700,
    'small_fork_AL': 30,
    'small_knife_AL': 30,
    'small_spoon_AL': 30,
    'stove_iron': 70000,
    'table_iron': 15000,
    'tap_iron': 2000,
    'washer_machine_iron': 75000,
    
    # Paper
    'Cartoon_paper': 60,
    'Tissue_Paper': 1,
    'newspapers': 350,
    'paper': 5,
    
    # Plastic
    'plastic_bags': 8,
    'plastic_bottles_large': 25,
    'plastic_bottles_medium': 22,
    'plastic_bottles_small': 10,
    'plastic_boxes': 1000,
    'plastic_chairs': 3500,
    'plastic_cups': 12,
    'plastic_utensils': 3,
    
    # Textile (HIGHLY VARIABLE)
    'child_chemise': 100,
    'child_dress': 150,
    'child_jacket': 400,
    'child_pullover': 250,
    'child_short': 100,
    'child_skirt': 100,
    'child_t_shirt': 100,
    'child_trouser': 200,
    'men_chemise': 150,
    'men_jacket': 1100,
    'men_pullover': 350,
    'men_short': 250,
    'men_t_shirt': 170,
    'men_trouser': 400,
    'women_blouse': 130,
    'women_chemise': 130,
    'women_gloves': 50,
    'women_jacket': 900,
    'women_pullover': 300,
    'women_scarves': 100,
    'women_skirt': 300,
    'women_socks': 40,
    'women_summer_dress': 250,
    'women_trousers': 350,
    'women_winter_dress': 500,
    
    # Wood (EXTREMELY VARIABLE)
    'King&Queen bed': 100000,
    'Tall chair': 7000,
    'closet': 80000,
    'dining table': 60000,
    'full bed': 55000,
    'seat': 6000,
    'side table': 10000,
}

# --- Step 2: Load your ALREADY-TRAINED model ---
# *** USE 'r' FOR YOUR WINDOWS PATHS ***
model_path = r'G:\split\yolov8_custom\train\weights\best.pt'
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check the path to your 'best.pt' file.")
    exit()

# Get the class names from the model itself
class_names = model.names
print(f"Model loaded. Detecting {len(class_names)} classes.")

# --- Step 3: Run prediction (NO 'save=True' needed) ---
# *** USE 'r' FOR YOUR WINDOWS PATHS ***
source_image_path = r'all_data\glass\images\1ad6c730-images_96.jpeg'

# *** NEW: Load the image with OpenCV ***
try:
    img = cv2.imread(source_image_path)
    if img is None:
        raise Exception("Image not found or path is incorrect")
except Exception as e:
    print(f"Error loading image {source_image_path}: {e}")
    exit()

# Run prediction
try:
    results = model.predict(source_image_path)
except Exception as e:
    print(f"Error during prediction: {e}")
    exit()

# --- Step 4: Process results, calculate weight, and DRAW on the image ---
total_weight = 0
detected_items_report = {}

# *** NEW: Create an 'Annotator' object to draw on our image ***
annotator = Annotator(img, line_width=2, example=str(class_names))

if results:
    boxes = results[0].boxes
    
    if len(boxes) == 0:
        print("No objects detected in the image.")
    else:
        for box in boxes:
            # Get class ID and name
            cls_id = int(box.cls[0])
            class_name = class_names[cls_id]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Add to our text report
            detected_items_report[class_name] = detected_items_report.get(class_name, 0) + 1
            
            # Look up the weight
            weight_of_item = AVERAGE_WEIGHTS_G.get(class_name)
            
            # *** THIS IS THE NEW PART ***
            # Create the custom label
            if weight_of_item:
                total_weight += weight_of_item
                label = f"{class_name}: {weight_of_item}g"
            else:
                label = f"{class_name}: ??g" # Mark as unknown
                print(f"**Warning: No weight defined for '{class_name}'.**")

            # Draw the box and our custom label on the image
            annotator.box_label((x1, y1, x2, y2), label, color=(0, 200, 0)) # Green box

# --- Step 5: Save the new image and print the report ---

# *** NEW: Save the image we drew on ***
output_image_path = 'result_with_weights2.jpg'
cv2.imwrite(output_image_path, annotator.result())
print(f"\nSaved annotated image to: {output_image_path}")


print("\n--- 📦 Detection Report ---")
if not detected_items_report:
    print("No items found to report.")
else:
    for item, count in detected_items_report.items():
        print(f"Detected: {count} x {item}")

print("---------------------------------")
print(f"TOTAL ESTIMATED WEIGHT: {total_weight} grams")
print(f"TOTAL ESTIMATED WEIGHT: {total_weight / 1000:.2f} kg")
