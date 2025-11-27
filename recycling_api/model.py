from ultralytics import YOLO
import numpy as np
from ultralytics.utils.plotting import Annotator
import cv2
import math
import io # Needed to handle the image bytes from FastAPI

# --- Step 1: Define your "Weight Lookup Table" ---
# (This remains a global variable, accessible by the inference function)
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



def load_model(model_path: str) -> YOLO:
    """
    Loads the YOLO model from the specified path.
    """
    try:
        # *** USE 'r' FOR YOUR WINDOWS PATHS ***
        # The path should be passed from the main FastAPI file
        model = YOLO(model_path)
        print(f"Model loaded successfully. Detecting {len(model.names)} classes.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the path to your 'best.pt' file.")
        raise # Re-raise the exception to stop FastAPI startup


def run_inference(model: YOLO, image_bytes: bytes) -> dict:
    """
    Runs inference on the image bytes, calculates weight, and returns results.

    Args:
        model: The loaded YOLO model object.
        image_bytes: The raw bytes of the image file received from FastAPI.
        
    Returns:
        A dictionary containing detections, total weight, and image URL (or bytes).
    """
    # 1. Convert image bytes to an OpenCV image format (numpy array)
    image_stream = io.BytesIO(image_bytes)
    image_np = cv2.imdecode(
        np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR
    )

    if image_np is None:
        raise ValueError("Could not decode image bytes.")
    
    # Run prediction directly on the numpy array (OpenCV image)
    # The 'stream=True' is often useful in FastAPI to prevent blocking
    results = model.predict(image_np, stream=True)

    total_weight = 0
    detections = []
    
    # NEW: Create an 'Annotator' object to draw on our image
    annotator = Annotator(image_np.copy(), line_width=2, example=str(model.names))

    # 2. Process results
    if results:
        # The stream=True returns a generator, so we iterate to get the result
        result = next(results) 
        boxes = result.boxes
        
        for box in boxes:
            # Get class ID and name
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            
            # Get bounding box coordinates (normalized for output, absolute for drawing)
            x1_abs, y1_abs, x2_abs, y2_abs = [int(x) for x in box.xyxy[0]]
            
            # Look up the weight
            weight_of_item = AVERAGE_WEIGHTS_G.get(class_name)
            
            # Structure the detection result
            detection_data = {
                "box_xyxy": [x1_abs, y1_abs, x2_abs, y2_abs],
                "material": class_name,
                "weight_g": weight_of_item,
            }
            detections.append(detection_data)

            # Update total weight and draw label
            if weight_of_item:
                total_weight += weight_of_item
                label = f"{class_name}: {weight_of_item}g"
            else:
                label = f"{class_name}: ??g"
                
            # Draw the box and our custom label on the image
            annotator.box_label((x1_abs, y1_abs, x2_abs, y2_abs), label, color=(0, 200, 0)) # Green box

    # 3. Prepare the annotated image for response (e.g., as base64 or saved file path)
    # For simplicity, we'll skip returning the image bytes, but the logic is ready.
    annotated_img = annotator.result()
    # cv2.imwrite('annotated_result.jpg', annotated_img) # Save for debugging
    
    # 4. Return the structured results
    return {
        "total_weight_g": total_weight,
        "detections": detections,
    }

# We can remove the `if __name__ == "__main__":` block from this file
# since it's now meant to be a module imported by FastAPI.