from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import numpy as np 
from contextlib import asynccontextmanager

# Change: Import your module with a clear alias (e.g., model_logic) 
# and ensure the file is named model.py
import model as model_logic

# Define the model path here, as it's specific to your environment
MODEL_PATH = "best.pt"

# Change: Global variable name changed to avoid conflict
loaded_model = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global loaded_model # Change: Reference the new global variable
    print("Starting up: Loading YOLO model...")
    try:
        # Change: Call load_model on the imported module alias
        loaded_model = model_logic.load_model(MODEL_PATH)
        yield 
    except Exception as e:
        print(f"FATAL: Model loading failed: {e}")
        raise

    print("Application shutting down.")

# Initialize the FastAPI app using the lifespan context manager
app = FastAPI(title="Recycling Sorter API", lifespan=lifespan)


@app.get("/")
def home():
    """A simple endpoint to check if the API is online."""
    # Change: Reference the new global variable
    return {"status": "OK", "message": "Recycling Model API is running!", "model_loaded": loaded_model is not None}

@app.post("/detect/")
async def detect_materials(image: UploadFile = File(...)):
    """
    Main endpoint to detect, classify, and weigh materials from an image file.
    """
    # Change: CRITICAL: Check if the model is loaded before proceeding
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check startup logs for errors.")
        
    try:
        image_bytes = await image.read()
        
        # Change: Call run_inference on the imported module alias 
        # and pass the loaded model object as the first argument
        results = model_logic.run_inference(loaded_model, image_bytes)
        
        return {
            "total_weight_g": results["total_weight_g"],
            "total_weight_kg": results["total_weight_g"] / 1000,
            "detections": results["detections"],
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Image processing error: {e}")
    except Exception as e:
        print(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during detection.")

# A way to run the app from the command line (for testing)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)