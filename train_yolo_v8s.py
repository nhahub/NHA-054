import os
import torch
from ultralytics import YOLO


YAML_PATH = 'splitted/data.yaml' 
MODEL_SIZE = 'yolov8n.pt'        
EPOCHS = 50                      
PROJECT_NAME = 'yolov8_custom'
SAVE_PERIOD = 10


if torch.cuda.is_available():
    DEVICE = 0                    # GPU 0
    BATCH_SIZE = 16
    IMG_SIZE = 640
    print("GPU Detected")
else:
    DEVICE = 'cpu'
    BATCH_SIZE = 4                
    IMG_SIZE = 416                
    print("CPU Mode")

# Verify YAML
if not os.path.exists(YAML_PATH):
    raise FileNotFoundError(f"YAML missing: {os.path.abspath(YAML_PATH)}")

# ESSENTIALS
if __name__ == '__main__':
    print(f"\n🚀 Training: {MODEL_SIZE} on {DEVICE}")
    print(f"📁 Dataset: {YAML_PATH} | Epochs: {EPOCHS}")
    print(f"⚙️  Batch: {BATCH_SIZE} | ImgSz: {IMG_SIZE}")

    #  Load Model
    model = YOLO(MODEL_SIZE)

    # TRAIN
    results = model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name='train',
        save_period=SAVE_PERIOD,
        plots=True,
        save_json=True
    )

    #  Validate
    print("\n📊 Val Results:")
    model.val()

    # 4. Export
    model.export(format='onnx')
    print("\n DONE! Check: runs/detect/yolov8_custom/train/")
    print(" Best: best.pt | Last: last.pt")