# src/server.py
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io
from pathlib import Path

app = FastAPI()


MODEL_WEIGHTS = Path("runs/detect/train/weights/best.pt")

MODEL_LOAD = str(MODEL_WEIGHTS) if MODEL_WEIGHTS.exists() else "yolov8n.pt"

print("Loading model:", MODEL_LOAD)
model = YOLO(MODEL_LOAD)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # save temporary if needed
    tmp = Path("tmp.jpg")
    img.save(tmp)
    results = model.predict(str(tmp), save=False)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": conf, "class_id": cls})
    return {"detections": detections}
