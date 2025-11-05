# src/test_inference.py
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import sys

# <- change dataset path here if needed
DATASET_DIR = Path(r"C:\Users\Sakshi\OneDrive\Desktop\pothole_detection_task\dataset\Pothole_Image_Data")
SAMPLES_DIR = Path("samples")
OUT_DIR = Path("out/annotated")

def ensure_samples_from_dataset():
    """Return a list of image paths to use for testing.
       Prefer dataset images; if dataset empty, use/create samples/."""
    imgs = []
    if DATASET_DIR.exists():
        imgs = [p for p in DATASET_DIR.glob("**/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if imgs:
        return imgs[:10]  # limit to first 10 for speed
    # fallback: ensure samples folder has at least one image
    SAMPLES_DIR.mkdir(exist_ok=True)
    imgs = [p for p in SAMPLES_DIR.glob("*.jpg")] + [p for p in SAMPLES_DIR.glob("*.png")]
    if imgs:
        return imgs
    # create synthetic sample
    p = SAMPLES_DIR / "sample_test.jpg"
    img = Image.new("RGB", (640,480), (128,128,128))
    d = ImageDraw.Draw(img)
    d.rectangle([150,120,490,360], outline=(255,0,0), width=6)
    d.text((160,130), "TEST", fill=(255,255,255))
    img.save(p)
    return [p]

def run_yolo_smoke(samples):
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Ultralytics not installed or failed to import:", e)
        return
    model = YOLO("yolov8n.pt")  # will download if missing
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples):
        print("Running YOLO on", s)
        try:
            res = model.predict(source=str(s), save=False, verbose=False)
            ann = res[0].plot()
            # convert BGR->RGB and save
            Image.fromarray(ann[:, :, ::-1]).save(OUT_DIR / f"annotated_{i}.jpg")
            print("Saved", OUT_DIR / f"annotated_{i}.jpg")
        except Exception as e:
            print("Predict/plot failed, trying fallback save:", e)
            try:
                model.predict(source=str(s), save=True, project=str(OUT_DIR.parent), name="annotated", exist_ok=True)
                print("Saved via fallback into out/")
            except Exception as e2:
                print("Fallback also failed:", e2)

def main():
    samples = ensure_samples_from_dataset()
    print(f"Using {len(samples)} sample(s) from: {DATASET_DIR if samples and samples[0].is_relative_to(SAMPLES_DIR) else DATASET_DIR}")
    run_yolo_smoke(samples)
    print("Done. Check out/annotated/ for results.")

if __name__ == "__main__":
    main()
