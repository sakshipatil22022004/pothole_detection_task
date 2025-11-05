# src/tflite_test.py
r"""
Verbose TFLite test runner.
Loads models/pothole_model.tflite and runs inference on images from:
C:\Users\Sakshi\OneDrive\Desktop\pothole_detection_task\dataset\Pothole_Image_Data
...
"""

import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import csv
import sys

# === Configuration ===
DATASET_DIR = Path(r"C:\Users\Sakshi\OneDrive\Desktop\pothole_detection_task\dataset\Pothole_Image_Data")

MODEL_PATH = Path("models/pothole_model.tflite")
OUT_DIR = Path("out/tflite_out")
LOG_CSV = Path("tflite_results.csv")
MAX_IMAGES = 50   # limit how many images we test at once

# === Helpers ===
def find_images():
    imgs = []
    if DATASET_DIR.exists():
        imgs = [p for p in DATASET_DIR.glob("**/*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    if not imgs:
        samples_dir = Path("samples")
        samples_dir.mkdir(exist_ok=True)
        imgs = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
        if not imgs:
            # create a sample so script always has at least one image
            p = samples_dir / "sample_test.jpg"
            im = Image.new("RGB", (640,480), (128,128,128))
            d = ImageDraw.Draw(im); d.text((10,10), "TEST IMAGE", fill=(255,255,255)); im.save(p)
            imgs = [p]
    return imgs[:MAX_IMAGES]

def load_and_preprocess(img_path, size, dtype):
    img = Image.open(img_path).convert("RGB").resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    if np.issubdtype(dtype, np.integer):
        arr = (arr * 255).astype(dtype)
    arr = np.expand_dims(arr, 0)
    return arr

# === Main ===
def main():
    print("\n=== TFLite Test Runner ===\n")
    print("Looking for model at:", MODEL_PATH.resolve())
    if not MODEL_PATH.exists():
        print("ERROR: TFLite model not found at", MODEL_PATH.resolve())
        print("Place your pothole_model.tflite at the path above (models/pothole_model.tflite)")
        sys.exit(1)

    imgs = find_images()
    print(f"Found {len(imgs)} image(s) to test. First up to {MAX_IMAGES} will be used.")
    for i,p in enumerate(imgs[:5]):
        print(f" sample[{i}]: {p}")

    # Load interpreter
    try:
        interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
        interpreter.allocate_tensors()
    except Exception as e:
        print("ERROR loading TFLite model:", e)
        sys.exit(1)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    h, w = input_details['shape'][1], input_details['shape'][2]
    dtype = input_details['dtype']

    print("\nModel input details:", input_details)
    print("Model output details:", output_details)
    print(f"Expecting input size: {w}x{h}, dtype: {dtype}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # open CSV log
    with LOG_CSV.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image", "probability", "label", "latency_ms"])

        for img_path in imgs:
            try:
                arr = load_and_preprocess(img_path, (w,h), dtype)
            except Exception as e:
                print(f"SKIP {img_path.name}: could not open/preprocess ({e})")
                continue

            interpreter.set_tensor(input_details['index'], arr)
            start = time.time()
            try:
                interpreter.invoke()
            except Exception as e:
                print(f"INVOKE ERROR for {img_path.name}: {e}")
                continue
            latency = (time.time() - start) * 1000.0
            out = interpreter.get_tensor(output_details['index'])
            prob = float(out.flatten()[0])
            label = "pothole" if prob >= 0.5 else "normal"

            # print one clear line per image
            print(f"{img_path.name:40s}  prob={prob:7.4f}  label={label:6s}  latency={latency:6.1f} ms")

            # write to csv
            writer.writerow([str(img_path), f"{prob:.6f}", label, f"{latency:.3f}"])

            # Save a copy of the input image into out with a small label overlay (for quick checking)
            try:
                im = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(im)
                draw.text((10,10), f"{label} {prob:.3f}", fill=(255,255,255))
                out_file = OUT_DIR / f"{img_path.stem}_result.jpg"
                im.resize((w*2, h*2)).save(out_file)
            except Exception:
                pass

    print("\nResults saved to:", LOG_CSV.resolve())
    print("Annotated copies (if created) are in:", OUT_DIR.resolve())
    print("\n=== Done ===\n")

if __name__ == "__main__":
    main()
