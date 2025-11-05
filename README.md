# Pothole Detection 

Files:
- models/pothole_model.tflite   (TensorFlow Lite model, float32 input)
- labels.txt                    (index 0 = normal, index 1 = pothole)
- src/tflite_test.py            (test script to verify model)
- samples/                      (optional sample images)
- README.md

Model I/O:
- Input: 1 x 128 x 128 x 3, float32 (resize image to 128x128, RGB), normalize by /255.0
- Output: 1 x 1 float32 scalar → probability of 'pothole' (sigmoid-like)
  - Interpretation: probability >= 0.5 => "pothole", else "normal"

How to test locally:
1. Activate virtualenv:
   .\venv\Scripts\Activate.ps1
2. Run:
   python src/tflite_test.py
   -> prints per-image probability, label, latency and creates tflite_results.csv

Android integration snippet (Kotlin, high-level):
- Load pothole_model.tflite with TensorFlow Lite Interpreter.
- Prepare input ByteBuffer of 1*128*128*3 floats (row-major RGB normalized to [0,1]).
- Run interpreter.run(input, output) where output is a FloatBuffer of length 1.
- Interpret: prob = output[0]; if prob>=0.5 => pothole.

 Model Information

| Property | Details |
|-----------|----------|
| **Architecture** | MobileNetV2 (transfer learning) |
| **Input Shape** | 128 × 128 × 3 (RGB) |
| **Output** | 1 neuron (Sigmoid → probability of pothole) |
| **Classes** | 0 → Normal, 1 → Pothole |
| **Framework** | TensorFlow / Keras |
| **Converted Format** | TensorFlow Lite (.tflite) |
| **Model Size** | ~8.8 MB |
| **Average Inference Time** | ~8–10 ms (CPU) |
| **Compatibility** | Android (TensorFlow Lite) / Python |

Model Performance (sample results)

| Metric | Training | Validation |
| Accuracy | 0.96 | 0.93 |
| Loss | 0.12 | 0.18 |

 Model trained for 5 epochs using MobileNetV2 base with frozen layers.
 
  File Overview

| File | Purpose |
| `src/train_real_model.py` | Train model on dataset |
| `src/convert_to_tflite.py` | Convert `.h5` → `.tflite` |
| `src/tflite_test.py` | Test TensorFlow Lite inference |
| `src/server.py` | Optional FastAPI inference endpoint |
| `models/pothole_model.h5` | Trained TensorFlow model |
| `models/pothole_model.tflite` | Converted mobile-ready model |
| `labels.txt` | Class labels |
| `README.md` | Documentation |




