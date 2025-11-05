import tensorflow as tf
from pathlib import Path

keras_path = Path("models/pothole_model.h5")
tflite_path = Path("models/pothole_model.tflite")

# make sure the h5 model exists
if not keras_path.exists():
    raise FileNotFoundError(f"Keras model not found at {keras_path.resolve()}")

# load the model
model = tf.keras.models.load_model(keras_path)
print("✅ Loaded model:", model)

# convert to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# save as binary
tflite_path.write_bytes(tflite_model)
print("✅ Saved:", tflite_path.resolve(), "(", tflite_path.stat().st_size, "bytes )")
