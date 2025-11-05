# src/create_and_convert_dummy.py
import tensorflow as tf
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

h5_path = models_dir / "pothole_model.h5"
tflite_path = models_dir / "pothole_model.tflite"


print("Creating dummy Keras model (MobileNetV2 head) ...")
model = tf.keras.applications.MobileNetV2(input_shape=(128,128,3), weights=None, include_top=True, classes=2)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print("Saving .h5 to:", h5_path)
model.save(str(h5_path))


print("Converting to TFLite ...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_bytes = converter.convert()
tflite_path.write_bytes(tflite_bytes)

print("Done. Sizes:")
print(" -", h5_path.resolve(), "->", h5_path.stat().st_size, "bytes")
print(" -", tflite_path.resolve(), "->", tflite_path.stat().st_size, "bytes")
