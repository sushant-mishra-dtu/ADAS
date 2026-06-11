import numpy as np
import tensorflow as tf
# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="android_app/app/src/main/assets/yolov8n.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# Run inference with dummy input
dummy_input = np.random.uniform(0, 1, input_details[0]['shape']).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# Print first few bounding box values (first 4 rows, first 10 anchors)
print("\nFirst 4 coordinates for first 10 anchors:")
print("Row 0 (cx):", output_data[0, :10])
print("Row 1 (cy):", output_data[1, :10])
print("Row 2 (w) :", output_data[2, :10])
print("Row 3 (h) :", output_data[3, :10])
