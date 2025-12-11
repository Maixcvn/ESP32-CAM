from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load labels
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip().split(' ',1)[1] for line in f.readlines()]

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224,224))  # Kích thước theo model Teachable Machine mặc định
    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if 'file' in request.files:
            img_bytes = request.files['file'].read()
        else:
            img_bytes = request.data

        input_data = preprocess_image(img_bytes)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_index = int(np.argmax(output_data))
        pred_label = labels[pred_index]

        return jsonify({"class": pred_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
