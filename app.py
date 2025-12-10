import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- ĐƯỜNG DẪN MÔ HÌNH ĐÃ SỬA ---
MODEL_DIR = "model/model.savedmodel"      # chứa saved_model.pb
LABELS_PATH = "model/labels"              # file nhãn phải nằm trong /model/

# Kích thước ảnh TM (thường là 224x224)
IMG_HEIGHT, IMG_WIDTH = 224, 224

# --- LOAD MODEL ---
try:
    model = load_model(MODEL_DIR)
    print(">>> ĐÃ TẢI MÔ HÌNH THÀNH CÔNG !")

    # load labels
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]
    print(">>> Labels:", class_names)

except Exception as e:
    print("LỖI KHI TẢI MÔ HÌNH:", e)
    model = None
    class_names = []


# --- API DỰ ĐOÁN ---
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "Image file missing"}), 400

    try:
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        img_array = np.asarray(image, dtype=np.float32)
        img_array = (img_array / 255.0).reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)

        predictions = model.predict(img_array)
        score = float(np.max(predictions[0]))
        class_id = int(np.argmax(predictions[0]))
        class_name = class_names[class_id]

        return jsonify({
            "status": "success",
            "class": class_name,
            "confidence": f"{score*100:.2f}%"
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500


# --- SERVER CHO RENDER ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
