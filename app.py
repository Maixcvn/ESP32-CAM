import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# ==== 1. LOAD MÔ HÌNH TEACHABLE MACHINE SAVEDMODEL ====

MODEL_DIR = "model"          # thư mục chứa saved_model.pb
LABELS_PATH = "model/labels" # file nhãn

try:
    model = tf.saved_model.load(MODEL_DIR)
    infer = model.signatures["serving_default"]
    print("Loaded Teachable Machine model OK")
    
    # Đọc file nhãn
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]
    print("Labels:", class_names)

except Exception as e:
    print("ERROR loading model:", e)
    model = None
    infer = None
    class_names = []


# ==== 2. API NHẬN ẢNH TỪ ESP32 ====

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        # ESP32 gửi ảnh RAW (image/jpeg) trong body → dùng request.data
        img_bytes = request.data
        
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Resize đúng chuẩn Teachable Machine (luôn 224x224)
        img = img.resize((224, 224))
        
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        x_tensor = tf.convert_to_tensor(x)

        result = infer(x_tensor)

        output = list(result.values())[0].numpy()[0]

        idx = int(np.argmax(output))
        confidence = float(output[idx])
        label = class_names[idx]

        return jsonify({
            "class": label,
            "confidence": confidence,
            "index": idx
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==== 3. RUN SERVER (RENDER) ====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
