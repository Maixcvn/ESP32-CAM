import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Import Keras 3 TFSMLayer để load SavedModel
from keras.layers import TFSMLayer

app = Flask(__name__)

# ---- CẤU HÌNH ĐƯỜNG DẪN MODEL ----
MODEL_DIR = "model/model.savedmodel"     # thư mục chứa saved_model.pb
LABELS_PATH = "model/labels"             # file nhãn

IMG_HEIGHT, IMG_WIDTH = 224, 224         # kích thước ảnh Teachable Machine

# ---- LOAD MODEL & LABELS ----
try:
    # Load SavedModel giống TensorFlow 2.x nhưng qua TFSMLayer
    model = TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
    print("✔ Đã load mô hình SavedModel bằng TFSMLayer")

    # Load nhãn
    with open(LABELS_PATH, "r") as f:
        class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

    print("✔ Nhãn:", class_names)

except Exception as e:
    print("❌ LỖI KHI TẢI MÔ HÌNH:", e)
    model = None
    class_names = []


# ---- API NHẬN ẢNH TỪ ESP32-CAM ----
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    # ESP32 gửi ảnh qua key "image"
    if "image" not in request.files:
        return jsonify({"error": "Thiếu file ảnh (image)"}), 400

    file = request.files["image"]

    try:
        # ---- 1. XỬ LÝ ẢNH ----
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        img_array = np.asarray(image, dtype=np.float32) / 255.0
        img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)

        # ---- 2. DỰ ĐOÁN ----
        output = model(img_array)             # Gọi SavedModel
        predictions = output.numpy()[0]       # Lấy kết quả về numpy

        class_id = int(np.argmax(predictions))
        score = float(np.max(predictions))
        predicted_class = class_names[class_id]

        # ---- 3. LOGIC RA LỆNH CHO ARDUINO ----
        command = "KHONG_LAM_GI"

        if predicted_class == "Vat_can_A" and score > 0.90:
            command = "BAT_DEN"
        elif predicted_class == "Vat_can_B" and score > 0.85:
            command = "TAT_DEN"

        # ---- 4. TRẢ JSON VỀ ESP32 ----
        return jsonify({
            "status": "success",
            "class": predicted_class,
            "confidence": f"{score*100:.2f}%",
            "command": command
        })

    except Exception as e:
        print("❌ Lỗi xử lý ảnh:", e)
        return jsonify({"error": str(e)}), 500


# ---- CHẠY SERVER TRÊN RENDER ----
@app.route("/", methods=["GET"])
def home():
    return "ESP32-CAM AI Server is running!", 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
