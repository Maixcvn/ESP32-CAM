# app.py

import tensorflow.lite as tflite
import numpy as np
import json
import io
from PIL import Image
from flask import Flask, request, jsonify

# --- Cấu hình Mô hình ---
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"
MODEL_INPUT_CHANNELS = 3  # Giả định RGB, cần kiểm tra lại

app = Flask(__name__)

# [span_0](start_span)Tải nhãn từ file labels.txt[span_0](end_span)
# [span_1](start_span)labels.txt: "0 nhua", "1 giay", v.v...[span_1](end_span)
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    # Chỉ lấy phần nhãn tiếng Việt (ví dụ: 'nhua', 'giay')
    LABELS = [line.strip().split()[-1] for line in f.readlines()]

# Tải mô hình TFLite và khởi tạo Interpreter
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Lấy thông tin đầu vào và đầu ra
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Kích thước ảnh đầu vào (ví dụ: [1, 96, 96, 3])
    INPUT_SHAPE = input_details[0]['shape']
    IMG_HEIGHT = INPUT_SHAPE[1]
    IMG_WIDTH = INPUT_SHAPE[2]
    
    print(f"Server đã tải mô hình thành công. Đầu vào: {IMG_WIDTH}x{IMG_HEIGHT}")

except Exception as e:
    print(f"Lỗi tải mô hình TFLite: {e}")
    # Nếu tải mô hình thất bại, server không nên chạy
    interpreter = None

@app.route('/classify', methods=['POST'])
def classify_image():
    if interpreter is None:
        return jsonify({"error": "Model initialization failed"}), 500
        
    # Kiểm tra xem có file ảnh được gửi lên không
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' file in request"}), 400

    file = request.files['image']
    
    try:
        # 1. Đọc và Tiền xử lý Ảnh
        # Mở ảnh JPEG được gửi từ ESP32
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Đảm bảo ảnh đúng kích thước mô hình mong đợi (ví dụ: 96x96)
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        
        # Chuyển đổi sang mảng numpy (uint8 cho mô hình lượng tử hóa)
        input_data = np.array(img, dtype=np.uint8)
        
        # Thêm chiều batch (1) vào trước: [H, W, C] -> [1, H, W, C]
        input_data = np.expand_dims(input_data, axis=0)
        
        # 2. Thực thi Mô hình (Inference)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 3. Lấy Kết quả và Phân tích
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Tìm chỉ số lớp có xác suất cao nhất
        class_index = np.argmax(output_data)
        result_class = LABELS[class_index]
        
        # Lấy xác suất. Vì là mô hình 8-bit, giá trị tối đa thường là 255.
        # Chúng ta chia cho 255 để chuẩn hóa về (0, 1)
        # Nếu output là float (dynamic range quantization), bước chia này không cần thiết
        confidence_score = output_data[0][class_index] / 255.0

        # 4. Trả về Kết quả JSON
        return jsonify({
            "status": "success",
            "class": result_class,
            "confidence": float(confidence_score),
            "label_index": int(class_index)
        })

    except Exception as e:
        # Ghi lại lỗi và trả về lỗi cho ESP32
        print(f"Processing error: {e}")
        return jsonify({"error": "Image processing or inference failed", "details": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Trash Classifier Server is running. Send POST request to /classify with 'image' file."

if __name__ == '__main__':
    # Khi triển khai lên Render, bạn nên dùng Gunicorn hoặc uWSGI.
    # Tuy nhiên, khi phát triển cục bộ, bạn có thể chạy:
    # app.run(host='0.0.0.0', port=5000)
    
    # Đối với Render, bạn sẽ sử dụng Gunicorn (hoặc tương đương)
    # Lệnh Gunicorn (ví dụ): gunicorn app:app -w 4
    # Nếu chạy thủ công cho Render test:
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
