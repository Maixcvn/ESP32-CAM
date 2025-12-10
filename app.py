import os
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify

# Import thư viện TF/Keras
# Lưu ý: Nếu mô hình của bạn rất lớn, việc load TensorFlow có thể mất thời gian trên Render Free Tier.
from tensorflow.keras.models import load_model

app = Flask(__name__)

# --- 1. Cài đặt Mô hình (Đảm bảo đường dẫn đúng với cấu trúc GitHub) ---
# Trỏ đến thư mục chứa SavedModel (model/model.savedmodel)
MODEL_PATH = 'model/model.savedmodel' 
# Trỏ đến file labels (model/labels)
LABELS_PATH = 'model/labels' 

# Kích thước ảnh đầu vào của Teachable Machine. 
# Hầu hết các dự án TM Image đều dùng 224x224
IMG_HEIGHT, IMG_WIDTH = 224, 224 

# Load mô hình và nhãn khi server khởi động
try:
    # Load mô hình từ thư mục SavedModel
    model = load_model(MODEL_PATH) 
    print("Đã tải mô hình SavedModel thành công.")
    
    with open(LABELS_PATH, 'r') as f:
        # Đọc nhãn và chỉ lấy tên lớp (loại bỏ "0 ", "1 ", v.v.)
        class_names = [name.strip().split(' ', 1)[-1] for name in f.readlines()]
    print(f"Nhãn đã tải: {class_names}")

except Exception as e:
    print(f"LỖI QUAN TRỌNG: Không thể tải mô hình hoặc nhãn: {e}")
    model = None
    class_names = []
    
# --- 2. Tuyến đường Chính để Nhận Ảnh từ ESP32-CAM ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Server chưa tải mô hình thành công"}), 500

    # Kiểm tra xem request có file ảnh đính kèm không (dùng khóa 'image')
    if 'image' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh (key 'image') trong request"}), 400
    
    file = request.files['image']
    
    try:
        # 1. Xử lý ảnh
        # Đọc dữ liệu ảnh từ request và chuyển thành đối tượng PIL Image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        # Thay đổi kích thước ảnh theo yêu cầu của TM (224x224)
        image = image.resize((IMG_WIDTH, IMG_HEIGHT)) 
        
        # Chuyển đổi ảnh sang mảng numpy
        image_array = np.asarray(image, dtype=np.float32)
        # Chuẩn hóa (Normalization) từ 0-255 về 0.0-1.0
        normalized_image_array = (image_array / 255.0).reshape(1, IMG_HEIGHT, IMG_WIDTH, 3) 

        # 2. Chạy dự đoán
        predictions = model.predict(normalized_image_array)
        score = np.max(predictions[0])
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        
        # 3. Logic Quyết định và Tạo Lệnh cho Arduino
        command_to_arduino = "KHONG_LAM_GI" # Lệnh mặc định

        # Thay đổi logic dưới đây theo nhu cầu phân loại của bạn
        if predicted_class_name == "Vat_can_A" and score > 0.90:
            command_to_arduino = "BAT_DEN"
        elif predicted_class_name == "Vat_can_B" and score > 0.85:
            command_to_arduino = "TAT_DEN"
        
        # 4. Trả về kết quả JSON cho ESP32-CAM
        return jsonify({
            "status": "success",
            "class": predicted_class_name,
            "confidence": f"{score*100:.2f}%",
            "command": command_to_arduino # ESP32-CAM sẽ gửi lệnh này qua Serial tới Arduino
        })

    except Exception as e:
        print(f"Lỗi xử lý ảnh hoặc dự đoán: {e}")
        return jsonify({"error": str(e)}), 500

# --- 3. Khởi động Server (Cần cho Render) ---
if __name__ == '__main__':
    # Render sẽ tự động cung cấp cổng (PORT)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
