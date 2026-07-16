from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

# Khởi tạo trạng thái hệ thống in-memory (Bộ nhớ tạm thời)
system_state = {
    "mode": "AUTO",              # AUTO (AI vận hành) hoặc MANUAL (Thủ công)
    "current_replicas": 24,       # Khớp với số mặc định trên UI của ông
    "current_cpu": 78,
    "current_ram": 61,
    "thresholds": {
        "cpu_warn": 75,
        "cpu_crit": 90,
        "ram_warn": 70
    }
}

@app.route('/')
def index():
    # Trả về giao diện Dashboard
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Endpoint cập nhật trạng thái hệ thống mỗi 3 giây"""
    # Giả lập biến đổi tải nhẹ để đồ thị uốn lượn mượt mà
    if system_state["mode"] == "AUTO":
        # Tải giả lập dao động trong vùng cao để kích hoạt AI cảnh báo
        system_state["current_cpu"] = random.randint(70, 88)
        system_state["current_ram"] = random.randint(58, 65)
        
        # Logic tự động scale giả lập của AI
        if system_state["current_cpu"] > 82:
            system_state["current_replicas"] = min(40, system_state["current_replicas"] + 1)
        elif system_state["current_cpu"] < 72:
            system_state["current_replicas"] = max(10, system_state["current_replicas"] - 1)
    else:
        # Nếu ở chế độ thủ công (MANUAL), tải biến thiên tự do, số pod giữ nguyên do user bấm
        system_state["current_cpu"] = random.randint(40, 95)
        system_state["current_ram"] = random.randint(50, 75)

    return jsonify(system_state)

@app.route('/api/control', methods=['POST'])
def control_system():
    """Endpoint tiếp nhận lệnh điều phối từ nút bấm trên Dashboard"""
    data = request.json
    
    if "mode" in data:
        system_state["mode"] = data["mode"]
        
    if "action" in data:
        if data["action"] == "SCALE_UP":
            system_state["current_replicas"] += 2
        elif data["action"] == "SCALE_DOWN":
            system_state["current_replicas"] = max(2, system_state["current_replicas"] - 1)
            
    return jsonify({"status": "success", "state": system_state})

@app.route('/api/thresholds', methods=['POST'])
def update_thresholds():
    """Endpoint lưu cấu hình ngưỡng cảnh báo mới"""
    data = request.json
    system_state["thresholds"].update(data)
    print(f"[*] Đã cập nhật ngưỡng hệ thống mới: {system_state['thresholds']}")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Chạy Web Server tại port 5000 công khai
    app.run(host='0.0.0.0', port=5000, debug=True)