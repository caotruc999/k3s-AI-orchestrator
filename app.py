from flask import Flask, render_template, jsonify, request
import random

# Import chính xác từ thư mục modules/ theo đúng sơ đồ VS Code của ông
from modules.k3s_client import get_live_metrics, execute_k3s_scale
from modules.ai_agent import predict_future_load


app = Flask(__name__)

# Trạng thái điều phối lưu trữ trong bộ nhớ tạm (In-memory)
orchestrator_state = {
    "mode": "AUTO",          # AUTO (AI tự quyết định) hoặc MANUAL (Người bấm)
    "cpu_history": [60, 65, 70, 72, 75, 78, 80], # Lưu lịch sử tải để truyền cho AI
    "thresholds": {
        "cpu_warn": 75,
        "cpu_crit": 90,
        "ram_warn": 70
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    # 1. Gọi Người 1 để lấy chỉ số CPU/RAM thực tế từ K3s
    live_data = get_live_metrics()
    
    # Cập nhật lịch sử tải CPU
    orchestrator_state["cpu_history"].append(live_data["cpu"])
    if len(orchestrator_state["cpu_history"]) > 10:
        orchestrator_state["cpu_history"].pop(0)

    # 2. Gọi Người 2 để lấy dự báo tải từ mô hình AI
    ai_predictions = predict_future_load(orchestrator_state["cpu_history"])
    max_predicted_load = max(ai_predictions) if ai_predictions else 0

    # 3. Logic điều phối tự động của ông (Orchestrator Logic)
    current_pods = live_data["pods"]
    suggested_pods = current_pods

    if orchestrator_state["mode"] == "AUTO":
        # Nếu AI dự báo tải sắp vượt ngưỡng cảnh báo trong 5 phút tới -> Scale Up!
        if max_predicted_load > orchestrator_state["thresholds"]["cpu_warn"]:
            suggested_pods = min(10, current_pods + 1) # Giới hạn tối đa 10 pods
            print(f"[AI Orchestrator] Tải dự báo đạt {max_predicted_load}%. Scale Up lên {suggested_pods} pods.")
        # Nếu tải thực tế và tải dự báo đều thấp -> Scale Down để tiết kiệm tài nguyên
        elif live_data["cpu"] < 40 and max_predicted_load < 50:
            suggested_pods = max(1, current_pods - 1) # Giới hạn tối thiểu 1 pod
            print(f"[AI Orchestrator] Tải hệ thống thấp. Scale Down xuống {suggested_pods} pods.")

        # Nếu có sự thay đổi, ra lệnh cho Người 1 thực thi scale trên K3s thật
        if suggested_pods != current_pods:
            execute_k3s_scale(suggested_pods)
            live_data["pods"] = suggested_pods

    return jsonify({
        "mode": orchestrator_state["mode"],
        "current_cpu": live_data["cpu"],
        "current_ram": live_data["ram"],
        "current_replicas": live_data["pods"],
        "predictions": ai_predictions,
        "thresholds": orchestrator_state["thresholds"]
    })

@app.route('/api/control', methods=['POST'])
def control_system():
    """Nhận lệnh điều phối thủ công từ các nút bấm trên giao diện"""
    data = request.json
    live_data = get_live_metrics()
    
    if "mode" in data:
        orchestrator_state["mode"] = data["mode"]
        
    if "action" in data:
        current_pods = live_data["pods"]
        if data["action"] == "SCALE_UP":
            target_pods = min(10, current_pods + 2)
            execute_k3s_scale(target_pods)
        elif data["action"] == "SCALE_DOWN":
            target_pods = max(1, current_pods - 1)
            execute_k3s_scale(target_pods)
            
    return jsonify({"status": "success"})

@app.route('/api/thresholds', methods=['POST'])
def update_thresholds():
    data = request.json
    clean = {k: float(v) for k, v in data.items() if k in orchestrator_state["thresholds"]}
    orchestrator_state["thresholds"].update(clean)
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
