import os
import sys
import time
import psutil
from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULES_DIR)

if MODULES_DIR not in sys.path:
    sys.path.append(MODULES_DIR)

EMAIL_DIR = os.path.join(BASE_DIR, "notification", "email")
if EMAIL_DIR not in sys.path:
    sys.path.append(EMAIL_DIR)

from decision_policy import DecisionPolicy
from ai_agent import AIAgent, AgentConfig
from k3s_client import K3sClientMock, K3sResourceState
from send_mail import send_alert_email, build_alert_message

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# ==========================================
# 1. KHỞI TẠO ONNX & K3S CLIENT & AI AGENT
# ==========================================
ONNX_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.onnx")

FEATURE_NAMES = [
    "cpu_usage",
    "memory_usage",
    "cpu_request",
    "cpu_limit",
    "memory_request",
    "memory_limit",
    "network_bandwidth_usage",
    "network_latency",
    "disk_io",
    "pod_lifetime_seconds",
]

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"Không tìm thấy mô hình ONNX tại: {ONNX_PATH}")

session = ort.InferenceSession(ONNX_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

policy = DecisionPolicy()

# Khởi tạo K3s Client Mock
k3s_client = K3sClientMock(
    K3sResourceState(
        deployment_name="edge-ai-app",
        namespace="default",
        current_replicas=1,
        min_replicas=1,
        max_replicas=5
    )
)

# Khởi tạo AI Agent
agent = AIAgent(
    AgentConfig(
        min_replicas=1,
        max_replicas=5,
        scale_up_step=1,
        scale_down_step=1,
        cooldown_seconds=30,
        mode="AUTO"
    )
)

# ==========================================
# 2. HÀM BỔ TRỢ ĐỒNG BỘ TRẠNG THÁI (STATE SYNC)
# ==========================================
def sync_agent_with_k3s():
    """Cập nhật current_replicas trong agent theo đúng thực tế từ K3s Client"""
    cluster_status = k3s_client.get_status()
    agent.current_replicas = cluster_status["current_replicas"]

# ==========================================
# 3. API ENDPOINTS
# ==========================================

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    cluster_status = k3s_client.get_status()
    return jsonify({
        "mode": agent.config.mode,
        "deployment_name": cluster_status["deployment_name"],
        "namespace": cluster_status["namespace"],
        "current_replicas": cluster_status["current_replicas"],
        "min_replicas": cluster_status["min_replicas"],
        "max_replicas": cluster_status["max_replicas"],
        "cooldown_seconds": agent.config.cooldown_seconds
    })


@app.route("/mode", methods=["POST"])
def change_mode():
    try:
        data = request.get_json() or {}
        new_mode = data.get("mode", "").upper()

        agent.set_mode(new_mode)

        return jsonify({
            "message": "Chuyển mode thành công.",
            "mode": agent.config.mode
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/manual-scale", methods=["POST"])
def manual_scale():
    try:
        data = request.get_json() or {}
        action = data.get("action", "").strip()

        # Đảm bảo Agent đồng bộ dữ liệu số replica hiện tại với K3s trước khi ra quyết định
        sync_agent_with_k3s()

        agent_result = agent.manual_scale(action)

        # Map kết quả từ Agent sang K3s Client Execution
        if agent_result["action_taken"] == "Manual Scaled Up":
            execution_result = k3s_client.scale_up()
        elif agent_result["action_taken"] == "Manual Scaled Down":
            execution_result = k3s_client.scale_down()
        else:
            execution_result = k3s_client.keep()

        # Cập nhật ngược lại cho agent để ghi nhận số replica thực tế sau execution
        sync_agent_with_k3s()

        return jsonify({
            "mode": agent_result["mode"],
            "action_taken": agent_result["action_taken"],
            "reason": agent_result["reason"],
            "replicas_before": execution_result["replicas_before"],
            "replicas_after": execution_result["replicas_after"],
            "execution_action": execution_result["action"],
            "changed": execution_result["changed"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/reset-state", methods=["POST"])
def reset_state():
    try:
        # Reset K3s mock state về ban đầu
        k3s_client.state.current_replicas = k3s_client.state.min_replicas

        # Reset AI agent state
        agent.current_replicas = agent.config.min_replicas
        agent.last_scale_monotonic = None

        # Đồng bộ lại agent theo K3s
        sync_agent_with_k3s()

        return jsonify({
            "message": "Đã reset state hệ thống.",
            "mode": agent.config.mode,
            "current_replicas": agent.current_replicas,
            "cooldown_active": False
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/test-email", methods=["POST"])
def test_email():
    try:
        subject, message = build_alert_message(
            current_cpu=78,
            predicted_cpu=92,
            threshold=80
        )
        email_sent = send_alert_email(subject, message)

        return jsonify({
            "message": "Đã thử gửi email cảnh báo.",
            "email_sent": email_sent
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}

        # 1. Trích xuất features & Suy luận qua ONNX
        features = [float(data[name]) for name in FEATURE_NAMES]
        input_array = np.array([features], dtype=np.float32)

        predicted_score = session.run(
            [output_name],
            {input_name: input_array}
        )[0][0][0]

        # 2. Đưa ra quyết định Scaling
        scaling_decision = policy.decide(float(predicted_score))

        # Đảm bảo Agent đồng bộ dữ liệu số replica hiện tại với K3s trước khi ra quyết định
        sync_agent_with_k3s()

        agent_result = agent.apply_decision(scaling_decision)
        email_sent = False
        try:
            if scaling_decision == "Scale Up" or agent_result["action_taken"] == "Scaled Up":
                current_cpu = round(float(features[0]), 2)
                predicted_cpu = round(min(float(predicted_score) / 1.5 * 100, 100), 2)
                threshold = 80.0

                subject, message = build_alert_message(
                    current_cpu=current_cpu,
                    predicted_cpu=predicted_cpu,
                    threshold=threshold
                )
                email_sent = send_alert_email(subject, message)
        except Exception as mail_err:
            print(f"[Mail Warning] Không gửi được email: {mail_err}")
        
        # 3. Map quyết định sang thực thi thật/mock trên K3s Cluster
        if agent_result["action_taken"] == "Scaled Up":
            execution_result = k3s_client.scale_up()
        elif agent_result["action_taken"] == "Scaled Down":
            execution_result = k3s_client.scale_down()
        else:
            execution_result = k3s_client.keep()

        # Cập nhật lại replica state sau khi cluster đã scale
        sync_agent_with_k3s()

        psutil.cpu_percent(interval=None)


        def get_current_system_metrics() -> dict:
            cpu_percent = psutil.cpu_percent(interval=0.3)
            memory = psutil.virtual_memory()

            return {
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "memory_used_mb": round(memory.used / 1024 / 1024, 2),
                "memory_total_mb": round(memory.total / 1024 / 1024, 2),
                "timestamp": int(time.time())
            }
        return jsonify({
            "predicted_score": round(float(predicted_score), 6),
            "scaling_decision": scaling_decision,
            "mode": agent_result["mode"],
            "replicas_before": agent_result["current_replicas_before"],
            "replicas_after": agent_result["current_replicas_after"],
            "action_taken": agent_result["action_taken"],
            "reason": agent_result["reason"],
            "email_sent": email_sent
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)