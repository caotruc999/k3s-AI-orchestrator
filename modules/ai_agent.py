"""
modules/ai_agent.py
====================
Tích hợp model Linear Regression đã huấn luyện (models/linear_regression_model.pkl)
thay cho code mock (random) trước đây.

QUAN TRỌNG - đọc trước khi sửa file này:
------------------------------------------------------------------
Model hiện tại được huấn luyện để chấm "resource_pressure_score" (0 -> ~1.5)
dựa trên 10 đặc trưng TĨNH của 1 pod tại 1 thời điểm (cpu_usage, memory_usage,
cpu_request, cpu_limit, memory_request, memory_limit, network_bandwidth_usage,
network_latency, disk_io, pod_lifetime_seconds) - đây là bài toán "chấm điểm
hiện trạng", KHÔNG PHẢI bài toán "dự báo tương lai" mà predict_future_load()
được thiết kế để làm (xem docstring hàm bên dưới).

Vì app.py hiện tại vẫn gọi predict_future_load(cpu_history) và mong nhận về
1 list 5 giá trị %CPU dự báo, hàm bên dưới dùng resource_pressure_score làm
"điểm neo" rồi tạo ra 1 chuỗi 5 giá trị dao động quanh điểm đó - đây là GIẢI
PHÁP TẠM THỜI để cắm được model thật vào hệ thống đang chạy, KHÔNG PHẢI dự
báo chuỗi thời gian thật sự. Khi model LSTM (dự báo thật) huấn luyện xong,
cần thay thế toàn bộ khối "BRIDGE" bên dưới bằng lời gọi LSTM trực tiếp.
------------------------------------------------------------------

Các giá trị KHÔNG lấy được từ live metrics hiện tại (cpu_request, cpu_limit,
memory_request, memory_limit, network_bandwidth_usage, network_latency,
disk_io, pod_lifetime_seconds) đang dùng giá trị mặc định khai báo trong
DEFAULT_POD_PROFILE bên dưới - SỬA LẠI các số này cho khớp với cấu hình pod
thật của nhóm khi có dữ liệu, hoặc nhờ Người 1 bổ sung vào get_live_metrics().
"""

import os
import random
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Đường dẫn tới model - tính tương đối theo vị trí file này, KHÔNG dùng path
# tuyệt đối kiểu "D:\..." để mọi máy trong nhóm chạy đều được sau khi git pull.
# ---------------------------------------------------------------------------
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULES_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# Thứ tự 10 cột PHẢI khớp chính xác với lúc huấn luyện (đọc trực tiếp từ
# model.feature_names_in_ nên không cần bạn tự gõ tay, tránh sai thứ tự).
FEATURE_ORDER = [
    "cpu_usage", "memory_usage", "cpu_request", "cpu_limit",
    "memory_request", "memory_limit", "network_bandwidth_usage",
    "network_latency", "disk_io", "pod_lifetime_seconds",
]

# Giá trị mặc định cho các đặc trưng mà live metrics hiện tại CHƯA cung cấp.
# TODO (Người 1 / Người 2): thay các số này bằng cấu hình pod thật khi có,
# hoặc bổ sung trực tiếp vào get_live_metrics() trong k3s_client.py.
DEFAULT_POD_PROFILE = {
    "cpu_request": 1.0,
    "cpu_limit": 2.0,
    "memory_request": 2000.0,
    "memory_limit": 4000.0,
    "network_bandwidth_usage": 400.0,
    "network_latency": 80.0,
    "disk_io": 400.0,
    "pod_lifetime_seconds": 50000.0,
}

_model = None
_scaler = None
_load_attempted = False


def _try_load_artifacts():
    """
    Nạp model + scaler đúng 1 lần (lazy load), không làm sập app nếu thiếu file.
    In cảnh báo rõ ràng ra console để biết ngay đang chạy chế độ thật hay mock.
    """
    global _model, _scaler, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    if not os.path.exists(MODEL_PATH):
        print(f"[AI Agent] KHÔNG tìm thấy model tại {MODEL_PATH}. "
              f"Đang chạy ở CHẾ ĐỘ MOCK (số ngẫu nhiên).")
        return

    if not os.path.exists(SCALER_PATH):
        print(f"[AI Agent] Tìm thấy model nhưng THIẾU scaler.pkl tại {SCALER_PATH}. "
              f"Không thể đưa dữ liệu thô vào model (sẽ cho kết quả sai). "
              f"Đang chạy ở CHẾ ĐỘ MOCK cho tới khi có scaler.pkl. "
              f"Cần lấy file này từ bước preprocessing.py (thư mục dataset/data/output/scaler.pkl).")
        return

    try:
        _model = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        print("[AI Agent] Đã nạp thành công model + scaler thật. Chạy CHẾ ĐỘ THẬT.")
    except Exception as e:
        print(f"[AI Agent] Lỗi khi nạp model/scaler: {e}. Lùi về CHẾ ĐỘ MOCK.")
        _model = None
        _scaler = None


def _build_feature_row(live_cpu: float, live_ram: float) -> np.ndarray:
    """Ghép live metrics (cpu/ram thật) với các giá trị mặc định thành đúng 10 cột model cần."""
    row = dict(DEFAULT_POD_PROFILE)
    row["cpu_usage"] = live_cpu
    row["memory_usage"] = live_ram
    return np.array([[row[col] for col in FEATURE_ORDER]])


def _predict_pressure_score(live_cpu: float, live_ram: float) -> float:
    """Chạy inference thật: scale đúng cách rồi đưa qua Linear Regression."""
    X = _build_feature_row(live_cpu, live_ram)
    X_scaled = _scaler.transform(X)
    score = _model.predict(X_scaled)[0]
    return float(np.clip(score, 0, 1.5))


def predict_future_load(cpu_history):
    """
    Hàm nhận vào list lịch sử CPU, trả về list 5 phần tử dự báo cho 5 phút tới.

    Ưu tiên dùng model Linear Regression thật (nếu đã có đủ model.pkl +
    scaler.pkl). Nếu thiếu file, tự động lùi về code mock cũ - KHÔNG BAO GIỜ
    làm crash app.py, để nhóm vẫn chạy demo được ngay cả khi model chưa sẵn sàng.
    """
    _try_load_artifacts()

    last_val = cpu_history[-1] if cpu_history else 70

    if _model is not None and _scaler is not None:
        # ----- BRIDGE: dùng resource_pressure_score làm điểm neo (xem docstring đầu file) -----
        live_ram_estimate = last_val * 0.9  # chưa có RAM lịch sử riêng, ước lượng tạm theo CPU
        score = _predict_pressure_score(last_val, live_ram_estimate)
        anchor_percent = float(np.clip(score / 1.5 * 100, 0, 100))

        predictions = [
            anchor_percent + random.uniform(-3, 3) for _ in range(5)
        ]
        return [max(0, min(100, round(v))) for v in predictions]

    # ----- CHẾ ĐỘ MOCK (giữ nguyên hành vi cũ nếu chưa có model) -----
    predictions = [
        int(last_val + random.randint(-4, 4)),
        int(last_val + random.randint(-3, 6)),
        int(last_val + random.randint(-2, 8)),
        int(last_val + random.randint(-4, 5)),
        int(last_val + random.randint(-5, 3)),
    ]
    return [max(0, min(100, val)) for val in predictions]