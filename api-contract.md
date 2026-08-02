# API Contract — Edge AI Orchestrator (K3s)

> File này thay thế mọi bản mô tả hợp đồng cũ (nhắc tới `app.py`,
> `get_live_metrics()`, `execute_k3s_scale()`, `predict_future_load()`).
> Những hàm đó **không còn tồn tại** trong repo — `app.py` đã bị xóa và
> kiến trúc thực tế đang chạy là `modules/ai_service.py` (xem `Dockerfile` CMD).
> Đây là bản mô tả khớp với code hiện có, đã chạy thử và xác nhận hoạt động.

## Entrypoint

- **Server thật:** `modules/ai_service.py` (Flask), chạy bằng
  `python modules/ai_service.py`, hoặc qua Docker (`CMD ["python", "modules/ai_service.py"]`).
- Template HTML nằm ở `templates/index.html`, `ai_service.py` tự trỏ
  `template_folder` về đó nên không cần chạy từ thư mục gốc.
- Model dùng: `models/linear_regression_model.onnx` (bắt buộc phải tồn tại,
  server sẽ raise `FileNotFoundError` khi khởi động nếu thiếu).
- **`models/lstm_model.onnx` hiện là file rỗng/placeholder (2 byte) — chưa dùng ở đâu, đừng load nó.**

## Phạm vi cố định (không tự ý đổi)

- 1 node duy nhất: `NODE_NAME = "edge-node-01"` (khai báo trong `modules/k3s_client.py`).
- 1 deployment: `deployment_name = "edge-ai-app"`, `namespace = "default"`
  (khai báo trong `K3sResourceState`, `modules/k3s_client.py`).
- Đây là quyết định phạm vi đồ án — không mở rộng thành multi-node/multi-cluster.

## Hợp đồng module (tên hàm/tham số — KHÔNG đổi)

### `modules/k3s_client.py` — Tầng 1, đã có bản real + auto-fallback

```python
class K3sResourceState:
    deployment_name: str
    namespace: str
    current_replicas: int
    min_replicas: int
    max_replicas: int

# Cùng interface, chọn 1 trong 2 qua create_k3s_client():
class K3sClientMock:  ...
class K3sClient:      ...   # gọi Kubernetes API thật

def create_k3s_client(state=None) -> tuple[K3sClient | K3sClientMock, str]:
    # Trả (client, "real") nếu kết nối được cluster + đọc được deployment.
    # Trả (client, "mock") nếu không (không có kubeconfig, cluster tắt,
    # deployment chưa apply...) — KHÔNG crash server, tự rơi về mock.
```

Cả hai class cùng interface:
```python
def get_status(self) -> dict          # {deployment_name, namespace, current_replicas, min_replicas, max_replicas}
def scale_up(self, step: int = 1) -> dict     # {action, replicas_before, replicas_after, changed}
def scale_down(self, step: int = 1) -> dict   # cùng shape scale_up
def keep(self) -> dict                         # {action:"keep", replicas_before, replicas_after, changed:False}
def get_pods(self, cpu_percent: float, memory_percent: float) -> list[dict]
    # [{name, node, cpu_percent, memory_percent, status}]
```

**`K3sClient` (bản thật) đã cài đặt:**
- `get_status()` đọc `spec.replicas` thật qua `AppsV1Api.read_namespaced_deployment`.
- `scale_up/scale_down/keep` gọi `patch_namespaced_deployment_scale` thật,
  giới hạn trong `[min_replicas, max_replicas]`.
- `get_pods()` gọi `list_namespaced_pod` (lọc theo `label_selector=app=<deployment_name>`)
  + metrics từ `metrics.k8s.io/v1beta1` (CustomObjectsApi), quy đổi
  usage/limit (hoặc usage/request nếu không có limit) ra %. Nếu
  metrics-server chưa cài hoặc pod chưa khai `resources.requests/limits`,
  rơi về `cpu_percent`/`memory_percent` truyền vào (CPU/RAM host qua psutil)
  làm dự phòng — không trả `null`/crash.
- Mọi call K8s API đều có `_request_timeout` ngắn (5–10s) để fallback nhanh
  thay vì treo theo timeout OS/TCP mặc định (từng đo được ~60s nếu không set).

**Để bật chế độ "real":**
1. `kubectl apply -f k8s/edge-ai-app-deployment.yaml` (tạo Deployment
   `edge-ai-app` trong namespace `default`, có sẵn labels + resource
   requests/limits đúng như `K3sClient` cần).
2. Cài metrics-server nếu cluster K3s chưa có sẵn (xem comment cuối file
   YAML) — thiếu bước này `get_status/scale` vẫn chạy thật, chỉ riêng
   `get_pods()` phải dùng CPU/RAM host làm dự phòng.
3. Đảm bảo `~/.kube/config` trỏ đúng context của cluster đó (kiểm tra bằng
   `kubectl get nodes`).
4. Chạy `python modules/ai_service.py`, xem log dòng `[K3s Client] Đang
   chạy ở chế độ: REAL`. `/status` cũng trả `"k3s_mode": "real"`, dashboard
   hiện badge xanh "K3s cluster thật" ở topbar thay vì badge vàng mock.

**Đã test end-to-end trên cluster thật (minikube, driver docker):**
`kubectl apply -f k8s/edge-ai-app-deployment.yaml` → khởi động `ai_service.py`
→ log in đúng `[K3s Client] Đang chạy ở chế độ: REAL` → `/status` trả
`"k3s_mode": "real"` với tên pod thật (`edge-ai-app-<hash>`), node thật
(`minikube`), CPU/RAM per-pod thật từ metrics-server → gọi `/manual-scale`
với action `Scale Up` → xác nhận bằng `kubectl get deployment edge-ai-app`
thấy replicas thật tăng `1 → 2`, `kubectl get pods` thấy pod thứ 2 thật được
tạo. Scale Down đưa cluster về lại 1 pod thành công. Không còn là giả định —
đường real đã chạy đúng trên cluster thật.

### `modules/ai_agent.py` + `modules/decision_policy.py` — việc của Người 2

```python
class AgentConfig: min_replicas, max_replicas, scale_up_step, scale_down_step, cooldown_seconds, mode
class AIAgent:
    def set_mode(self, mode: str)                 # "AUTO" | "MANUAL"
    def cooldown_active(self) -> bool
    def apply_decision(self, decision: str) -> dict   # decision ∈ {"Scale Up","Scale Down","Keep"}
    def manual_scale(self, action: str) -> dict

class DecisionPolicy:
    def decide(self, score: float) -> str         # trả "Scale Up" | "Scale Down" | "Keep"
```

`ai_service.py` suy luận `predicted_score` bằng ONNX rồi gọi
`DecisionPolicy.decide(score)` → `AIAgent.apply_decision(decision)`. Việc của
Người 2 là cải thiện model/ngưỡng quyết định — không đổi tên hàm.

### `notification/email/send_mail.py` — đã xong, đừng sửa

```python
def send_alert_email(subject: str, message: str) -> bool
def build_alert_message(current_cpu, predicted_cpu, threshold) -> tuple[str, str]
```

Cấu hình qua `notification/email/.env` (`EMAIL_SENDER`, `EMAIL_PASSWORD`,
`EMAIL_RECEIVER`, `SMTP_SERVER`, `SMTP_PORT`). File `config.json` cùng thư mục
**không được code đọc ở đâu cả** — chỉ để tham khảo, đừng để lộ secret thật
vào đó (đã dọn — xem mục Bảo mật bên dưới).

## Flask routes (đã test bằng curl, hoạt động đúng)

| Method | Path | Request body | Response |
|---|---|---|---|
| GET | `/` | – | render `index.html` |
| GET | `/status` | – | `{mode, deployment_name, namespace, current_replicas, min_replicas, max_replicas, cooldown_seconds, cpu_percent, memory_percent, pods[], history[], ai_accuracy, k3s_mode}` |
| GET | `/pods` | – | `{pods: [...]}` (giống field `pods` trong `/status`) |
| GET | `/history` | – | `{history: [...]}` (toàn bộ, không giới hạn 20 mục như trong `/status`) |
| POST | `/mode` | `{mode: "AUTO"\|"MANUAL"}` | `{message, mode}` |
| POST | `/manual-scale` | `{action: "Scale Up"\|"Scale Down"\|"Keep"}` | `{mode, action_taken, reason, replicas_before, replicas_after, execution_action, changed}` |
| POST | `/reset-state` | – | `{message, mode, current_replicas, cooldown_active}` |
| POST | `/test-email` | – | `{message, email_sent}` |
| POST | `/predict` | 10 field ONNX feature (xem `FEATURE_NAMES` trong `ai_service.py`) | `{predicted_score, scaling_decision, mode, replicas_before, replicas_after, action_taken, reason, email_sent}` |

`cpu_percent`/`memory_percent` trong `/status` là **CPU/RAM thật của máy host**
đo bằng `psutil` (không phải mock) — dùng làm proxy tạm thời cho tải cluster
trong lúc Người 1 chưa nối metrics-server thật.

`history` là danh sách sự kiện thật (không mock): mỗi lần `/predict` gây ra
Scale Up/Down thật, hoặc gửi được email cảnh báo, hoặc `/manual-scale` thay
đổi số replica, một entry `{type, message, detail, timestamp}` được append
vào `orchestrator_history` trong bộ nhớ (`modules/ai_service.py`).

`ai_accuracy` tính từ `predictions_log` (bộ nhớ, không mock): so sánh quyết
định của lần dự đoán trước với chiều biến động CPU thực tế đo được ở lần kế
tiếp. Trả `null` khi chưa đủ 2 lần dự đoán.

`k3s_mode` ∈ `"real" | "mock"` — cho biết `/status` đang phản ánh dữ liệu
cluster K8s thật hay đang chạy fallback mock. Dashboard hiển thị badge tương
ứng ở topbar, không bao giờ "giả vờ" là thật khi thực ra là mock.

## Đã hoàn thiện Tầng 1 (K8s API thật) trong lượt này

- `modules/k3s_client.py` có thêm class `K3sClient` gọi Kubernetes API thật
  (đọc/scale deployment, đọc CPU/RAM per-pod qua metrics-server) và factory
  `create_k3s_client()` tự chọn real/mock, không cần sửa gì ở `ai_service.py`
  ngoài đổi 1 dòng khởi tạo.
- Thêm `k8s/edge-ai-app-deployment.yaml` — manifest Deployment mẫu để có thứ
  thật mà scale (trước đây repo không có bất kỳ file YAML nào).
- **Đã test thật trên minikube** (bật lại được sau khi fix `hypervisorlaunchtype`
  + khởi động Docker Desktop): apply manifest, chạy server ở mode REAL, gọi
  `/manual-scale` và xác nhận bằng `kubectl` thấy deployment thật scale
  1 → 2 → 1 pod. Xem chi tiết ở mục `K3sClient` phía trên.

## Đã sửa trong lượt review này

- `render_template` được dùng ở route `/` nhưng chưa import trong `ai_service.py`
  → route `/` sập với `NameError` mọi lúc. Đã thêm import.
- `psutil` được import và dùng trong `ai_service.py` nhưng thiếu trong
  `requirements.txt` → cài từ `requirements.txt` sạch sẽ crash khi chạy server.
  Đã thêm `psutil==6.0.0`.
- `print()` các thông báo tiếng Việt (cảnh báo email) làm `/predict` trả
  HTTP 400 vì `UnicodeEncodeError` trên codepage mặc định của Windows console.
  Đã ép `sys.stdout`/`sys.stderr` sang UTF-8 khi khởi động server.
- Rò rỉ Gmail App Password thật trong `notification/email/config.json` (đã
  commit lên GitHub) — đã xóa secret khỏi file, untrack khỏi git, thêm vào
  `.gitignore`. **App Password cũ phải được thu hồi/đổi thủ công trên Google
  Account — không thể tự động hóa bước này.** Lịch sử git vẫn còn secret cũ
  cho tới khi lịch sử được viết lại (chưa làm, cần xác nhận riêng vì
  force-push sẽ ảnh hưởng tới mọi clone khác).
- Xóa thư mục `mail/` (bản sao trùng lặp không được code nào import, chứa
  cùng secret thật).
- Frontend (`templates/index.html`): xóa toàn bộ UI multi-chi nhánh
  (Hà Nội/HCM/Đà Nẵng), nhãn `admin@hospital.vn`, dữ liệu pod/lịch sử/stat
  card viết cứng, node giả `node-01/02/03`; nối lại bằng dữ liệu thật từ
  `/status`.

## Đã sửa thêm — lỗi ảnh hưởng trực tiếp tới độ đúng của AI (quan trọng)

- **`/predict` thiếu bước chuẩn hóa input.** `dataset/preprocessing.py` fit
  `StandardScaler` trên toàn bộ feature rồi mới train model (bước 12), lưu
  lại `dataset/data/output/scaler.pkl` đúng để dùng lúc suy luận — nhưng
  `ai_service.py` trước đây đưa thẳng input thô vào ONNX, bỏ qua scaler này
  hoàn toàn. Hệ quả: mọi giá trị người dùng nhập vào form "AI Predict trực
  tiếp" (vd. cpu_usage=78) lệch hẳn phân phối lúc train (cpu_usage lúc train
  chỉ nằm trong khoảng 0–4), khiến `predicted_score` vô nghĩa.
  Đã sửa: `ai_service.py` load `scaler.pkl` bằng `joblib`, gọi
  `scaler.transform([features])` trước khi đưa vào ONNX. **`/predict` giờ
  nhận input ở đơn vị GỐC của dataset (không phải z-score đã chuẩn hóa)** —
  xem cột tương ứng trong `dataset/data/output/cleaned_dataset.csv` để biết
  khoảng giá trị hợp lý cho từng field.
  Đã verify: gọi `/predict` với dòng đầu `cleaned_dataset.csv`
  (`resource_pressure_score` thật = 0.603) → `predicted_score` = 0.708,
  gần đúng — trước khi sửa, đưa thẳng giá trị thô này vào sẽ cho kết quả
  sai lệch hoàn toàn vì model chưa từng thấy input ở scale đó.
  Ảnh hưởng liên quan: `templates/index.html` (`fillSampleData()`) và
  `modules/test_predict_from_xtest.py` / `test_predict_batch_api.py` trước
  đó dùng dữ liệu đã scale sẵn từ `X_test.csv` — nếu không sửa theo sẽ bị
  scale 2 lần. Đã đổi cả 3 chỗ sang đọc từ `cleaned_dataset.csv` (dữ liệu
  thô, đúng hợp đồng mới của `/predict`).
- **`/reset-state` không thực sự scale cluster thật.** Route cũ gán tay
  `k3s_client.state.current_replicas = min_replicas` rồi gọi
  `sync_agent_with_k3s()` — với `K3sClient` thật, hàm sync này đọc lại
  `spec.replicas` từ cluster và **ghi đè** giá trị vừa gán, nên reset không
  có tác dụng gì trên deployment thật (chỉ đúng "tình cờ" với mock vì mock
  không có nguồn sự thật nào khác để ghi đè lại).
  Đã verify lỗi này bằng cách gọi `/reset-state` trên cluster thật (minikube)
  — `kubectl get deployment` vẫn giữ nguyên số replica cũ dù response trả
  `current_replicas: 1`.
  Đã sửa: route giờ gọi `k3s_client.scale_down(step=...)` với đúng số bước
  cần thiết để về `min_replicas`, hoạt động đúng trên cả real và mock. Đã
  verify lại bằng `kubectl` — deployment thật scale đúng về 1/1.
