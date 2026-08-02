# Edge AI Orchestrator cho K3s

AI dự báo tải (CPU/RAM) để chủ động scale pod trước khi tải tăng, thay vì
chờ tải vượt ngưỡng rồi mới phản ứng như HPA mặc định. Phạm vi demo: 1
cluster, 1 node, 1 deployment (`edge-ai-app`) — không phải hệ thống
multi-tenant/multi-node.

Chi tiết hợp đồng API/module giữa các phần (tên hàm, request/response
shape) xem [`api-contract.md`](api-contract.md).

## Kiến trúc

```
modules/ai_service.py      Flask server — entrypoint thật (không phải app.py)
modules/k3s_client.py      Đọc/scale K8s: K3sClient (thật) + K3sClientMock (fallback)
modules/ai_agent.py        Áp quyết định scale (AUTO/MANUAL), cooldown
modules/decision_policy.py Quy điểm dự đoán ONNX ra "Scale Up/Down/Keep"
notification/email/        Gửi email cảnh báo qua SMTP
templates/index.html       Dashboard giám sát (fetch dữ liệu từ các route bên dưới)
models/                    Model đã huấn luyện (.pkl) + đã export (.onnx)
dataset/                   Dữ liệu train/test + script tiền xử lý, huấn luyện
k8s/                       Manifest Deployment mẫu để scale trên cluster thật
```

## Cài đặt

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

Model ONNX (`models/linear_regression_model.onnx`) đã có sẵn trong repo —
không cần huấn luyện lại để chạy demo. Nếu muốn huấn luyện lại từ đầu:

```bash
python modules/train_linear.py     # đọc dataset/data/output/*.csv, ghi ra models/*.pkl
python modules/export_onnx.py      # chuyển .pkl -> .onnx
```

## Cấu hình email cảnh báo (tùy chọn)

Sửa `notification/email/.env`:

```
EMAIL_SENDER=
EMAIL_PASSWORD=
EMAIL_RECEIVER=
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

`EMAIL_PASSWORD` phải là **App Password** của Gmail (không phải mật khẩu
đăng nhập thường). File này bị `.gitignore`, không commit lên git. Không
điền gì thì hệ thống vẫn chạy bình thường, chỉ bỏ qua bước gửi email.

## Chạy server

```bash
python modules/ai_service.py
```

Mặc định chạy ở `http://localhost:5000`. Mở trình duyệt vào `/` để xem
dashboard.

Ngay khi khởi động, server tự thử kết nối cluster K8s thật qua kubeconfig
(`~/.kube/config`). Log sẽ in một trong hai dòng:

```
[K3s Client] Đang chạy ở chế độ: REAL   # đã đọc được cluster thật
[K3s Client] Đang chạy ở chế độ: MOCK   # không kết nối được, tự dùng dữ liệu giả lập
```

`k3s_mode` cũng được trả trong response của `GET /status` và hiển thị bằng
badge màu ở góc trên dashboard — không bao giờ hiển thị "thật" khi thực ra
đang chạy mock.

### Vòng lặp AI tự động (khép kín — không cần bấm nút)

Ngay sau khi khởi động, một thread nền tự chạy: cứ mỗi
`AUTO_PREDICT_INTERVAL_SECONDS` giây (mặc định **15s**, đổi bằng biến môi
trường cùng tên) hệ thống tự lấy số liệu thật (CPU/RAM host, cpu/memory
request-limit thật của deployment, network/disk throughput host, độ trễ tới
K8s API, tuổi pod thật) rồi tự chạy lại đúng pipeline suy luận ONNX + quyết
định của `/predict` — **khi mode = AUTO, nó tự scale cluster thật, không cần
ai bấm nút.** Xem bảng "feature nào lấy từ đâu" trong
[`api-contract.md`](api-contract.md) để biết cái nào đo per-pod thật, cái
nào là proxy ở mức host (do K3s/Minikube demo chưa có Prometheus/cAdvisor).

Route `/predict` vẫn còn — dùng để test tay với giá trị tùy ý (nút "Nạp
sample" ở dashboard nạp 1 dòng từ `dataset/data/output/cleaned_dataset.csv`).

### Bật chế độ REAL trên cluster thật (K3s hoặc Minikube)

1. Đảm bảo `kubectl get nodes` chạy được (kubeconfig đúng context).
2. Áp deployment mẫu:
   ```bash
   kubectl apply -f k8s/edge-ai-app-deployment.yaml
   ```
3. Cluster cần có **metrics-server** để dashboard đọc được CPU/RAM per-pod
   thật (Minikube: `minikube addons enable metrics-server`; K3s: xem ghi chú
   cuối file `k8s/edge-ai-app-deployment.yaml`). Thiếu bước này thì
   `get_status`/scale vẫn chạy thật, riêng số CPU/RAM hiển thị sẽ tạm dùng
   CPU/RAM của máy host làm dự phòng.
4. Chạy lại `python modules/ai_service.py`.

## API chính

| Method | Path | Mô tả |
|---|---|---|
| GET | `/status` | Trạng thái tổng hợp: replicas, CPU/RAM, pod list, lịch sử, AI accuracy, k3s_mode, lần dự đoán tự động gần nhất |
| GET | `/pods` | Danh sách pod hiện tại |
| GET | `/history` | Toàn bộ lịch sử scale/cảnh báo |
| POST | `/mode` | Đổi `AUTO`/`MANUAL` |
| POST | `/manual-scale` | Scale thủ công khi ở mode `MANUAL` |
| GET/POST | `/thresholds` | Đọc/đổi ngưỡng `scale_down`/`scale_up` thật của `DecisionPolicy` |
| POST | `/predict` | Chạy suy luận ONNX + tự scale nếu ở mode `AUTO` |
| POST | `/reset-state` | Reset trạng thái orchestrator |
| POST | `/test-email` | Gửi thử email cảnh báo |

Chi tiết request/response shape: [`api-contract.md`](api-contract.md).

## Test nhanh không qua dashboard

```bash
# Suy luận trực tiếp bằng model ONNX, không cần server chạy
python modules/onnx_predict_and_decide.py

# Gọi thử /predict qua HTTP (cần server đang chạy)
python modules/test_predict_from_xtest.py
python modules/test_predict_batch_api.py
```

## Unit test

```bash
python -m unittest discover -s tests -v
```

Test các phần logic thuần (không cần cluster/model): `decision_policy`,
`ai_agent`, `K3sClientMock`.

## Hạn chế đã biết

- Bảo mật: repo từng commit nhầm App Password Gmail thật vào
  `notification/email/config.json` (đã sửa ở commit hiện tại, nhưng **secret
  cũ vẫn còn trong lịch sử git** — cần rotate password + rewrite history
  trước khi coi là đã xử lý dứt điểm).
- Môi trường demo hiện dùng **Minikube** (driver Docker), không phải cụm K3s
  thật — về mặt API thì tương thích (cùng Kubernetes API), nhưng nên nói rõ
  khi báo cáo/bảo vệ.
