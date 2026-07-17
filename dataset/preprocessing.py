import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

RAW_PATH = "data/raw"
OUTPUT_PATH = "data/output"
VISUAL_PATH = os.path.join(OUTPUT_PATH, "visualization")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUAL_PATH, exist_ok=True)

print("🔍 ĐANG TIẾN HÀNH TIỀN XỬ LÝ DỮ LIỆU...")

# 3. Đọc dữ liệu
performance_df = pd.read_csv(os.path.join(RAW_PATH, "kubernetes_performance_metrics_dataset.csv"))
resource_df = pd.read_csv(os.path.join(RAW_PATH, "kubernetes_resource_allocation_dataset.csv"))

# 4. Merge Dataset
dataset = performance_df.merge(resource_df, on=["pod_name", "namespace"], how="inner")
print(f"👉 Số dòng sau khi gộp file (Merge): {dataset.shape[0]}")

# ==================== PHẦN SỬA ĐỔI CHÍNH ====================

# 5. Chọn Feature (bỏ scaling_event vì sẽ tạo label mới)
selected_columns = [
    "cpu_usage", "memory_usage", "cpu_request", "cpu_limit", 
    "memory_request", "memory_limit", "network_bandwidth_usage", 
    "network_latency", "disk_io", "pod_lifetime_seconds",
    "cpu_allocation_efficiency"   # Thêm cột này để làm CPU dự báo
]
dataset = dataset[selected_columns]

# 6. Làm sạch dữ liệu (Xóa trùng & Null cơ bản)
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()
print(f"👉 Số dòng sau khi xóa Trùng & Null (B6): {dataset.shape[0]}")

# ===================== TẠO LABEL MỚI =====================
# Theo quy tắc bạn yêu cầu:
# - Scale Up (2) : CPU dự báo > 75%
# - Scale Down (0): CPU thực tế < 40% VÀ CPU dự báo < 50%
# - Keep (1): Còn lại

print("🔄 Đang tạo Label theo quy tắc CPU Live & Predicted...")

# Tính % utilization
dataset['cpu_live_percent'] = (dataset['cpu_usage'] / dataset['cpu_limit']) * 100
dataset['cpu_pred_percent'] = dataset['cpu_allocation_efficiency'] * 100   # Dùng efficiency làm predicted

def create_label(row):
    live = row['cpu_live_percent']
    pred = row['cpu_pred_percent']
    
    if pd.isna(live) or pd.isna(pred):
        return 1  # Keep nếu dữ liệu thiếu
    
    # Scale Up
    if pred > 75:
        return 2
    # Scale Down
    elif live < 40 and pred < 50:
        return 0
    # Keep
    else:
        return 1

dataset['label'] = dataset.apply(create_label, axis=1)

# Xóa các cột trung gian (không cần thiết cho training)
dataset = dataset.drop(columns=['cpu_live_percent', 'cpu_pred_percent'])

print(f"👉 Phân bố Label sau khi tạo:")
print(dataset['label'].value_counts())
print(dataset['label'].value_counts(normalize=True) * 100)

# =========================================================

# Định nghĩa feature columns (loại trừ label)
feature_cols = [c for c in dataset.columns if c != "label"]

# 8. Loại Outlier bằng IQR
Q1 = dataset[feature_cols].quantile(0.25)
Q3 = dataset[feature_cols].quantile(0.75)
IQR = Q3 - Q1
mask = ~(((dataset[feature_cols] < (Q1 - 1.5 * IQR)) | (dataset[feature_cols] > (Q3 + 1.5 * IQR))).any(axis=1))
dataset = dataset[mask]
print(f"👉 Số dòng sau khi lọc Outlier IQR (B8): {dataset.shape[0]}")

# 9. Xuất cleaned_dataset
dataset.to_csv(os.path.join(OUTPUT_PATH, "cleaned_dataset.csv"), index=False)

# 10. Xuất thống kê
dataset.describe().to_csv(os.path.join(OUTPUT_PATH, "dataset_summary.csv"))

# 11. Feature Dictionary (cập nhật)
feature_dictionary = pd.DataFrame({
    "feature": [
        "cpu_usage", "memory_usage", "cpu_request", "cpu_limit", 
        "memory_request", "memory_limit", "network_bandwidth_usage", 
        "network_latency", "disk_io", "pod_lifetime_seconds", "label"
    ],
    "role": [
        "Feature", "Feature", "Feature", "Feature", 
        "Feature", "Feature", "Feature", "Feature", 
        "Feature", "Feature", "Target"
    ]
})
feature_dictionary.to_csv(os.path.join(OUTPUT_PATH, "feature_dictionary.csv"), index=False)

# Phần vẽ biểu đồ và chuẩn hóa giữ nguyên (từ 12 đến 16)
# ... (giữ nguyên code cũ từ Histogram đến StandardScaler)

# 12. Histogram
dataset.hist(figsize=(14, 10))
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "histogram.png"))
plt.close()

# 13. Boxplot
plt.figure(figsize=(12, 6))
dataset.boxplot(rot=90)
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "boxplot.png"))
plt.close()

# 14. Correlation Matrix
corr = dataset.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "correlation.png"))
plt.close()

# 15. Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(dataset["cpu_usage"], dataset["memory_usage"], alpha=0.5)
plt.xlabel("CPU Usage")
plt.ylabel("Memory Usage")
plt.title("CPU vs Memory")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "cpu_vs_memory.png"))
plt.close()

# 16. Chuẩn hóa dữ liệu
training_dataset = dataset.copy()
scaler = StandardScaler()
training_dataset[feature_cols] = scaler.fit_transform(training_dataset[feature_cols])
training_dataset.to_csv(os.path.join(OUTPUT_PATH, "training_dataset.csv"), index=False)

print("\n🎉 [THÀNH CÔNG] Pipeline đã chạy xong với Label mới theo quy tắc CPU!")