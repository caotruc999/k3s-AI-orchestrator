import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# 5. Chọn Feature
selected_columns = [
    "cpu_usage", "memory_usage", "cpu_request", "cpu_limit", 
    "memory_request", "memory_limit", "network_bandwidth_usage", 
    "network_latency", "disk_io", "pod_lifetime_seconds", "scaling_event"
]
dataset = dataset[selected_columns]

# 6. Làm sạch dữ liệu (Xóa trùng & Null cơ bản)
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()
print(f"👉 Số dòng sau khi xóa Trùng & Null (B6): {dataset.shape[0]}")

# 7. Encode Label (Đã sửa đổi cho kiểu dữ liệu Boolean True/False)
# Ánh xạ: False (Không scale) -> 0, True (Có thực hiện scale) -> 1
label_mapping = {False: 0, True: 1}
dataset["label"] = dataset["scaling_event"].map(label_mapping)
dataset.drop(columns=["scaling_event"], inplace=True)
dataset = dataset.dropna(subset=["label"])
print(f"👉 Số dòng sau khi mã hóa Label (B7): {dataset.shape[0]}")

# Định nghĩa các cột tính năng (loại trừ cột nhãn 'label')
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

# 11. Feature Dictionary
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

# 16. Chuẩn hóa dữ liệu (StandardScaler)
training_dataset = dataset.copy()
scaler = StandardScaler()
training_dataset[feature_cols] = scaler.fit_transform(training_dataset[feature_cols])
training_dataset.to_csv(os.path.join(OUTPUT_PATH, "training_dataset.csv"), index=False)

print("\n🎉 [THÀNH CÔNG] Toàn bộ Pipeline đã chạy xong. Dữ liệu và biểu đồ đã được xuất đầy đủ!")