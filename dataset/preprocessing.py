import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  
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

# 5. Chọn Feature
selected_columns = [
    "cpu_usage", "memory_usage", "cpu_request", "cpu_limit", 
    "memory_request", "memory_limit", "network_bandwidth_usage", 
    "network_latency", "disk_io", "pod_lifetime_seconds", "scaling_event"
]
dataset = dataset[selected_columns]

# 6. Làm sạch
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()
print(f"👉 Số dòng sau khi xóa Trùng & Null: {dataset.shape[0]}")

print("🔄 Đang tạo Label theo quy tắc mới...")

dataset['cpu_util'] = dataset['cpu_usage'] / dataset['cpu_limit']
dataset['memory_util'] = dataset['memory_usage'] / dataset['memory_limit']

def create_label(row):
    if row['cpu_util'] >= 0.80 or row['memory_util'] >= 0.80:
        return 2
    # Ưu tiên 2: Dư thừa tài nguyên -> Scale Down (0)
    elif row['cpu_util'] <= 0.30 and row['memory_util'] <= 0.30:
        return 0
    # Còn lại: Giữ nguyên -> Keep (1)
    else:
        return 1

dataset['label'] = dataset.apply(create_label, axis=1)
dataset = dataset.drop(columns=['cpu_util', 'memory_util', 'scaling_event'])

print("👉 Phân bố Label:")
print(dataset['label'].value_counts())
print("\nTỷ lệ:")
print(dataset['label'].value_counts(normalize=True)*100)

# Định nghĩa feature columns (loại trừ label)
feature_cols = [c for c in dataset.columns if c != "label"]

# 8. Loại Outlier bằng IQR
Q1 = dataset[feature_cols].quantile(0.25)
Q3 = dataset[feature_cols].quantile(0.75)
IQR = Q3 - Q1
mask = ~(((dataset[feature_cols] < (Q1 - 1.5 * IQR)) | (dataset[feature_cols] > (Q3 + 1.5 * IQR))).any(axis=1))
dataset = dataset[mask]
print(f"👉 Số dòng sau khi lọc Outlier IQR: {dataset.shape[0]}")

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

# ==================== VẼ BIỂU ĐỒ ====================
# 12. Histogram
dataset.hist(figsize=(14, 10))
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "histogram.png"))
plt.close()

# 13. Boxplot
plt.figure(figsize=(12, 6))
dataset.boxplot(rot=90)
plt.title("Boxplot after Outlier Removal")
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
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "correlation.png"))
plt.close()

# 15. Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(dataset["cpu_usage"], dataset["memory_usage"], alpha=0.5)
plt.xlabel("CPU Usage")
plt.ylabel("Memory Usage")
plt.title("CPU vs Memory Usage")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "cpu_vs_memory.png"))
plt.close()


print("\n🔄 Đang chuẩn hóa dữ liệu và chia Train/Test...")

training_dataset = dataset.copy()
scaler = StandardScaler()
training_dataset[feature_cols] = scaler.fit_transform(training_dataset[feature_cols])

X = training_dataset[feature_cols]
y = training_dataset['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,           # 80% train - 20% test
    random_state=42, 
    stratify=y              
)

print(f"✅ Train size: {X_train.shape[0]} mẫu")
print(f"✅ Test size : {X_test.shape[0]} mẫu")

# Lưu các file
X_train.to_csv(os.path.join(OUTPUT_PATH, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_PATH, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_PATH, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_PATH, "y_test.csv"), index=False)

import joblib
joblib.dump(scaler, os.path.join(OUTPUT_PATH, "scaler.pkl"))

print("\n🎉 [THÀNH CÔNG]")
print(f"   - Train: {X_train.shape[0]} mẫu")
print(f"   - Test : {X_test.shape[0]} mẫu")