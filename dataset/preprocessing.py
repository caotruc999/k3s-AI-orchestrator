import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw"
OUTPUT_PATH = "data/output"
VISUAL_PATH = os.path.join(OUTPUT_PATH, "visualization")

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VISUAL_PATH, exist_ok=True)

print("Dang tien hanh tien xu ly du lieu...")

# =========================
# 1. Helpers
# =========================
def normalize_columns(df):
    df.columns = [
        re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
        for c in df.columns
    ]
    return df

def find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Khong tim thay cot nao trong danh sach: {candidates}")
    return None

# =========================
# 2. Read data
# =========================
performance_df = pd.read_csv(os.path.join(RAW_PATH, "kubernetes_performance_metrics_dataset.csv"))
resource_df = pd.read_csv(os.path.join(RAW_PATH, "kubernetes_resource_allocation_dataset.csv"))

performance_df = normalize_columns(performance_df)
resource_df = normalize_columns(resource_df)

print("Cot performance:", performance_df.columns.tolist())
print("Cot resource   :", resource_df.columns.tolist())

# =========================
# 3. Detect key columns
# =========================
pod_col_perf = find_col(performance_df, ["pod_name", "podname"])
ns_col_perf = find_col(performance_df, ["namespace"])

pod_col_res = find_col(resource_df, ["pod_name", "podname"])
ns_col_res = find_col(resource_df, ["namespace"])

performance_df = performance_df.rename(columns={pod_col_perf: "pod_name", ns_col_perf: "namespace"})
resource_df = resource_df.rename(columns={pod_col_res: "pod_name", ns_col_res: "namespace"})

# =========================
# 4. Merge
# =========================
dataset = performance_df.merge(resource_df, on=["pod_name", "namespace"], how="inner")
print(f"So dong sau merge: {dataset.shape[0]}")

# =========================
# 5. Auto-map feature columns
# =========================
cpu_usage_col = find_col(dataset, ["cpu_usage", "node_cpu_usage", "cpu_allocation_efficiency"])
memory_usage_col = find_col(dataset, ["memory_usage", "node_memory_usage", "memory_allocation_efficiency"])

cpu_request_col = find_col(dataset, ["cpu_request"], required=False)
cpu_limit_col = find_col(dataset, ["cpu_limit"], required=False)
memory_request_col = find_col(dataset, ["memory_request"], required=False)
memory_limit_col = find_col(dataset, ["memory_limit"], required=False)

network_bw_col = find_col(dataset, ["network_bandwidth_usage", "network_usage"], required=False)
network_latency_col = find_col(dataset, ["network_latency"], required=False)
disk_io_col = find_col(dataset, ["disk_io"], required=False)
pod_lifetime_col = find_col(dataset, ["pod_lifetime_seconds"], required=False)
scaling_event_col = find_col(dataset, ["scaling_event"], required=False)

selected_map = {
    "cpu_usage": cpu_usage_col,
    "memory_usage": memory_usage_col,
    "cpu_request": cpu_request_col,
    "cpu_limit": cpu_limit_col,
    "memory_request": memory_request_col,
    "memory_limit": memory_limit_col,
    "network_bandwidth_usage": network_bw_col,
    "network_latency": network_latency_col,
    "disk_io": disk_io_col,
    "pod_lifetime_seconds": pod_lifetime_col,
    "scaling_event": scaling_event_col
}

# chỉ giữ cột tìm thấy
selected_map = {k: v for k, v in selected_map.items() if v is not None}
dataset = dataset[list(selected_map.values())].copy()
dataset = dataset.rename(columns={v: k for k, v in selected_map.items()})

print("Cac cot duoc su dung:", dataset.columns.tolist())

# =========================
# 6. Clean data
# =========================
dataset = dataset.drop_duplicates()
dataset = dataset.dropna()

# ép kiểu numeric nếu cần
for col in dataset.columns:
    if col != "scaling_event":
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce")

dataset = dataset.dropna()
print(f"So dong sau xoa trung & null: {dataset.shape[0]}")

# =========================
# 7. Tao target cho Linear Regression
# =========================
# Nếu có request/limit thì tận dụng, nếu không thì fallback sang usage trực tiếp
eps = 1e-6

if "cpu_limit" in dataset.columns:
    dataset["cpu_util"] = dataset["cpu_usage"] / (dataset["cpu_limit"] + eps)
else:
    dataset["cpu_util"] = dataset["cpu_usage"]

if "memory_limit" in dataset.columns:
    dataset["memory_util"] = dataset["memory_usage"] / (dataset["memory_limit"] + eps)
else:
    dataset["memory_util"] = dataset["memory_usage"]

# Điểm áp lực tài nguyên liên tục cho bài toán hồi quy
dataset["resource_pressure_score"] = (
    0.45 * dataset["cpu_util"] +
    0.45 * dataset["memory_util"]
)

if "network_latency" in dataset.columns:
    latency_norm = dataset["network_latency"] / (dataset["network_latency"].max() + eps)
    dataset["resource_pressure_score"] += 0.10 * latency_norm

# clip để tránh giá trị quá lớn
dataset["resource_pressure_score"] = dataset["resource_pressure_score"].clip(0, 1.5)

print(dataset["resource_pressure_score"].describe())

# =========================
# 8. Feature set
# =========================
drop_cols = ["resource_pressure_score", "scaling_event", "cpu_util", "memory_util"]
feature_cols = [c for c in dataset.columns if c not in drop_cols]

# =========================
# 9. Remove outlier bằng IQR
# =========================
Q1 = dataset[feature_cols].quantile(0.25)
Q3 = dataset[feature_cols].quantile(0.75)
IQR = Q3 - Q1

mask = ~(
    ((dataset[feature_cols] < (Q1 - 1.5 * IQR)) |
     (dataset[feature_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
)

dataset = dataset[mask].copy()
print(f"So dong sau loc outlier IQR: {dataset.shape[0]}")

# =========================
# 10. Save cleaned dataset
# =========================
dataset.to_csv(os.path.join(OUTPUT_PATH, "cleaned_dataset.csv"), index=False)
dataset.describe().to_csv(os.path.join(OUTPUT_PATH, "dataset_summary.csv"))

feature_dictionary = pd.DataFrame({
    "feature": feature_cols + ["resource_pressure_score"],
    "role": ["Feature"] * len(feature_cols) + ["Target_Regression"]
})
feature_dictionary.to_csv(os.path.join(OUTPUT_PATH, "feature_dictionary.csv"), index=False)

# =========================
# 11. Visualization
# =========================
dataset[feature_cols + ["resource_pressure_score"]].hist(figsize=(14, 10))
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "histogram.png"), dpi=200)
plt.close()

plt.figure(figsize=(12, 6))
dataset[feature_cols + ["resource_pressure_score"]].boxplot(rot=90)
plt.title("Boxplot after Outlier Removal")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "boxplot.png"), dpi=200)
plt.close()

corr = dataset[feature_cols + ["resource_pressure_score"]].corr(numeric_only=True)
plt.figure(figsize=(10, 8))
plt.imshow(corr, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(VISUAL_PATH, "correlation.png"), dpi=200)
plt.close()

if "cpu_usage" in dataset.columns and "memory_usage" in dataset.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(dataset["cpu_usage"], dataset["memory_usage"], alpha=0.5)
    plt.xlabel("CPU Usage")
    plt.ylabel("Memory Usage")
    plt.title("CPU vs Memory Usage")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUAL_PATH, "cpu_vs_memory.png"), dpi=200)
    plt.close()

# =========================
# 12. Scaling + Split
# =========================
training_dataset = dataset.copy()

scaler = StandardScaler()
training_dataset[feature_cols] = scaler.fit_transform(training_dataset[feature_cols])

X = training_dataset[feature_cols]
y = training_dataset["resource_pressure_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print(f"Train size: {X_train.shape[0]} mau")
print(f"Test size : {X_test.shape[0]} mau")

X_train.to_csv(os.path.join(OUTPUT_PATH, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(OUTPUT_PATH, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(OUTPUT_PATH, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(OUTPUT_PATH, "y_test.csv"), index=False)

joblib.dump(scaler, os.path.join(OUTPUT_PATH, "scaler.pkl"))

print("Thanh cong preprocessing.")