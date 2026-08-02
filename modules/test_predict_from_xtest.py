import os
import sys
import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULES_DIR)

# /predict tự áp scaler.pkl (xem modules/ai_service.py) nên phải test bằng dữ
# liệu THÔ (cleaned_dataset.csv), không phải X_test.csv (đã StandardScaler
# sẵn) — nếu không sẽ bị scale 2 lần và dự đoán sai.
CLEANED_PATH = os.path.join(BASE_DIR, "dataset", "data", "output", "cleaned_dataset.csv")
API_URL = "http://127.0.0.1:5000/predict"

FEATURE_NAMES = [
    "cpu_usage", "memory_usage", "cpu_request", "cpu_limit",
    "memory_request", "memory_limit", "network_bandwidth_usage",
    "network_latency", "disk_io", "pod_lifetime_seconds",
]


def main():
    df = pd.read_csv(CLEANED_PATH)

    sample = df.iloc[0][FEATURE_NAMES].to_dict()

    response = requests.post(API_URL, json=sample)

    print("=== TEST /predict từ cleaned_dataset.csv (dữ liệu thô) ===")
    print("Input sample index: 0")
    print("Payload:")
    print(sample)
    print("\nResponse:")
    print(response.json())


if __name__ == "__main__":
    main()