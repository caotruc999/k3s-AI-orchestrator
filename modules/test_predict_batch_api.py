import os
import sys
import pandas as pd
import requests
from collections import Counter

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

    results = []
    decision_counter = Counter()

    sample_count = 10

    print("=== TEST BATCH /predict từ cleaned_dataset.csv (dữ liệu thô) ===")

    for i in range(sample_count):
        sample = df.iloc[i][FEATURE_NAMES].to_dict()
        response = requests.post(API_URL, json=sample)
        result = response.json()

        results.append(result)
        decision_counter[result["scaling_decision"]] += 1

        print(f"\nSample {i}")
        print(f"Predicted score  : {result['predicted_score']}")
        print(f"Decision         : {result['scaling_decision']}")
        print(f"Action taken     : {result['action_taken']}")
        print(f"Replicas         : {result['replicas_before']} -> {result['replicas_after']}")
        print(f"Reason           : {result['reason']}")

    print("\n=== Decision Summary ===")
    for decision, count in decision_counter.items():
        print(f"{decision}: {count}")


if __name__ == "__main__":
    main()