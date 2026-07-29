import os
import pandas as pd
import requests
from collections import Counter

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULES_DIR)

X_TEST_PATH = os.path.join(BASE_DIR, "dataset", "data", "output", "X_test.csv")
API_URL = "http://127.0.0.1:5000/predict"


def main():
    df = pd.read_csv(X_TEST_PATH)

    results = []
    decision_counter = Counter()

    sample_count = 10

    print("=== TEST BATCH /predict từ X_test.csv ===")

    for i in range(sample_count):
        sample = df.iloc[i].to_dict()
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