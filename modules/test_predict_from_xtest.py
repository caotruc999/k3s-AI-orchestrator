import os
import pandas as pd
import requests

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULES_DIR)

X_TEST_PATH = os.path.join(BASE_DIR, "dataset", "data", "output", "X_test.csv")
API_URL = "http://127.0.0.1:5000/predict"


def main():
    df = pd.read_csv(X_TEST_PATH)

    sample = df.iloc[0].to_dict()

    response = requests.post(API_URL, json=sample)

    print("=== TEST /predict từ X_test.csv ===")
    print("Input sample index: 0")
    print("Payload:")
    print(sample)
    print("\nResponse:")
    print(response.json())


if __name__ == "__main__":
    main()