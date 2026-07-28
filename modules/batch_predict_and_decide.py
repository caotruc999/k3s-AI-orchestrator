import os
import joblib
import pandas as pd

from decision_policy import DecisionPolicy

# Đường dẫn tương đối - lý do sửa giống predict_and_decide.py (xem comment ở đó).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.pkl")
INPUT_PATH = os.path.join(BASE_DIR, "dataset", "data", "output", "X_test.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "data", "output", "decision_results.csv")


def main():
    model = joblib.load(MODEL_PATH)
    policy = DecisionPolicy()

    df = pd.read_csv(INPUT_PATH)

    predicted_scores = model.predict(df)
    decisions = [policy.decide(score) for score in predicted_scores]

    result_df = df.copy()
    result_df["predicted_score"] = predicted_scores
    result_df["scaling_decision"] = decisions

    result_df.to_csv(OUTPUT_PATH, index=False)

    print("=== Batch Predict And Decide ===")
    print(f"So mau du doan      : {len(result_df)}")
    print(f"Da luu ket qua tai  : {OUTPUT_PATH}")
    print("\nPhan bo quyet dinh:")
    print(result_df["scaling_decision"].value_counts())


if __name__ == "__main__":
    main()