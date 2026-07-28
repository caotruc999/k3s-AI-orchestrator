import os
import joblib
import pandas as pd

from decision_policy import DecisionPolicy

# Đường dẫn tương đối theo vị trí file này, chạy được trên mọi máy sau khi
# git pull (trước đây dùng path cứng "d:\Thuc_tap_tot_nghiep\..." chỉ chạy
# được trên đúng máy đã tạo ra, máy khác trong nhóm sẽ báo lỗi FileNotFoundError).
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(THIS_DIR)  # thư mục gốc project (k3s-AI-orchestrator)

MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.pkl")
INPUT_PATH = os.path.join(BASE_DIR, "dataset", "data", "output", "X_test.csv")


def main():
    model = joblib.load(MODEL_PATH)
    policy = DecisionPolicy()

    df = pd.read_csv(INPUT_PATH)

    sample = df.iloc[[0]]
    predicted_score = model.predict(sample)[0]
    decision = policy.decide(predicted_score)

    print("=== Predict And Decide ===")
    print("Input sample index : 0")
    print(f"Predicted score    : {predicted_score:.6f}")
    print(f"Scaling decision   : {decision}")


if __name__ == "__main__":
    main()