import joblib
import pandas as pd

from decision_policy import DecisionPolicy

MODEL_PATH = r"d:\Thuc_tap_tot_nghiep\k3s-AI-orchestrator\models\linear_regression_model.pkl"
INPUT_PATH = r"d:\Thuc_tap_tot_nghiep\k3s-AI-orchestrator\dataset\data\output\X_test.csv"


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