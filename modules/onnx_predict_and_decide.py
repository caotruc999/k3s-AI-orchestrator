import numpy as np
import pandas as pd
import onnxruntime as ort

from decision_policy import DecisionPolicy

ONNX_PATH = r"d:\Thuc_tap_tot_nghiep\k3s-AI-orchestrator\models\linear_regression_model.onnx"
INPUT_PATH = r"d:\Thuc_tap_tot_nghiep\k3s-AI-orchestrator\dataset\data\output\X_test.csv"


def main():
    session = ort.InferenceSession(ONNX_PATH)
    policy = DecisionPolicy()

    df = pd.read_csv(INPUT_PATH)
    sample = df.iloc[[0]].astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    predicted_score = session.run([output_name], {input_name: sample.to_numpy()})[0][0][0]
    decision = policy.decide(float(predicted_score))

    print("=== ONNX Predict And Decide ===")
    print("Input sample index : 0")
    print(f"Predicted score    : {predicted_score:.6f}")
    print(f"Scaling decision   : {decision}")


if __name__ == "__main__":
    main()