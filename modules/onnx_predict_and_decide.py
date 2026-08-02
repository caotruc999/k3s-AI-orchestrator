import os
import numpy as np
import pandas as pd
import onnxruntime as ort

from decision_policy import DecisionPolicy

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULES_DIR)

ONNX_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.onnx")
INPUT_PATH = os.path.join(BASE_DIR, "dataset", "data", "output", "X_test.csv")


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