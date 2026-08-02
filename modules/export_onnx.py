import os
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(MODULES_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.pkl")
ONNX_PATH = os.path.join(BASE_DIR, "models", "linear_regression_model.onnx")

NUM_FEATURES = 10
TARGET_OPSET = 21


def main():
    model = joblib.load(MODEL_PATH)

    initial_type = [("float_input", FloatTensorType([None, NUM_FEATURES]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=TARGET_OPSET
    )

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("=== Export ONNX ===")
    print(f"Model PKL     : {MODEL_PATH}")
    print(f"Model ONNX    : {ONNX_PATH}")
    print(f"Target opset  : {TARGET_OPSET}")
    print("Export thanh cong.")


if __name__ == "__main__":
    main()