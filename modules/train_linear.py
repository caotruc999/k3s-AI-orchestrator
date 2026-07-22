import os
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN ĐỘNG (PATH CONFIG)
# ==========================================
# Vị trí file train_linear.py (trong /modules)
MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
# Thư mục gốc project (k3s-AI-orchestrator)
BASE_DIR = os.path.dirname(MODULES_DIR)

# Thư mục chứa dữ liệu đầu vào
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "dataset", "data", "output")
# Thư mục chứa model lưu ra
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True) # Tự tạo folder models nếu chưa có

# ==========================================
# 2. ĐỌC DỮ LIỆU
# ==========================================
X_train = pd.read_csv(os.path.join(DATA_OUTPUT_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_OUTPUT_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_OUTPUT_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_OUTPUT_DIR, "y_test.csv")).squeeze()

# ==========================================
# 3. HUẤN LUYỆN MODEL & ĐÁNH GIÁ
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("=== Linear Regression Evaluation ===")
print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R2   : {r2:.6f}")

coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_
}).sort_values(by="coefficient", key=abs, ascending=False)

print("\n=== Feature Coefficients ===")
print(coef_df)

# ==========================================
# 4. LƯU MODEL VÀ KẾT QUẢ ĐẦU RA
# ==========================================
# Lưu model .pkl vào thư mục models/
model_path = os.path.join(MODELS_DIR, "linear_regression_model.pkl")
joblib.dump(model, model_path)

# Lưu file hệ số & dự đoán vào thư mục dataset/data/output/
coef_path = os.path.join(DATA_OUTPUT_DIR, "linear_feature_coefficients.csv")
coef_df.to_csv(coef_path, index=False)

result_df = pd.DataFrame({
    "actual": y_test,
    "predicted": y_pred
})
pred_path = os.path.join(DATA_OUTPUT_DIR, "linear_predictions.csv")
result_df.to_csv(pred_path, index=False)

print(f"\nModel da duoc luu: {model_path}")
print(f"Da luu he so feature: {coef_path}")
print(f"Da luu ket qua du doan: {pred_path}")