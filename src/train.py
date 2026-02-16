import pandas as pd
import numpy as np
import joblib 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# models 폴더가 없으면 자동으로 생성
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# -----------------------------
# 1. 데이터 불러오기 
# -----------------------------
DATA_PATH = os.path.join(BASE_DIR, 'data', 'dummy_data_1.csv')
df = pd.read_csv(DATA_PATH)

# -----------------------------
# 2. feature / label 분리
# -----------------------------
X = df.drop(columns=['user_id', 'session_id', 'focus_rating', 'date'])
y = df['focus_rating']

# 범주형 변수 one-hot 인코딩
X = pd.get_dummies(X, columns=['chronotype', 'caffeine_sensitivity_prior', 'noise_senserance_prior', 'session_time_bucket'])

# -----------------------------
# 3. 학습/검증 데이터 분리
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 4. 모델 학습
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. 평가
# -----------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.3f}")
print(f"Test R^2: {r2:.3f}")

# ----------------------------
# 6. 저장
# ----------------------------
# 학습 시 사용된 컬럼 순서 저장 (예측 시 시나리오 데이터의 형식을 맞추기 위함)
# train.py 저장 부분
joblib.dump(model, os.path.join(MODEL_DIR, 'focus_model.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'model_columns.pkl'))
print("모델 저장 완료!")

