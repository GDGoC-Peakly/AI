from fastapi import FastAPI, HTTPException
import os
from typing import List, Dict
from contextlib import asynccontextmanager
import joblib

from src.train import train
from src.predict import predict_with_blended_logic

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 서버 메모리에 모델 저장
ml_context = {
    "model": None,
    "columns": None
}

def load_models_to_memory():
    """파일로부터 모델을 읽어 메모리에 올림"""
    try:
        model_path = os.path.join(MODEL_DIR, 'focus_model.pkl')
        columns_path = os.path.join(MODEL_DIR, 'model_columns.pkl')
        if os.path.exists(model_path):
            ml_context["model"] = joblib.load(model_path)
            ml_context["columns"] = joblib.load(columns_path)
            print("최신 모델 로드 완료")
    except Exception as e:
        print(f"로드 실패: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models_to_memory() # 시작시 로드
    yield
    ml_context.clear()

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------
# 1. 학습 API (새벽 4시 - 학습 시간을 고려해 4시에 학습 시작)
# ---------------------------------------------------------
@app.post("/train", summary="피크타임 학습", description="유저의 데이터들을 학습합니다.")
async def train_api(payload: List[Dict]):
    
    # 아예 데이터가 들어오지 않은 오류
    if not payload:
        raise HTTPException(status_code=400, detail="데이터가 비어있습니다.")
    
    new_model, new_cols = train(payload)
    
    if new_model:
        ml_context["model"] = new_model
        ml_context["columns"] = new_cols
        return {"status": "success", "message": "Model updated"}
    
    # 데이터 부족으로 학습 스킵 (백엔드가 잘 줬지만, 10개 이하라 None 반환된 경우)
    if new_model is None:
        return {
            "status": "skipped", 
            "message": "학습할 충분한 데이터가 없어 학습을 건너 뜁니다. 최소 10개의 데이터를 넣어주세요."
        }
    
    # 정말 알 수 없는 그 이외의 오류
    raise HTTPException(status_code=500, detail="Training failed")

# ---------------------------------------------------------
# 2. 예측 API (새벽 5시 - 하루의 시작으로 삼아 예측 후 추천을 내뱉음)
# ---------------------------------------------------------
@app.post("/predict", summary="피크타임 예측", description="유저의 최근 7일 기록을 바탕으로 집중력 피크타임 Top 3를 반환합니다.")
async def predict_api(user_data: Dict):
    if not ml_context["model"]:
        raise HTTPException(status_code=500, detail="모델을 찾을 수 없습니다.")
    
    # 최근 기록이 없는 경우 방어
    recent_records = user_data.get('recent_records', [])
    
    try:
        top_3, s_avg, f_avg = predict_with_blended_logic(
            ml_context["model"], 
            ml_context["columns"],
            user_data['user_profile'],
            recent_records  # 데이터가 없어도 predict 함수 내부의 3.5점 로직이 방어
        )
        return {
            "user_id": user_data.get("user_id"),
            "top_peak_times": [
                {
                    "hour": item['start_hour'], 
                    "duration": item['duration'], 
                    "score": round(item['score'], 2)
                } for item in top_3
            ]
        }
    except Exception as e:
        # 정말 알 수 없는 그 이외의 오류
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="예측 과정에서 오류가 발생했습니다.")
    