from pydantic import BaseModel
from typing import List, Optional

# 1. 예측(Predict)용: 유저 한 명의 상태를 받을 때
class UserProfile(BaseModel):
    chronotype: str                 # 아침형 / 중간형 / 저녁형
    caffeine_sensitivity_prior: str # high / medium / low
    noise_senserance_prior: str     # high / medium / low
    optimal_hours: float            # 랜덤 최적 시간

class RecentRecord(BaseModel):
    date: str              # 해당 날짜 (YYYY-MM-DD)
    sleep_feeling: float   # 오늘 수면 만족도 (1.0 ~ 5.0)
    fatigue_level: int     # 세션 시작 시 피로도

class PredictRequest(BaseModel):
    user_id: str
    user_profile: UserProfile
    recent_records: List[RecentRecord] # 최근 7일치 기록 리스트


# 2. 학습(Train)용: 전체 세션 이력을 받을 때
class TrainData(BaseModel):
    # 학습용 식별자
    user_id: str
    session_id: str
    date: str
    
    # 공통 변수
    chronotype: str
    caffeine_sensitivity_prior: str
    noise_senserance_prior: str
    
    # 일별/세션별 변수 (float 강조 반영)
    sleep_feeling: float
    session_start_hour: int
    session_time_bucket: str   # 아침/오전/오후/저녁/밤 (백엔드 변환)
    session_hours: float       # 2.4시간 등
    fatigue_level: int
    caffeine_level: int
    noise_level: int
    
    # Label 및 파생 변수
    focus_rating: float        # 최종 레이블 (1.0 ~ 5.0)
    sleep_fatigue_gap: float   # sleep_feeling - fatigue_level
    optimal_hours: float       # 최적 시간