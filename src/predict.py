import pandas as pd
import joblib
import os

# 현재 파일의 위치를 기준으로 모델 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 모델 로드
model_path = os.path.join(MODEL_DIR, 'focus_model.pkl')
columns_path = os.path.join(MODEL_DIR, 'model_columns.pkl')

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

def get_weighted_value(user_avg, system_default, weight=0.3):
    """
    시스템 고정값과 유저 평균을 블렌딩하는 함수
    유저 평균 데이터가 없으면 시스템 고정값 반환
    """
    if user_avg is None:
        return system_default
    return (user_avg * weight) + (system_default * (1 - weight))

def get_weighted_user_avg(user_7day_avg, user_2day_avg, weight=0.3):
    """
    최근 7일 평균과 최근 2일 평균 블렌딩
    """
    if user_7day_avg is None:
        return user_2day_avg if user_2day_avg is not None else 3.5  # 기본값
    if user_2day_avg is None:
        return user_7day_avg
    return (user_7day_avg * (1 - weight)) + (user_2day_avg * weight)

def predict_with_blended_logic(user_profile, user_7day_avg_sleep=None, user_2day_avg_sleep=None,
                               user_7day_avg_fatigue=None, user_2day_avg_fatigue=None):
    # 시스템 고정값 정의
    SYS_SLEEP = 3.5
    SYS_FATIGUE = 2.0
    
    # 전략 채택: (유저의 7일 평균 * 0.7) + (유저의 2일 평균 * 0.3)
    final_sleep = get_weighted_user_avg(user_7day_avg_sleep, user_2day_avg_sleep, weight=0.3)
    final_fatigue = get_weighted_user_avg(user_7day_avg_fatigue, user_2day_avg_fatigue, weight=0.3)
    final_gap = final_sleep - final_fatigue

    
    results = []
    simulation_hours = list(range(5, 24)) + list(range(0, 5))

    optimal_hours = user_profile.get('optimal_hours', 1.5)  # 기본 1.5시간

    for hour in simulation_hours:
        
        if 5 <= hour < 10:
            bucket = '아침'
        elif 10 <= hour < 12:
            bucket = '오전'
        elif 12 <= hour < 17:
            bucket = '오후'
        elif 17 <= hour < 22:
            bucket = '저녁'
        else:
            bucket = '밤'

        scenario = {
            'sleep_feeling': final_sleep,
            'fatigue_level': final_fatigue,
            'sleep_fatigue_gap': final_gap,
            'session_start_hour': hour,
            'session_hours': 1.5,
            'caffeine_level': 1,
            'noise_level': 1,
            f"chronotype_{user_profile['chronotype']}": 1,
            f"caffeine_sensitivity_prior_{user_profile['caffeine_sens']}": 1,
            f"noise_senserance_prior_{user_profile['noise_sens']}": 1,
            f"session_time_bucket_{bucket}": 1
        }
        
        scenario_df = pd.DataFrame([scenario]).reindex(columns=model_columns, fill_value=0)
        rating = model.predict(scenario_df)[0]
        
        # -----------------------------
        # 후처리: 세션 길이 보너스 적용
        # -----------------------------
        session_hours = 1.5
        session_diff = abs(session_hours - optimal_hours)
        session_bonus = max(0, 0.5 - session_diff * 0.3)
        rating += session_bonus
        results.append((hour, rating))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3], final_sleep, final_fatigue

# --- 테스트 실행 ---
user_info = {
    'chronotype': '저녁형',
    'caffeine_sens': 'high',
    'noise_sens': 'high',
    'optimal_hours': 2  # 유저별 최적 세션 시간
}
# 만약 유저의 최근 7일이 최악이었다면 (수면 1.0, 피로 5.0 가정)
best_times, s_val, f_val = predict_with_blended_logic(
    user_info,
    user_7day_avg_sleep=1.0,
    user_2day_avg_sleep=2.0,
    user_7day_avg_fatigue=5.0,
    user_2day_avg_fatigue=4.0
)

print(f"추천 시간 1위: {best_times[0][0]}시 (점수: {best_times[0][1]:.2f})")