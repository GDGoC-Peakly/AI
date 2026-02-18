import pandas as pd
import joblib
import os
import random
import numpy as np


# 현재 파일의 위치를 기준으로 모델 폴더 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

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

def predict_with_blended_logic(model, model_columns, user_profile, recent_records):
    """
    recent_records: 백엔드에서 넘겨준 최근 7일치 기록 리스트
    [{"sleep_feeling": 4, "fatigue_level": 2}, ...] 형태
    """
    # 원본 데이터를 데이터프레임으로 변환
    df_recent = pd.DataFrame(recent_records)
    
    # 최근 7일 평균 (전체 평균)
    user_7day_avg_sleep = df_recent['sleep_feeling'].mean()
    user_7day_avg_fatigue = df_recent['fatigue_level'].mean()
    
    # 최근 2일 평균 (상위 2개 행만 슬라이싱해서 평균)
    user_2day_avg_sleep = df_recent.head(2)['sleep_feeling'].mean()
    user_2day_avg_fatigue = df_recent.head(2)['fatigue_level'].mean()
    
    # 전략 채택: (유저의 7일 평균 * 0.7) + (유저의 2일 평균 * 0.3)
    final_sleep = get_weighted_user_avg(user_7day_avg_sleep, user_2day_avg_sleep, weight=0.3)
    final_fatigue = get_weighted_user_avg(user_7day_avg_fatigue, user_2day_avg_fatigue, weight=0.3)
    final_gap = final_sleep - final_fatigue
    
    simulation_hours = np.arange(0, 24, 0.5).tolist()
    simulation_hours = [h for h in simulation_hours if h >= 5] + [h for h in simulation_hours if h < 5]
            
    # optimal_hours 랜덤 생성
    optimal_hours = user_profile.get('optimal_hours')
    if optimal_hours is None:
        optimal_hours = round(random.uniform(1.5, 2.5), 1) 

    # 피로도에 따른 검토 세션 조정
    if final_fatigue >= 4.0:
        recommended_duration_range = [1.0, 1.5]  # 컨디션 안 좋을 땐 긴 시간은 아예 후보에서 제외
    elif final_fatigue >= 3.5:
        recommended_duration_range = [1.5, 2.0, 2.5]
    else:
        recommended_duration_range = [2.0, 2.5, 3.0] # 컨디션 좋으면 긴 시간도 검토
    
    results = []

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

        for duration in recommended_duration_range:
            scenario = {
                'sleep_feeling': final_sleep,
                'fatigue_level': final_fatigue,
                'sleep_fatigue_gap': final_gap,
                'session_start_hour': hour,
                'session_hours': duration,
                'caffeine_level': 1,
                'noise_level': 1,
                f"chronotype_{user_profile['chronotype']}": 1,
                f"caffeine_sensitivity_prior_{user_profile['caffeine_sensitivity_prior']}": 1,
                f"noise_senserance_prior_{user_profile['noise_senserance_prior']}": 1,
                f"session_time_bucket_{bucket}": 1
            }
        
            scenario_df = pd.DataFrame([scenario]).reindex(columns=model_columns, fill_value=0)
            rating = model.predict(scenario_df)[0]
        
            # -----------------------------
            # 후처리: 세션 길이 보너스 적용
            # -----------------------------
            session_hours = duration
            session_diff = abs(session_hours - optimal_hours)
            session_bonus = max(0, 0.05 - session_diff * 0.2)
            rating += session_bonus
            rating = min(rating, 5.0)
            results.append({
                    'start_hour': hour,
                    'duration': duration,
                    'score': round(float(rating), 3)
                })
            
    full_df = pd.DataFrame(results)
    
    # 
    # 같은 시작 시간(start_hour) 중에서 가장 score가 높은 행의 인덱스만 추출
    idx_max = full_df.groupby("start_hour")["score"].idxmax()
    final_candidates = full_df.loc[idx_max]

    # 전체 후보 중 최종 점수 순으로 정렬하여 TOP 3 리턴
    top3_results = (
        final_candidates
        .sort_values("score", ascending=False)
        .head(3)
        .to_dict('records')
    )
    
    return top3_results, final_sleep, final_fatigue

"""
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

# --- 테스트 실행 출력 부분 ---
print(f"추천 1위: {best_times[0]['start_hour']}시 시작, {best_times[0]['duration']}시간 집중 (점수: {best_times[0]['score']:.2f})")
print(f"추천 2위: {best_times[1]['start_hour']}시 시작, {best_times[1]['duration']}시간 집중 (점수: {best_times[1]['score']:.2f})")
"""