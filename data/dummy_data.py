import pandas as pd
import numpy as np
import itertools
import uuid
import random
from datetime import datetime, timedelta

# -----------------------------
# 설정
# -----------------------------
num_users_per_persona = 5   # 페르소나당 유저 수
num_days = 30               # 한 달치 시뮬레이션
max_sessions_per_day = 3    # 하루 최대 세션 수

# 페르소나 정의
chronotypes = ['아침형', '중간형', '저녁형']
caffeine_sensitivity = ['high', 'medium', 'low']
noise_sensitivity = ['high', 'medium', 'low']

# -----------------------------
# 타임 버킷 및 가중치 정의
# -----------------------------
time_buckets = [
    ('아침', 5, 10),
    ('오전', 10, 12),
    ('오후', 12, 17),
    ('저녁', 17, 22),
    ('밤', 22, 5)  
]

time_map = {
    '아침형': {'아침': 1.2, '오전': 0.6, '오후': -0.2, '저녁': -0.8, '밤': -1.5},
    '중간형': {'아침': 0.3, '오전': 0.8, '오후': 0.5, '저녁': -0.2, '밤': -0.8},
    '저녁형': {'아침': -1.2, '오전': -0.5, '오후': 0.2, '저녁': 0.8, '밤': 1.5}
}

# -----------------------------
# 페르소나 조합 생성
# -----------------------------
personas = list(itertools.product(chronotypes, caffeine_sensitivity, noise_sensitivity))

# -----------------------------
# 데이터 생성
# -----------------------------
all_data = []

for persona_id, (chronotype, caffeine_sens, noise_sens) in enumerate(personas):
    for u in range(num_users_per_persona):
        user_id = str(uuid.uuid4())
        start_date = datetime(2026, 1, 1)
        
        for day in range(num_days):
            date = start_date + timedelta(days=day)
            
            # 하루 1회 수면 만족도
            sleep_feeling = np.clip(np.random.normal(3.5, 0.7), 1, 5)
            
            # 하루 세션 수 랜덤
            num_sessions = np.random.randint(1, max_sessions_per_day+1)
            
            for s in range(num_sessions):
                session_id = str(uuid.uuid4())
                
                # session_start_hour 랜덤 0~23
                session_start_hour = np.random.randint(0, 24)
                
                # session_time_bucket 결정
                for bucket_name, start_hour, end_hour in time_buckets:
                    if start_hour < end_hour:
                        if start_hour <= session_start_hour < end_hour:
                            session_bucket = bucket_name
                            break
                    else:  # 밤 22~5
                        if session_start_hour >= start_hour or session_start_hour < end_hour:
                            session_bucket = bucket_name
                            break
                
                # session_date: 새벽 0~4시는 전날로
                session_date = date
                if session_start_hour < 5:
                    session_date = date - timedelta(days=1)
                
                # session_hours 랜덤 0.5~3시간
                session_hours = np.round(np.random.uniform(0.5, 3), 1)
                
                # 세션 상태 랜덤
                fatigue_level = np.clip(np.random.randint(1, 6), 1, 5)
                caffeine_level = np.clip(np.random.randint(1, 6), 1, 5)
                noise_level = np.clip(np.random.randint(1, 6), 1, 5)
                optimal_hours = random.choice([1, 1.5, 2, 3])
                
                # sleep_fatigue_gap
                sleep_fatigue_gap = np.round(sleep_feeling - fatigue_level, 2)
                
                # -------------------------------
                # focus_rating 계산
                # -------------------------------
                focus = 5.0
                
                # (1) 피로도 감점
                fatigue_map = {1:0, 2:-0.1, 3:-0.3, 4:-0.5, 5:-0.7}
                focus += fatigue_map[fatigue_level]

                # (2) 카페인 가점                
                if caffeine_sens == 'high':
                    focus += 0.5 * caffeine_level
                elif caffeine_sens == 'medium':
                    focus += 0.2 * caffeine_level
                else:
                    focus += -0.1 * caffeine_level
                
                # (3) 소음 감점
                if noise_sens == 'high':
                    focus += -0.5 * noise_level
                elif noise_sens == 'medium':
                    focus += -0.25 * noise_level
                else:
                    focus += -0.1 * noise_level
                
                # (4) 시간대 가중치
                focus += time_map[chronotype][session_bucket]

                # (5) 수면-피로 보정 (gap)
                focus += (sleep_fatigue_gap * 0.1)

                # (6) 세션 길이 보너스
                session_diff = abs(session_hours - optimal_hours)
                session_bonus = max(0, 0.5 - session_diff * 0.3)
                focus += session_bonus
                # 최종 클리핑 및 미세한 노이즈 추가 (현실성)
                focus = np.clip(focus + np.random.normal(0, 0.1), 1, 5)

                all_data.append({
                    'user_id': user_id,
                    'chronotype': chronotype,
                    'caffeine_sensitivity_prior': caffeine_sens,
                    'noise_senserance_prior': noise_sens,
                    'date': session_date.strftime('%Y-%m-%d'),
                    'sleep_feeling': round(sleep_feeling,2),
                    'session_id': session_id,
                    'session_start_hour': session_start_hour,
                    'session_time_bucket': session_bucket,
                    'session_hours': session_hours,
                    'optimal_hours': optimal_hours,
                    'session_length_bonus': round(session_bonus,2),
                    'fatigue_level': fatigue_level,
                    'caffeine_level': caffeine_level,
                    'noise_level': noise_level,
                    'sleep_fatigue_gap': sleep_fatigue_gap,
                    'focus_rating': round(focus,2)
                })

# -----------------------------
# 데이터프레임 생성 및 저장
# -----------------------------
df = pd.DataFrame(all_data)
df.to_csv('dummy_data_1.csv', index=False)
print("한 달치 더미 데이터 생성 완료! Rows:", len(df))
