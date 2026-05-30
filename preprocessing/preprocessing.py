import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

# =========================================================
# 1. 데이터 로드
# =========================================================
df = pd.read_csv("hotel_bookings.csv")
print("원본 shape:", df.shape)

# =========================================================
# 2. 이상 데이터 제거
# =========================================================
df = df[(df['adults'] + df['children'].fillna(0) + df['babies']) > 0]
df = df[(df['adr'] >= 0) & (df['adr'] <= 500)]
print("이상치 제거 후:", df.shape)

# =========================================================
# 3. 결측치 처리
# =========================================================
df['children'] = df['children'].fillna(0)
df['country']  = df['country'].fillna('Unknown')
print("\n결측치 처리 완료")

# =========================================================
# 4. 제거할 컬럼
# =========================================================
drop_cols = [
    'arrival_date_week_number',
    'arrival_date_day_of_month',
    'meal',
    'reservation_status',
    'reservation_status_date',
    'agent',
    'company',
    'distribution_channel',
    'previous_bookings_not_canceled',
    'booking_changes',
    'arrival_date_year',
    'deposit_type',
    'customer_type',
]
df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print("\n컬럼 제거 완료")
print(df.columns)

# =========================================================
# 5. arrival_date_month 숫자 변환
# =========================================================
month_map = {
    'January':1,  'February':2,  'March':3,    'April':4,
    'May':5,      'June':6,      'July':7,     'August':8,
    'September':9,'October':10,  'November':11,'December':12
}
df['arrival_date_month'] = df['arrival_date_month'].map(month_map)

# =========================================================
# 5-1. peak_season 생성
# =========================================================

peak_months = [4,5,6,7,8,9,10]

df['peak_season'] = (
    df['arrival_date_month'].isin(peak_months)
).astype(int)

print("\npeak_season 분포:")
print(df['peak_season'].value_counts())

# =========================================================
# 5-2. lead_time_group 생성
# =========================================================

def classify_lead_time(row):

    lt = row['lead_time']
    peak = row['peak_season']

    # 성수기
    if peak == 1:

        if lt <= 7:
            return 'Impulsive'

        elif lt <= 90:
            return 'Planned'

        else:
            return 'Long_plan'

    # 비성수기
    else:

        if lt <= 7:
            return 'Impulsive'

        elif lt <= 30:
            return 'Planned'

        else:
            return 'Long_plan'


df['lead_time_group'] = df.apply(
    classify_lead_time,
    axis=1
)

print("\nlead_time_group 분포:")
print(df['lead_time_group'].value_counts())

# =========================================================
# 6. country → is_domestic
# =========================================================
df['is_domestic'] = (df['country'] == 'PRT').astype(int)

# =========================================================
# 7. adults/children/babies → family_type
# =========================================================
def classify_family(row):
    has_child = (row['children'] > 0) or (row['babies'] > 0)
    if has_child:
        return 'Family'
    elif row['adults'] == 1:
        return 'Individual'
    elif row['adults'] == 2:
        return 'Couple'
    elif row['adults'] >= 3:
        return 'Friends'
    else:
        return 'Unknown'

df['family_type'] = df.apply(classify_family, axis=1)
print(df[df['family_type'] == 'Unknown'])

# =========================================================
# 8. waiting list 이진화
# =========================================================
df['has_waiting_list'] = (df['days_in_waiting_list'] > 0).astype(int)

# =========================================================
# 9. parking 이진화
# =========================================================
df['has_parking'] = (df['required_car_parking_spaces'] > 0).astype(int)

# =========================================================
# 10. room mismatch 생성
# =========================================================
df['room_mismatch'] = (
    df['reserved_room_type'] != df['assigned_room_type']
).astype(int)

# =========================================================
# 11. 변환 완료 원본 제거
# =========================================================
drop_after = [
    'country',
    'adults', 'children', 'babies',
    'days_in_waiting_list',
    'required_car_parking_spaces',
    'lead_time',
    'arrival_date_month',
]
df = df.drop(columns=drop_after)

# =========================================================
# 11-1. Sampling 전 저장
# =========================================================

df.to_csv(
    "hotel_bookings_pre_eda.csv",
    index=False
)

print("\nSampling 전 저장 완료 → hotel_bookings_pre_eda.csv")
print("Sampling 전 shape:", df.shape)

# =========================================================
# 12. Stratified Sampling (30,000 rows)
# =========================================================

from sklearn.model_selection import train_test_split

sample_df, _ = train_test_split(
    df,
    train_size=30000,
    stratify=df['lead_time_group'],
    random_state=42
)

df = sample_df.copy()

print("\nStratified Sampling 완료")
print(df.shape)

# =========================================================
# 13. EDA/시각화용 저장 (OHE·스케일링 전)
# =========================================================
df.to_csv(
    "hotel_bookings_sampling.csv",
    index=False
)
print("\nEDA용 저장 완료 → hotel_bookings_sampling.csv")
print("EDA용 shape:", df.shape)

# =========================================================
# 14. One-Hot Encoding
# =========================================================
ohe_cols = [
    'hotel',
    'market_segment',
    'family_type',
    'reserved_room_type',
    'assigned_room_type',
    'lead_time_group',
]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# =========================================================
# 15. Scaling (lead_time 제외)
# =========================================================
scale_cols = [
    'adr',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
]
scaler = RobustScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])
print("\n스케일링 완료")

# =========================================================
# 16. 최종 확인
# =========================================================
print("\n최종 shape:", df.shape)
print("\n결측치:", df.isnull().sum().sum())
print("\n타깃 분포:")
print(df['is_canceled'].value_counts())
print("\n최종 컬럼:")
print(df.columns)

# =========================================================
# 17. 모델링용 저장 (OHE·스케일링 후)
# =========================================================
df.to_csv(
    "hotel_bookings_preprocessed.csv",
    index=False
)
print("\n모델링용 저장 완료 → hotel_bookings_preprocessed.csv")
