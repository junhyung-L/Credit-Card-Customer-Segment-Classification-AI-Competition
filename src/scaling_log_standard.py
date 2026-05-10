import pandas as pd
import numpy as np

train_df=pd.read_csv('train_cat_plus.csv')

train_df = train_df.drop(columns=[col for col in train_df.columns if col.startswith('Unnamed')])

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def Scaler_data(df, save_path=None, batch_size=None):
    """
    수치형 컬럼에 대해 로그 변환, 스케일링, 이상치(Inf 처리)까지 수행하는 전처리 함수.
    결측치가 없다고 가정.

    Parameters:
    - df: 입력 데이터프레임
    - save_path: 저장 경로 지정 시, 전처리 완료 파일 CSV로 저장 (기본값: None)
    - batch_size: 대용량 데이터 처리 시 배치 크기 (기본값: None)

    Returns:
    - 전처리 완료된 데이터프레임
    - 로그 변환 적용된 컬럼 리스트
    - 음수값이 존재한 컬럼 리스트
    """
    
    # 1. 데이터 복사 (최소화)
    df_processed = df.copy()  # 최초 복사본 하나만 생성
    print(f"✅ 데이터 크기: {df_processed.shape}")

    # 2. 수치형 컬럼 선택
    numeric_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
    if not numeric_cols:
        print("⚠️ 수치형 컬럼이 없습니다.")
        return df_processed, [], []

    # 3. Inf 값 처리
    df_processed[numeric_cols] = df_processed[numeric_cols].replace([np.inf, -np.inf], np.nan)
    if df_processed[numeric_cols].isna().any().any():
        print("⚠️ Inf 값을 NaN으로 대체 후, NaN 발생. 평균값으로 대체.")
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
    print(f"✅ Inf 처리 완료. 처리된 컬럼: {numeric_cols}")

    # 4. 로그 변환이 필요한 컬럼 탐지
    log_transform_cols = []
    cols_with_negative_values = []

    for col in numeric_cols:
        col_min = df_processed[col].min()
        col_max = df_processed[col].max()
        if col_min < 0:
            cols_with_negative_values.append(col)
        if col_max > 1000 or (col_min > 0 and col_max / col_min > 50):
            log_transform_cols.append(col)
    print(f"📊 로그 변환 대상 컬럼: {log_transform_cols}")
    print(f"📊 음수 값 포함 컬럼: {cols_with_negative_values}")

    # 5. 로그 변환 (음수 컬럼 제외)
    for col in log_transform_cols:
        if col not in cols_with_negative_values:
            df_processed[col] = np.log1p(df_processed[col].clip(lower=0))  # 음수 방지
            print(f"📊 로그 변환 적용: {col}")

    # 6. 스케일링 (StandardScaler)
    scaler = StandardScaler()
    if batch_size is None:
        # 기본 방식: 전체 데이터 스케일링
        df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
        print("✅ 전체 데이터 스케일링 완료")
    else:
        # 대용량 데이터: 배치 처리
        for start in range(0, len(df_processed), batch_size):
            end = min(start + batch_size, len(df_processed))
            df_processed.iloc[start:end, df_processed.columns.get_indexer(numeric_cols)] = \
                scaler.partial_fit(df_processed[numeric_cols].iloc[start:end]).transform(
                    df_processed[numeric_cols].iloc[start:end]
                )
            print(f"📈 배치 스케일링 완료: {start} ~ {end} 행")

    # 7. 저장 (옵션)
    if save_path is not None:
        df_processed.to_csv(save_path, index=False)
        print(f"✅ 전처리 완료된 데이터가 저장되었습니다: {save_path}")

    return df_processed, log_transform_cols, cols_with_negative_values

# 함수 호출
train_df = Scaler_data(train_df, save_path='train_cat_sacled')

# 결과 확인
print(f"로그 변환된 컬럼 수: {len(log_cols)}개")
print(f"음수값 존재 컬럼 수: {len(negative_cols)}개")

'''
1. 로그

