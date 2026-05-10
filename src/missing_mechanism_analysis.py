pip install xgboost

import pandas as pd
import numpy as np
import gc
import os
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from google.colab import drive
drive.mount('/content/drive')
train_df=pd.read_csv('/content/drive/MyDrive/신용카드고객/train_df.csv')
test_df=pd.read_csv('/content/drive/MyDrive/신용카드고객/test_df.csv')

train_df
train_df1=train_df.copy()

train_df

import pandas as pd
total_rows = 2400000

# 1. 결측치 개수 계산
missing_counts = train_df.isnull().sum()

# 2. 결측치 비율 계산 (%)
missing_ratios = (missing_counts / total_rows) * 100

# 3. 결측치가 있는 변수만 필터링
missing_train_df = pd.DataFrame({
    '결측치 개수': missing_counts,
    '결측 비율 (%)': missing_ratios
})
missing_train_df = missing_train_df[missing_train_df['결측치 개수'] > 0]  # 결측치가 0인 변수 제외

# 4. 결과 정렬 (결측 비율 높은 순)
missing_train_df = missing_train_df.sort_values(by='결측 비율 (%)', ascending=False)

# 5. 출력
print(f"총 데이터 수: {total_rows}")
print("\n결측치가 있는 변수 목록:")
print(missing_train_df)


# 결측치를 '없음'으로 채우기
industry_list = [
    '_3순위여유업종',
    '_3순위납부업종',
    '_2순위여유업종',
    '_3순위교통업종',
    '_2순위납부업종',
    '_1순위여유업종',
    '_2순위교통업종',
    '_3순위쇼핑업종',
    '_1순위납부업종',
    '_1순위교통업종',
    '_2순위쇼핑업종',
    '_3순위업종',
    '_1순위쇼핑업종',
    '_2순위업종',
    '_1순위업종'
]

# 각 업종 목록의 결측치를 '없음'으로 채우기
for industry in industry_list:
    train_df1[industry].fillna('없음', inplace=True)



import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np

# 데이터 로드 (주어진 데이터는 일부이므로 전체 데이터가 있다고 가정)
data = train_df1  # 실제 파일 경로로 수정 필요

# 'ID'와 'Segment' 열 제거
data = data.drop(columns=['ID'])

# 결측치 비율이 제공된 변수 리스트
missing_vars = [
    '연체일자_B0M', '최종카드론_대출일자', '최종카드론_신청경로코드', '최종카드론_금융상환방식코드',
    'RV신청일자', 'OS구분코드', '_2순위신용체크구분', '혜택수혜율_B0M', '최종유효년월_신용_이용',
    '혜택수혜율_R3M', '가입통신회사코드', '직장시도명', '최종유효년월_신용_이용가능',
    '최종카드발급일자', 'RV전환가능여부', '_1순위신용체크구분'
]

# 결측치 비율 딕셔너리
missing_rates = {
    '연체일자_B0M': 99.764000, '최종카드론_대출일자': 82.847083, '최종카드론_신청경로코드': 81.592750,
    '최종카드론_금융상환방식코드': 81.588583, 'RV신청일자': 81.301500, 'OS구분코드': 68.065250,
    '_2순위신용체크구분': 39.921458, '혜택수혜율_B0M': 23.146750, '최종유효년월_신용_이용': 22.259625,
    '혜택수혜율_R3M': 20.364417, '가입통신회사코드': 16.148750, '직장시도명': 10.207042,
    '최종유효년월_신용_이용가능': 8.768625, '최종카드발급일자': 1.748542, 'RV전환가능여부': 1.228042,
    '_1순위신용체크구분': 1.164583
}

# 1. 결측치 시각화
msno.matrix(data[missing_vars])
plt.title("Missing Data Matrix")
plt.show()

# 2. 결측치 패턴 분석 (상관관계 확인)
msno.heatmap(data[missing_vars])
plt.title("Missing Data Correlation Heatmap")
plt.show()




import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import numpy as np
import seaborn as sns


# 결측치 비율이 제공된 변수 리스트
missing_vars = [
    '연체일자_B0M', '최종카드론_대출일자', '최종카드론_신청경로코드', '최종카드론_금융상환방식코드',
    'RV신청일자', 'OS구분코드', '_2순위신용체크구분', '혜택수혜율_B0M', '최종유효년월_신용_이용',
    '혜택수혜율_R3M', '가입통신회사코드', '직장시도명', '최종유효년월_신용_이용가능',
    '최종카드발급일자', 'RV전환가능여부', '_1순위신용체크구분'
]

# 결측치 비율 딕셔너리
missing_rates = {
    '연체일자_B0M': 99.764000, '최종카드론_대출일자': 82.847083, '최종카드론_신청경로코드': 81.592750,
    '최종카드론_금융상환방식코드': 81.588583, 'RV신청일자': 81.301500, 'OS구분코드': 68.065250,
    '_2순위신용체크구분': 39.921458, '혜택수혜율_B0M': 23.146750, '최종유효년월_신용_이용': 22.259625,
    '혜택수혜율_R3M': 20.364417, '가입통신회사코드': 16.148750, '직장시도명': 10.207042,
    '최종유효년월_신용_이용가능': 8.768625, '최종카드발급일자': 1.748542, 'RV전환가능여부': 1.228042,
    '_1순위신용체크구분': 1.164583
}

# 1. 결측치 간 상관관계 직접 계산
# 결측 여부를 나타내는 더미 변수 생성 (0: 결측 아님, 1: 결측)
missing_dummy_df = pd.DataFrame()
for var in missing_vars:
    if var in data.columns:
        missing_dummy_df[f'{var}_missing'] = data[var].isna().astype(int)

# 더미 변수들 간의 상관계수 계산
corr_matrix = missing_dummy_df.corr()

# 상관계수를 딕셔너리로 변환
correlation_pairs = {}
for i, var1 in enumerate(corr_matrix.index):
    for j, var2 in enumerate(corr_matrix.columns):
        if i < j:  # 중복 제거 (대칭 행렬이므로)
            corr_value = corr_matrix.iloc[i, j]
            if not np.isnan(corr_value):  # NaN 값 제외
                # 변수 이름에서 '_missing' 접미사 제거
                var1_clean = var1.replace('_missing', '')
                var2_clean = var2.replace('_missing', '')
                correlation_pairs[(var1_clean, var2_clean)] = corr_value

# 2. MCAR 테스트 (범주형 변수에 대해 카이제곱 검정)
def check_mcar_categorical(data, var, group_var):
    if var in data.columns and group_var in data.columns:
        data[f'{var}_missing'] = data[var].isna().astype(int)
        contingency_table = pd.crosstab(data[group_var], data[f'{var}_missing'])
        chi2, p, dof, ex = chi2_contingency(contingency_table)
        return p
    return None

# 3. 상관관계 기반으로 MNAR 판단 보조 함수
def get_max_correlation(var):
    max_corr = 0
    for (var1, var2), corr in correlation_pairs.items():
        if var1 == var or var2 == var:
            max_corr = max(max_corr, abs(corr))  # 절대값 사용
    return max_corr

# 4. 결측치 메커니즘 분석
def analyze_missing_mechanism(data, var, missing_rate):
    print(f"\n{var}: 결측률 {missing_rate}%")

    # 히트맵에서 확인된 최대 상관계수
    max_corr = get_max_correlation(var)
    print(f" - 히트맵에서 확인된 최대 상관계수: {max_corr:.2f}")

    # 결측률과 상관계수를 기준으로 판단
    if missing_rate > 80:
        print(" - 결측률이 매우 높음 (>80%). MNAR 가능성 있음 (데이터 생성 과정에서 발생하는 구조적 결측)")
    elif missing_rate > 30:
        if max_corr > 0.7:
            print(" - 결측률이 높고 상관계수가 높음 (30~80%, 상관계수 > 0.7). MNAR 가능성")
        else:
            print(" - 결측률이 높음 (30~80%). MAR 가능성 (다른 변수와 약한 연관성)")
    else:
        if max_corr > 0.7:
            print(" - 결측률이 낮으나 상관계수가 높음 (<30%, 상관계수 > 0.7). MNAR 가능성")
        else:
            print(" - 결측률이 낮음 (<30%). MCAR 또는 MAR 가능성")

    # 숫자형 변수와의 상관관계
    if var in data.columns:
        missing_dummy = data[var].isna().astype(int)
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_with_numeric = data[numeric_cols].corrwith(missing_dummy).dropna().sort_values(ascending=False)
        print(f" - 숫자형 변수와 결측 여부의 상관관계:\n{corr_with_numeric}")

        # 범주형 변수와의 연관성
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        print(f" - 범주형 변수와 결측 여부의 연관성 (카이제곱 p-value):")
        for cat_var in categorical_cols:
            if cat_var in data.columns:
                p_value = check_mcar_categorical(data, var, cat_var)
                if p_value is not None:
                    print(f"   {cat_var}: {p_value:.4f}")

# 각 변수에 대해 결측치 메커니즘 분석
for variable, rate in missing_rates.items():
    analyze_missing_mechanism(data, variable, rate)

# 5. 결론 출력 및 Segment 분류를 위한 제안
print("\n=== 결론 ===")
for var, rate in missing_rates.items():
    if var in data.columns:
        # Segment를 기준으로 카이제곱 검정
        p_value_segment = check_mcar_categorical(data, var, 'Segment')

        # 다른 범주형 변수들과의 연관성도 확인 (참고용)
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        p_values_other = {}
        for cat_var in categorical_cols:
            if cat_var != 'Segment':
                p_value = check_mcar_categorical(data, var, cat_var)
                if p_value is not None:
                    p_values_other[cat_var] = p_value

        max_corr = get_max_correlation(var)
        print(f"{var} (결측률: {rate}%):")
        if rate > 80 or max_corr > 0.7:
            print(" - MNAR로 판단 (결측률 > 80% 또는 상관계수 > 0.7)")
            print("   - Segment 분류 영향: Segment와의 연관성 확인 필요")
            if p_value_segment is not None:
                print(f"   - Segment와의 카이제곱 p-value: {p_value_segment:.4f}")
                if p_value_segment <= 0.05:
                    print("     - Segment와 유의미한 연관성 있음. 결측치가 Segment 분류에 영향을 줄 가능성 높음")
                else:
                    print("     - Segment와 유의미한 연관성 없음. 결측치가 Segment 분류에 큰 영향을 주지 않을 가능성")
        elif rate > 30 and p_value_segment is not None and p_value_segment <= 0.05:
            print(" - MAR로 판단 (결측률 30~80%, Segment와의 p-value <= 0.05)")
            print("   - Segment 분류 영향: Segment와 연관성 있음. 결측치가 Segment 분류에 영향을 줄 가능성")
        elif rate < 30 and (p_value_segment is None or p_value_segment > 0.05):
            print(" - MCAR로 판단 (결측률 < 30%, Segment와의 p-value > 0.05)")
            print("   - Segment 분류 영향: Segment와 연관성 낮음. 결측치가 Segment 분류에 큰 영향을 주지 않음")
        else:
            print(" - MAR로 판단 가능 (결측률과 Segment와의 p-value 기준)")
            print("   - Segment 분류 영향: Segment와 약한 연관성 가능")

        # 다른 범주형 변수와의 연관성 출력 (참고용)
        if p_values_other:
            print("   - 다른 범주형 변수와의 연관성 (참고):")
            for cat_var, p_val in p_values_other.items():
                print(f"     {cat_var}: p-value = {p_val:.4f}")


high_missing_cols = missing_ratios[missing_ratios >= 50].index.tolist()

train_df1 = train_df1.drop(columns=high_missing_cols)

