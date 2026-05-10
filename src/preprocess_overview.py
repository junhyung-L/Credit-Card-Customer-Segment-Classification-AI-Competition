
import pandas as pd
import gc
import os

from google.colab import drive
drive.mount('/content/drive')


os.chdir('/content/drive/MyDrive/신용카드고객')

train_df1_splits = ["train", "test"]


train_df1_categories = {
    "회원정보": {"folder": "1.회원정보", "suffix": "회원정보", "var_prefix": "customer"},
    "신용정보": {"folder": "2.신용정보", "suffix": "신용정보", "var_prefix": "credit"},
    "승인매출정보": {"folder": "3.승인매출정보", "suffix": "승인매출정보", "var_prefix": "sales"},
    "청구정보": {"folder": "4.청구입금정보", "suffix": "청구정보", "var_prefix": "billing"},
    "잔액정보": {"folder": "5.잔액정보", "suffix": "잔액정보", "var_prefix": "balance"},
    "채널정보": {"folder": "6.채널정보", "suffix": "채널정보", "var_prefix": "channel"},
    "마케팅정보": {"folder": "7.마케팅정보", "suffix": "마케팅정보", "var_prefix": "marketing"},
    "성과정보": {"folder": "8.성과정보", "suffix": "성과정보", "var_prefix": "performance"}
}
months = ['07', '08', '09', '10', '11', '12']

for split in train_df1_splits:
    for category, info in train_df1_categories.items():
        folder = info["folder"]
        suffix = info["suffix"]
        var_prefix = info["var_prefix"]

        for month in months:
            # 파일명 형식: 2018{month}_{split}_{suffix}.parquet
            file_path = f"./{split}/{folder}/2018{month}_{split}_{suffix}.parquet"
            # 변수명 형식: {var_prefix}_{split}_{month}
            variable_name = f"{var_prefix}_{split}_{month}"
            globals()[variable_name] = pd.read_parquet(file_path)
            print(f"{variable_name} is loaded from {file_path}")

gc.collect()


# 데이터 유형별 설정
info_categories = ["customer", "credit", "sales", "billing", "balance", "channel", "marketing", "performance"]

# 월 설정
months = ['07', '08', '09', '10', '11', '12']

#### Train ####

# 각 유형별로 월별 데이터를 합쳐서 새로운 변수에 저장
train_dfs = {}

for prefix in info_categories:
    # globals()에서 동적 변수명으로 데이터프레임들을 가져와 리스트에 저장
    df_list = [globals()[f"{prefix}_train_{month}"] for month in months]
    train_dfs[f"{prefix}_train_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_train_df is created with shape: {train_dfs[f'{prefix}_train_df'].shape}")


customer_train_df = train_dfs["customer_train_df"]
credit_train_df   = train_dfs["credit_train_df"]
sales_train_df    = train_dfs["sales_train_df"]
billing_train_df  = train_dfs["billing_train_df"]
balance_train_df  = train_dfs["balance_train_df"]
channel_train_df  = train_dfs["channel_train_df"]
marketing_train_df= train_dfs["marketing_train_df"]
performance_train_df = train_dfs["performance_train_df"]

gc.collect()

#### Test ####

# test 데이터에 대해 train과 동일한 방법 적용
test_dfs = {}

for prefix in info_categories:
    df_list = [globals()[f"{prefix}_test_{month}"] for month in months]
    test_dfs[f"{prefix}_test_df"] = pd.concat(df_list, axis=0)
    gc.collect()
    print(f"{prefix}_test_df is created with shape: {test_dfs[f'{prefix}_test_df'].shape}")


customer_test_df = test_dfs["customer_test_df"]
credit_test_df   = test_dfs["credit_test_df"]
sales_test_df    = test_dfs["sales_test_df"]
billing_test_df  = test_dfs["billing_test_df"]
balance_test_df  = test_dfs["balance_test_df"]
channel_test_df  = test_dfs["channel_test_df"]
marketing_test_df= test_dfs["marketing_test_df"]
performance_test_df = test_dfs["performance_test_df"]

gc.collect()

#### Train ####

train_df = customer_train_df.merge(credit_train_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: train_step1, shape:", train_df.shape)
del customer_train_df, credit_train_df
gc.collect()

# 이후 merge할 데이터프레임 이름과 단계 정보를 리스트에 저장
merge_list = [
    ("sales_train_df",    "Step2"),
    ("billing_train_df",  "Step3"),
    ("balance_train_df",  "Step4"),
    ("channel_train_df",  "Step5"),
    ("marketing_train_df","Step6"),
    ("performance_train_df", "최종")
]

# 나머지 단계 merge
for df_name, step in merge_list:
    # globals()로 동적 변수 접근하여 merge 수행
    train_df = train_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: train_{step}, shape:", train_df.shape)
    # 사용한 변수는 메모리 해제를 위해 삭제
    del globals()[df_name]
    gc.collect()

#### Test ####

test_df = customer_test_df.merge(credit_test_df, on=['기준년월', 'ID'], how='left')
print("Step1 저장 완료: test_step1, shape:", test_df.shape)
del customer_test_df, credit_test_df
gc.collect()

# 이후 merge할 데이터프레임 이름과 단계 정보를 리스트에 저장
merge_list = [
    ("sales_test_df",    "Step2"),
    ("billing_test_df",  "Step3"),
    ("balance_test_df",  "Step4"),
    ("channel_test_df",  "Step5"),
    ("marketing_test_df","Step6"),
    ("performance_test_df", "최종")
]

# 나머지 단계 merge
for df_name, step in merge_list:
    # globals()로 동적 변수 접근하여 merge 수행
    test_df = test_df.merge(globals()[df_name], on=['기준년월', 'ID'], how='left')
    print(f"{step} 저장 완료: test_{step}, shape:", test_df.shape)
    # 사용한 변수는 메모리 해제를 위해 삭제
    del globals()[df_name]
    gc.collect()

import pandas as pd
import gc
import os

from google.colab import drive
drive.mount('/content/drive')
train_df=pd.read_csv('/content/drive/MyDrive/신용카드고객/train_df.csv')
test_df=pd.read_csv('/content/drive/MyDrive/신용카드고객/test_df.csv')

train_summation=train_df.head(1000)
train_summation.to_csv('/content/drive/MyDrive/신용카드고객/train_summation.csv')

train_df1=train_df.copy()
test_df1=test_df.copy()

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


print(len(train_df.loc[train_df['_3순위여유업종_이용금액'] == 0, '_3순위여유업종']))
print(len(train_df.loc[train_df['_3순위납부업종_이용금액'] == 0, '_3순위납부업종']))
print(len(train_df.loc[train_df['_2순위여유업종_이용금액'] == 0, '_2순위여유업종']))
print(len(train_df.loc[train_df['_3순위교통업종_이용금액'] == 0, '_3순위교통업종']))
print(len(train_df.loc[train_df['_2순위납부업종_이용금액'] == 0, '_2순위납부업종']))
print(len(train_df.loc[train_df['_1순위여유업종_이용금액'] == 0, '_1순위여유업종']))
print(len(train_df.loc[train_df['_2순위교통업종_이용금액'] == 0, '_2순위교통업종']))
print(len(train_df.loc[train_df['_3순위쇼핑업종_이용금액'] == 0, '_3순위쇼핑업종']))
print(len(train_df.loc[train_df['_1순위납부업종_이용금액'] == 0, '_1순위납부업종']))
print(len(train_df.loc[train_df['_1순위교통업종_이용금액'] == 0, '_1순위교통업종']))
print(len(train_df.loc[train_df['_2순위쇼핑업종_이용금액'] == 0, '_2순위쇼핑업종']))
print(len(train_df.loc[train_df['_3순위업종_이용금액'] == 0, '_3순위업종']))
print(len(train_df.loc[train_df['_1순위쇼핑업종_이용금액'] == 0, '_1순위쇼핑업종']))
print(len(train_df.loc[train_df['_2순위업종_이용금액'] == 0, '_2순위업종']))
print(len(train_df.loc[train_df['_1순위업종_이용금액'] == 0, '_1순위업종']))


print((train_df['_3순위여유업종_이용금액'] == 0).sum())
print((train_df['_3순위납부업종_이용금액'] == 0).sum())
print((train_df['_2순위여유업종_이용금액'] == 0).sum())
print((train_df['_3순위교통업종_이용금액'] == 0).sum())
print((train_df['_2순위납부업종_이용금액'] == 0).sum())
print((train_df['_1순위여유업종_이용금액'] == 0).sum())
print((train_df['_2순위교통업종_이용금액'] == 0).sum())
print((train_df['_3순위쇼핑업종_이용금액'] == 0).sum())
print((train_df['_1순위납부업종_이용금액'] == 0).sum())
print((train_df['_1순위교통업종_이용금액'] == 0).sum())
print((train_df['_2순위쇼핑업종_이용금액'] == 0).sum())
print((train_df['_3순위업종_이용금액'] == 0).sum())
print((train_df['_1순위쇼핑업종_이용금액'] == 0).sum())
print((train_df['_2순위업종_이용금액'] == 0).sum())
print((train_df['_1순위업종_이용금액'] == 0).sum())


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



train_df1['RV전환가능여부'].value_counts()

# 필요한 라이브러리 임포트 (이미 되어 있다고 가정)
import pandas as pd

# "_1순위신용체크구분"과 "_2순위신용체크구분"이 모두 결측치인 행의 개수 확인
missing_both = train_df['_1순위신용체크구분'].isna() & train_df['_2순위신용체크구분'].isna()
count_missing_both = missing_both.sum()

# 결과 출력
print(f"_1순위신용체크구분과 _2순위신용체크구분이 모두 결측치인 행의 개수: {count_missing_both}")

total_rows = 2400000

# 1. 결측치 개수 계산
missing_counts = train_df1.isnull().sum()

# 2. 결측치 비율 계산 (%)
missing_ratios = (missing_counts / total_rows) * 100

# 3. 결측치가 있는 변수만 필터링
missing_train_df1 = pd.DataFrame({
    '결측치 개수': missing_counts,
    '결측 비율 (%)': missing_ratios
})
missing_train_df1 = missing_train_df1[missing_train_df1['결측치 개수'] > 0]  # 결측치가 0인 변수 제외

# 4. 결과 정렬 (결측 비율 높은 순)
missing_train_df1 = missing_train_df1.sort_values(by='결측 비율 (%)', ascending=False)

# 5. 출력
print(f"총 데이터 수: {total_rows}")
print("\n결측치가 있는 변수 목록:")
print(missing_train_df1)


high_missing_cols = missing_ratios[missing_ratios >= 50].index.tolist()

train_df1 = train_df1.drop(columns=high_missing_cols)

missing_bleow_20 = missing_train_df[(missing_ratios >= 1) & (missing_ratios <= 20)]
train_df1[missing_bleow_20.index]

train_df1['가입통신회사코드'].fillna(train_df1['가입통신회사코드'].mode()[0], inplace=True)
train_df1['직장시도명'].fillna(train_df1['직장시도명'].mode()[0], inplace=True)
train_df1['RV전환가능여부'].fillna(train_df1['RV전환가능여부'].mode()[0], inplace=True)
train_df1['_1순위신용체크구분'].fillna(train_df1['_1순위신용체크구분'].mode()[0], inplace=True)

print(train_df1[['혜택수혜율_B0M', '혜택수혜율_R3M']].dropna().corr())

import numpy as np
train_df1['혜택수혜율_B0M'] = np.where(
    train_df1['혜택수혜율_B0M'].isna() & train_df1['혜택수혜율_R3M'].notna(),
    train_df1['혜택수혜율_R3M'],
    train_df1['혜택수혜율_B0M']
)
train_df1['혜택수혜율_B0M'] = train_df1['혜택수혜율_B0M'].fillna(0)

# 2. 혜택수혜율_R3M 결측치 처리
# B0M 값이 있으면 B0M으로 대체, 둘 다 없으면 0
train_df1['혜택수혜율_R3M'] = np.where(
    train_df1['혜택수혜율_R3M'].isna() & train_df1['혜택수혜율_B0M'].notna(),
    train_df1['혜택수혜율_B0M'],
    train_df1['혜택수혜율_R3M']
)
train_df1['혜택수혜율_R3M'] = train_df1['혜택수혜율_R3M'].fillna(0)

# 도메인 반영: 소지여부_신용이 'N'이면 둘 다 0으로 설정
train_df1.loc[(train_df1['소지여부_신용'] == 'N') | (train_df1['이용금액_R3M_신용'] == 0), ['혜택수혜율_B0M', '혜택수혜율_R3M']] = 0

train_df1['_1순위업종'] = np.where(train_df1['이용금액_R3M_신용'] == 0, '미사용', train_df1['_1순위업종'].fillna('Unknown'))

def fill_missing_with_usage(df, column, usage_col='이용금액_R3M_신용'):
    return np.where(df[usage_col] == 0, '미사용', df[column].fillna('Unknown'))

# 변수별 처리
columns_to_process = [
    '_2순위교통업종', '_3순위쇼핑업종', '_1순위납부업종', '_1순위교통업종',
    '_2순위쇼핑업종', '_3순위업종', '_2순위신용체크구분', '_1순위쇼핑업종', '_2순위업종'
]

for col in columns_to_process:
    train_df1[col] = fill_missing_with_usage(train_df1, col)

train_df1['OS구분코드'] = train_df1['OS구분코드'].fillna('Unknown')

# Segment 컬럼의 클래스 분포 확인
print(train_df["Segment"].value_counts())

# 시각화로 확인
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.countplot(x=train_df["Segment"])
plt.title("Class Distribution in Segment")
plt.show()



