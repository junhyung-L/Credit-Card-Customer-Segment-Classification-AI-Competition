
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


train_df1=train_df.copy()

high_missing_cols = missing_ratios[missing_ratios >= 50].index.tolist()

train_df1 = train_df1.drop(columns=high_missing_cols)

missing_bleow_20 = missing_train_df[(missing_ratios >= 1) & (missing_ratios <= 20)]
train_df1[missing_bleow_20.index]

import numpy as np
#1. 가입통신회사코드
train_df1['가입통신회사코드'] = np.where(
    (train_df1['회원여부_이용가능'] == 'N') | (train_df1['이용금액_R3M_신용'] == 0),
    '미가입',
    train_df1['가입통신회사코드'].fillna('Unknown')
)

# 2. 직장시도명
train_df1['직장시도명'] = np.where(
    train_df1['직장시도명'].isna() & train_df1['거주시도명'].notna(),
    train_df1['거주시도명'],
    train_df1['직장시도명'].fillna('Unknown')
)

# 3. RV전환가능여부
train_df1['RV전환가능여부'] = np.where(
    (train_df1['이용금액_R3M_신용'] == 0) | (train_df1['소지여부_신용'] == 'N'),
    'N',
    train_df1['RV전환가능여부'].fillna('Unknown')
)

# 4. _1순위신용체크구분
train_df1['_1순위신용체크구분'] = np.where(
    train_df1['이용금액_R3M_신용'] == 0,
    '미사용',
    np.where(
        train_df1['_1순위신용체크구분'].isna() & train_df1['_1순위업종'].notna(),
        '신용',  # 업종 있으면 신용카드 사용 추정
        train_df1['_1순위신용체크구분'].fillna('Unknown')
    )
)

print(train_df1[['혜택수혜율_B0M', '혜택수혜율_R3M']].dropna().corr())


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

train_df1[['혜택수혜율_B0M', '혜택수혜율_R3M']]

train_df1[['_1순위교통업종','_2순위쇼핑업종','_3순위업종','_2순위신용체크구분','_1순위쇼핑업종','_2순위업종','_1순위업종']]

def fill_missing_20_50(df, target_col, usage_col='이용금액_R3M_신용', related_col=None):
    if related_col:
        return np.where(
            df[usage_col] == 0, '미사용',
            np.where(
                df[target_col].isna() & df[related_col].notna(), df[related_col],
                df[target_col].fillna('Unknown')
            )
        )
    return np.where(df[usage_col] == 0, '미사용', df[target_col].fillna('Unknown'))

# 변수별 처리
train_df1['_1순위교통업종'] = fill_missing_20_50(train_df1, '_1순위교통업종', related_col='_1순위업종')
train_df1['_2순위쇼핑업종'] = fill_missing_20_50(train_df1, '_2순위쇼핑업종', related_col='_1순위쇼핑업종')
train_df1['_3순위업종'] = fill_missing_20_50(train_df1, '_3순위업종', related_col='_2순위업종')
train_df1['_2순위신용체크구분'] = fill_missing_20_50(train_df1, '_2순위신용체크구분', related_col='_1순위신용체크구분')
train_df1['_1순위쇼핑업종'] = fill_missing_20_50(train_df1, '_1순위쇼핑업종', related_col='_1순위업종')
train_df1['_2순위업종'] = fill_missing_20_50(train_df1, '_2순위업종', related_col='_1순위업종')
train_df1['_1순위업종'] = fill_missing_20_50(train_df1, '_1순위업종')

train_df1[['_1순위교통업종','_2순위쇼핑업종','_3순위업종','_2순위신용체크구분','_1순위쇼핑업종','_2순위업종','_1순위업종']]


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


####Test 전처리####

test_df1=test_df.copy()
test_df1=test_df1.drop(columns=high_missing_cols)

import numpy as np

# 1. 가입통신회사코드
test_df1['가입통신회사코드'] = np.where(
    (test_df1['회원여부_이용가능'] == 'N') | (test_df1['이용금액_R3M_신용'] == 0),
    '미가입',
    test_df1['가입통신회사코드'].fillna('Unknown')
)

# 2. 직장시도명
test_df1['직장시도명'] = np.where(
    test_df1['직장시도명'].isna() & test_df1['거주시도명'].notna(),
    test_df1['거주시도명'],
    test_df1['직장시도명'].fillna('Unknown')
)

# 3. RV전환가능여부
test_df1['RV전환가능여부'] = np.where(
    (test_df1['이용금액_R3M_신용'] == 0) | (test_df1['소지여부_신용'] == 'N'),
    'N',
    test_df1['RV전환가능여부'].fillna('Unknown')
)

# 4. _1순위신용체크구분
test_df1['_1순위신용체크구분'] = np.where(
    test_df1['이용금액_R3M_신용'] == 0,
    '미사용',
    np.where(
        test_df1['_1순위신용체크구분'].isna() & test_df1['_1순위업종'].notna(),
        '신용',  # 업종 있으면 신용카드 사용 추정
        test_df1['_1순위신용체크구분'].fillna('Unknown')
    )
)

test_df1['혜택수혜율_B0M'] = np.where(
    test_df1['혜택수혜율_B0M'].isna() & test_df1['혜택수혜율_R3M'].notna(),
    test_df1['혜택수혜율_R3M'],
    test_df1['혜택수혜율_B0M']
)
test_df1['혜택수혜율_B0M'] = test_df1['혜택수혜율_B0M'].fillna(0)

# 2. 혜택수혜율_R3M 결측치 처리
# B0M 값이 있으면 B0M으로 대체, 둘 다 없으면 0
test_df1['혜택수혜율_R3M'] = np.where(
    test_df1['혜택수혜율_R3M'].isna() & test_df1['혜택수혜율_B0M'].notna(),
    test_df1['혜택수혜율_B0M'],
    test_df1['혜택수혜율_R3M']
)
test_df1['혜택수혜율_R3M'] = test_df1['혜택수혜율_R3M'].fillna(0)

# 도메인 반영: 소지여부_신용이 'N'이면 둘 다 0으로 설정
test_df1.loc[(test_df1['소지여부_신용'] == 'N') | (test_df1['이용금액_R3M_신용'] == 0), ['혜택수혜율_B0M', '혜택수혜율_R3M']] = 0

def fill_missing_20_50(df, target_col, usage_col='이용금액_R3M_신용', related_col=None):
    if related_col:
        return np.where(
            df[usage_col] == 0, '미사용',
            np.where(
                df[target_col].isna() & df[related_col].notna(), df[related_col],
                df[target_col].fillna('Unknown')
            )
        )
    return np.where(df[usage_col] == 0, '미사용', df[target_col].fillna('Unknown'))

# 변수별 처리
test_df1['_1순위교통업종'] = fill_missing_20_50(test_df1, '_1순위교통업종', related_col='_1순위업종')
test_df1['_2순위쇼핑업종'] = fill_missing_20_50(test_df1, '_2순위쇼핑업종', related_col='_1순위쇼핑업종')
test_df1['_3순위업종'] = fill_missing_20_50(test_df1, '_3순위업종', related_col='_2순위업종')
test_df1['_2순위신용체크구분'] = fill_missing_20_50(test_df1, '_2순위신용체크구분', related_col='_1순위신용체크구분')
test_df1['_1순위쇼핑업종'] = fill_missing_20_50(test_df1, '_1순위쇼핑업종', related_col='_1순위업종')
test_df1['_2순위업종'] = fill_missing_20_50(test_df1, '_2순위업종', related_col='_1순위업종')
test_df1['_1순위업종'] = fill_missing_20_50(test_df1, '_1순위업종')


feature_cols = [col for col in train_df1.columns if col not in ["ID", "Segment"]]

X = train_df1[feature_cols].copy()
y = train_df1["Segment"].copy()

# 타깃 라벨 인코딩
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

X_test = test_df1.copy()

encoders = {}  # 각 컬럼별 encoder 저장

for col in categorical_features:
    le_train = LabelEncoder()
    X[col] = le_train.fit_transform(X[col])
    encoders[col] = le_train
    unseen_labels_val = set(X_test[col]) - set(le_train.classes_)
    if unseen_labels_val:
        le_train.classes_ = np.append(le_train.classes_, list(unseen_labels_val))
    X_test[col] = le_train.transform(X_test[col])
gc.collect()

model = xgb.XGBClassifier(random_state=42)
model.fit(X, y_encoded)

X_test.drop(columns=['ID'],inplace=True)

# row-level 예측 수행
y_test_pred = model.predict(X_test)
# 예측 결과를 변환
y_test_pred_labels = le_target.inverse_transform(y_test_pred)

# row 단위 예측 결과를 test_data에 추가
test_data = test_df.copy()  # 원본 유지
test_data["pred_label"] = y_test_pred_labels

submission = test_data.groupby("ID")["pred_label"] \
    .agg(lambda x: x.value_counts().idxmax()) \
    .reset_index()

submission.columns = ["ID", "Segment"]
submission.to_csv('./base_submit.csv',index=False)

# Segment 컬럼의 클래스 분포 확인
print(train_df["Segment"].value_counts())

# 시각화로 확인
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
sns.countplot(x=train_df["Segment"])
plt.title("Class Distribution in Segment")
plt.show()



