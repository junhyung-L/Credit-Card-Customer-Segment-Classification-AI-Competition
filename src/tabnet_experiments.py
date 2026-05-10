!pip uninstall scikit-learn numpy -y
!pip install numpy scikit-learn --upgrade --force-reinstall


!pip install -U rtdl

import pandas as pd
import numpy as np
import gc
import os
from sklearn.preprocessing import LabelEncoder

train_sampled_df=pd.read_csv("C:/Users/user/Desktop/신용카드고객/train_sampled_df.csv")
evaluation_df=pd.read_csv("C:/Users/user/Desktop/신용카드고객/evaluation_df.csv")

import numpy as np
import pandas as pd

def preprocess_df(df):
    """
    train_sampled_df와 evaluation_df에 동일한 전처리를 적용하는 함수
    """
    # 1. 업종 목록 결측치 처리
    industry_list = [
        '_3순위여유업종', '_3순위납부업종', '_2순위여유업종', '_3순위교통업종', '_2순위납부업종',
        '_1순위여유업종', '_2순위교통업종', '_3순위쇼핑업종', '_1순위납부업종', '_1순위교통업종',
        '_2순위쇼핑업종', '_3순위업종', '_1순위쇼핑업종', '_2순위업종', '_1순위업종'
    ]
    for industry in industry_list:
        if industry in df.columns:
            df[industry] = df[industry].fillna('없음')

    # 2. 불필요한 열 삭제
    columns_to_drop = [
        '연체일자_B0M', '최종카드론_대출일자', '최종카드론_신청경로코드', '최종카드론_금융상환방식코드',
        'RV신청일자', 'OS구분코드'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    # 3. 가입통신회사코드 처리
    if '회원여부_이용가능' in df.columns and '이용금액_R3M_신용' in df.columns:
        df['가입통신회사코드'] = np.where(
            (df['회원여부_이용가능'] == 'N') | (df['이용금액_R3M_신용'] == 0),
            '미가입',
            df['가입통신회사코드'].fillna('Unknown')
        )

    # 4. 직장시도명 처리
    if '거주시도명' in df.columns:
        df['직장시도명'] = np.where(
            df['직장시도명'].isna() & df['거주시도명'].notna(),
            df['거주시도명'],
            df['직장시도명'].fillna('Unknown')
        )

    # 5. RV전환가능여부 처리
    if '소지여부_신용' in df.columns and '이용금액_R3M_신용' in df.columns:
        df['RV전환가능여부'] = np.where(
            (df['이용금액_R3M_신용'] == 0) | (df['소지여부_신용'] == 'N'),
            'N',
            df['RV전환가능여부'].fillna('Unknown')
        )

    # 6. _1순위신용체크구분 처리
    if '_1순위업종' in df.columns and '이용금액_R3M_신용' in df.columns:
        df['_1순위신용체크구분'] = np.where(
            df['이용금액_R3M_신용'] == 0,
            '미사용',
            np.where(
                df['_1순위신용체크구분'].isna() & df['_1순위업종'].notna() & (df['_1순위업종'] != '없음'),
                '신용',
                df['_1순위신용체크구분'].fillna('미사용')
            )
        )
        # _1순위와 _2순위 상호작용
        df.loc[df['_1순위신용체크구분'] == '신용', '_2순위신용체크구분'] = '체크'
        df.loc[df['_1순위신용체크구분'] == '체크', '_2순위신용체크구분'] = '신용'
        df.loc[df['_1순위신용체크구분'] == '미사용', '_2순위신용체크구분'] = '미사용'

    # 7. 혜택수혜율 처리
    if '혜택수혜율_R3M' in df.columns:
        df['혜택수혜율_B0M'] = np.where(
            df['혜택수혜율_B0M'].isna() & df['혜택수혜율_R3M'].notna(),
            df['혜택수혜율_R3M'],
            df['혜택수혜율_B0M']
        )
        df['혜택수혜율_B0M'] = df['혜택수혜율_B0M'].fillna(0)

        df['혜택수혜율_R3M'] = np.where(
            df['혜택수혜율_R3M'].isna() & df['혜택수혜율_B0M'].notna(),
            df['혜택수혜율_B0M'],
            df['혜택수혜율_R3M']
        )
        df['혜택수혜율_R3M'] = df['혜택수혜율_R3M'].fillna(0)

        # 도메인 반영
        if '소지여부_신용' in df.columns:
            df.loc[(df['소지여부_신용'] == 'N') | (df['이용금액_R3M_신용'] == 0), ['혜택수혜율_B0M', '혜택수혜율_R3M']] = 0

    # 8. 날짜 열 결측치 처리
    date_cols = ['최종유효년월_신용_이용', '최종유효년월_신용_이용가능', '최종카드발급일자']
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    return df

# 전처리 적용
train_sampled_df = preprocess_df(train_sampled_df)
evaluation_df = preprocess_df(evaluation_df)

# 결과 확인 (선택적)
print("train_sampled_df after preprocessing:")
print(train_sampled_df.head())
print("evaluation_df after preprocessing:")
print(evaluation_df.head())

# 필수 라이브러리
import numpy as np
import pandas as pd
import torch
import os
import random
import gc

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

# 시드 고정 함수
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 데이터셋 준비
feature_cols = [col for col in train_sampled_df.columns if col not in ["ID", "Segment"]]
X = train_sampled_df[feature_cols].copy()
y = train_sampled_df["Segment"].copy()

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

X_test = evaluation_df[feature_cols].copy()
y_true = evaluation_df["Segment"].copy()
y_true_encoded = le_target.transform(y_true)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = X_test[col].fillna('missing').astype(str)
    unseen = set(X_test[col]) - set(le.classes_)
    if unseen:
        le.classes_ = np.append(le.classes_, list(unseen))
    X_test[col] = le.transform(X_test[col])
    encoders[col] = le

# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# TabNet 학습 및 평가
# -------------------------------
try:
    print("Training TabNet...")
    clf_tabnet = TabNetClassifier(seed=42)
    clf_tabnet.fit(
        X.values, y_encoded,
        eval_set=[(X_test.values, y_true_encoded)],
        eval_name=['test'],
        eval_metric=['accuracy'],
        max_epochs=100,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )

    y_pred_train_tabnet = clf_tabnet.predict(X.values)
    y_pred_tabnet = clf_tabnet.predict(X_test.values)

    f1_train_tabnet = f1_score(y_encoded, y_pred_train_tabnet, average='weighted')
    f1_test_tabnet = f1_score(y_true_encoded, y_pred_tabnet, average='weighted')
    print(f"TabNet - Train F1: {f1_train_tabnet:.4f}, Test F1: {f1_test_tabnet:.4f}")
except Exception as e:
    print(f"TabNet failed: {e}")


# 필수 라이브러리
import numpy as np
import pandas as pd
import torch
import os
import random
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import rtdl  # FT Transformer 라이브러리
import torch.nn as nn

# 시드 고정 함수
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 데이터셋 준비
feature_cols = [col for col in train_sampled_df.columns if col not in ["ID", "Segment"]]  # 피처 열 선택
X = train_sampled_df[feature_cols].copy()  # 훈련 데이터 복사
y = train_sampled_df["Segment"].copy()  # 타겟 데이터 복사

X_test = evaluation_df[feature_cols].copy()  # 테스트 데이터 복사
y_true = evaluation_df["Segment"].copy()  # 테스트 타겟 복사

# 타겟 인코딩 (타겟은 여전히 숫자로 변환 필요)
from sklearn.preprocessing import LabelEncoder
le_target = LabelEncoder()  # 타겟 라벨 인코더
y_encoded = le_target.fit_transform(y)  # 훈련 타겟 인코딩
y_true_encoded = le_target.transform(y_true)  # 테스트 타겟 인코딩

# 범주형 및 수치형 피처 분리
categorical_features = X.select_dtypes(include=['object']).columns.tolist()  # 범주형 피처 목록
numerical_features = [col for col in feature_cols if col not in categorical_features]  # 수치형 피처 목록

# 수치형 피처 스케일링
scaler = StandardScaler()  # 표준화 스케일러
X_num = scaler.fit_transform(X[numerical_features])  # 훈련 데이터 수치형 피처 스케일링
X_test_num = scaler.transform(X_test[numerical_features])  # 테스트 데이터 수치형 피처 스케일링

# 범주형 피처의 카디널리티 계산
cat_cardinalities = [len(X[col].unique()) for col in categorical_features]  # 각 범주형 피처의 고유 값 개수

# 범주형 피처를 정수형으로 변환 (0부터 시작하는 인덱스로)
X_cat = np.stack([X[col].astype('category').cat.codes for col in categorical_features], axis=1)  # 훈련 데이터 범주형 변환
X_test_cat = np.stack([X_test[col].fillna('missing').astype('category').cat.codes for col in categorical_features], axis=1)  # 테스트 데이터 범주형 변환

# FT Transformer용 데이터 준비 (torch 텐서로 변환)
X_num_tensor = torch.tensor(X_num, dtype=torch.float32)  # 훈련 수치형 데이터 텐서
X_cat_tensor = torch.tensor(X_cat, dtype=torch.long) if categorical_features else None  # 훈련 범주형 데이터 텐서 (없으면 None)
y_encoded_tensor = torch.tensor(y_encoded, dtype=torch.long)  # 훈련 타겟 텐서
X_test_num_tensor = torch.tensor(X_test_num, dtype=torch.float32)  # 테스트 수치형 데이터 텐서
X_test_cat_tensor = torch.tensor(X_test_cat, dtype=torch.long) if categorical_features else None  # 테스트 범주형 데이터 텐서
y_true_encoded_tensor = torch.tensor(y_true_encoded, dtype=torch.long)  # 테스트 타겟 텐서

# -------------------------------
# FT Transformer 학습 및 평가
# -------------------------------
try:
    print("FT Transformer 학습 시작...")
    
    # FT Transformer 모델 정의
    d_out = 64  # FT Transformer 출력 차원 (임의로 64로 설정, 조정 가능)
    ft_transformer = rtdl.FTTransformer.make_default(
        n_num_features=len(numerical_features),  # 수치형 피처 수
        cat_cardinalities=cat_cardinalities if categorical_features else None,  # 범주형 피처 카디널리티
        d_out=d_out  # 출력 차원 명시
    )

    # 분류를 위한 출력 레이어 정의
    n_classes = len(np.unique(y_encoded))  # 클래스 수
    classifier = nn.Linear(d_out, n_classes)  # 출력 레이어

    # 옵티마이저와 손실 함수 정의
    optimizer = torch.optim.Adam(list(ft_transformer.parameters()) + list(classifier.parameters()), lr=1e-3)  # Adam 옵티마이저
    loss_fn = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 여부 확인
    ft_transformer.to(device)  # FT Transformer를 장치로 이동
    classifier.to(device)  # 출력 레이어를 장치로 이동
    X_num_tensor = X_num_tensor.to(device)  # 훈련 수치형 데이터 장치로 이동
    X_cat_tensor = X_cat_tensor.to(device) if X_cat_tensor is not None else None  # 훈련 범주형 데이터 장치로 이동
    y_encoded_tensor = y_encoded_tensor.to(device)  # 훈련 타겟 장치로 이동
    X_test_num_tensor = X_test_num_tensor.to(device)  # 테스트 수치형 데이터 장치로 이동
    X_test_cat_tensor = X_test_cat_tensor.to(device) if X_test_cat_tensor is not None else None  # 테스트 범주형 데이터 장치로 이동
    y_true_encoded_tensor = y_true_encoded_tensor.to(device)  # 테스트 타겟 장치로 이동

    # 학습 루프
    max_epochs = 100  # 최대 에포크 수
    patience = 10  # 조기 종료 인내심
    best_f1 = -float('inf')  # 최고 F1 점수 초기화
    patience_counter = 0  # 조기 종료 카운터

    for epoch in range(max_epochs):
        ft_transformer.train()  # 학습 모드
        classifier.train()
        optimizer.zero_grad()  # 기울기 초기화
        transformer_output = ft_transformer(X_num_tensor, X_cat_tensor)  # FT Transformer 출력
        outputs = classifier(transformer_output)  # 출력 레이어로 클래스 예측
        loss = loss_fn(outputs, y_encoded_tensor)  # 손실 계산
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트

        # 평가
        ft_transformer.eval()  # 평가 모드
        classifier.eval()
        with torch.no_grad():  # 기울기 계산 비활성화
            test_transformer_output = ft_transformer(X_test_num_tensor, X_test_cat_tensor)  # 테스트 데이터로 FT Transformer 출력
            y_pred_test = classifier(test_transformer_output).argmax(dim=1).cpu().numpy()  # 테스트 예측
            f1_test = f1_score(y_true_encoded, y_pred_test, average='weighted')  # 테스트 F1 점수
            print(f"에포크 {epoch+1}/{max_epochs}, 테스트 F1: {f1_test:.4f}")

            # 조기 종료
            if f1_test > best_f1:
                best_f1 = f1_test  # 최고 F1 점수 갱신
                patience_counter = 0  # 카운터 초기화
            else:
                patience_counter += 1  # 카운터 증가
                if patience_counter >= patience:
                    print("조기 종료가 실행되었습니다.")
                    break

    # 최종 예측
    ft_transformer.eval()  # 평가 모드
    classifier.eval()
    with torch.no_grad():
        train_transformer_output = ft_transformer(X_num_tensor, X_cat_tensor)
        y_pred_train_ft = classifier(train_transformer_output).argmax(dim=1).cpu().numpy()  # 훈련 데이터 예측
        test_transformer_output = ft_transformer(X_test_num_tensor, X_test_cat_tensor)
        y_pred_test_ft = classifier(test_transformer_output).argmax(dim=1).cpu().numpy()  # 테스트 데이터 예측

    f1_train_ft = f1_score(y_encoded, y_pred_train_ft, average='weighted')  # 훈련 F1 점수
    f1_test_ft = f1_score(y_true_encoded, y_pred_test_ft, average='weighted')  # 테스트 F1 점수
    print(f"FT Transformer - 훈련 F1: {f1_train_ft:.4f}, 테스트 F1: {f1_test_ft:.4f}")

except Exception as e:
    print(f"FT Transformer 실패: {e}")

# 필수 라이브러리
import numpy as np
import pandas as pd
import torch
import os
import random
import gc
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import rtdl  # FT Transformer 라이브러리
import torch.nn as nn

# 시드 고정 함수
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 데이터셋 준비
feature_cols = [col for col in train_sampled_df.columns if col not in ["ID", "Segment"]]  # 피처 열 선택
X = train_sampled_df[feature_cols].copy()  # 훈련 데이터 복사
y = train_sampled_df["Segment"].copy()  # 타겟 데이터 복사

X_test = evaluation_df[feature_cols].copy()  # 테스트 데이터 복사
y_true = evaluation_df["Segment"].copy()  # 테스트 타겟 복사

# 타겟 인코딩
from sklearn.preprocessing import LabelEncoder
le_target = LabelEncoder()  # 타겟 라벨 인코더
y_encoded = le_target.fit_transform(y)  # 훈련 타겟 인코딩
y_true_encoded = le_target.transform(y_true)  # 테스트 타겟 인코딩

# 범주형 및 수치형 피처 분리
categorical_features = X.select_dtypes(include=['object']).columns.tolist()  # 범주형 피처 목록
numerical_features = [col for col in feature_cols if col not in categorical_features]  # 수치형 피처 목록

# 수치형 피처 스케일링
scaler = StandardScaler()  # 표준화 스케일러
X_num = scaler.fit_transform(X[numerical_features])  # 훈련 데이터 수치형 피처 스케일링
X_test_num = scaler.transform(X_test[numerical_features])  # 테스트 데이터 수치형 피처 스케일링

# 범주형 피처의 카디널리티 계산
cat_cardinalities = [len(X[col].unique()) for col in categorical_features]  # 각 범주형 피처의 고유 값 개수

# 범주형 피처를 정수형으로 변환
X_cat = np.stack([X[col].astype('category').cat.codes for col in categorical_features], axis=1)  # 훈련 데이터 범주형 변환
X_test_cat = np.stack([X_test[col].fillna('missing').astype('category').cat.codes for col in categorical_features], axis=1)  # 테스트 데이터 범주형 변환

# FT Transformer용 데이터 준비 (torch 텐서로 변환)
X_num_tensor = torch.tensor(X_num, dtype=torch.float32)  # 훈련 수치형 데이터 텐서
X_cat_tensor = torch.tensor(X_cat, dtype=torch.long) if categorical_features else None  # 훈련 범주형 데이터 텐서
y_encoded_tensor = torch.tensor(y_encoded, dtype=torch.long)  # 훈련 타겟 텐서
X_test_num_tensor = torch.tensor(X_test_num, dtype=torch.float32)  # 테스트 수치형 데이터 텐서
X_test_cat_tensor = torch.tensor(X_test_cat, dtype=torch.long) if categorical_features else None  # 테스트 범주형 데이터 텐서
y_true_encoded_tensor = torch.tensor(y_true_encoded, dtype=torch.long)  # 테스트 타겟 텐서

# DataLoader로 배치 처리 준비
batch_size = 1024  # 배치 크기 (메모리에 따라 조정 가능)
train_dataset = TensorDataset(X_num_tensor, X_cat_tensor if X_cat_tensor is not None else torch.zeros_like(X_num_tensor), y_encoded_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(X_test_num_tensor, X_test_cat_tensor if X_test_cat_tensor is not None else torch.zeros_like(X_test_num_tensor), y_true_encoded_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# FT Transformer 학습 및 평가
# -------------------------------
try:
    print("FT Transformer 학습 시작...")
    
    # FT Transformer 모델 정의
    d_out = 32  # 출력 차원 줄임 (메모리 절약)
    ft_transformer = rtdl.FTTransformer.make_default(
        n_num_features=len(numerical_features),  # 수치형 피처 수
        cat_cardinalities=cat_cardinalities if categorical_features else None,  # 범주형 피처 카디널리티
        d_out=d_out,  # 출력 차원 명시
        d_token=96,  # 토큰 차원 줄임 (기본값 192 -> 96)
        n_layers=2   # 레이어 수 줄임 (기본값 3 -> 2)
    )

    # 분류를 위한 출력 레이어 정의
    n_classes = len(np.unique(y_encoded))  # 클래스 수
    classifier = nn.Linear(d_out, n_classes)  # 출력 레이어

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 사용 여부 확인
    ft_transformer.to(device)  # FT Transformer를 장치로 이동
    classifier.to(device)  # 출력 레이어를 장치로 이동

    # 옵티마이저와 손실 함수 정의
    optimizer = torch.optim.Adam(list(ft_transformer.parameters()) + list(classifier.parameters()), lr=1e-3)  # Adam 옵티마이저
    loss_fn = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수

    # 학습 루프
    max_epochs = 100  # 최대 에포크 수
    patience = 10  # 조기 종료 인내심
    best_f1 = -float('inf')  # 최고 F1 점수 초기화
    patience_counter = 0  # 조기 종료 카운터

    for epoch in range(max_epochs):
        ft_transformer.train()  # 학습 모드
        classifier.train()
        for batch in train_loader:
            x_num_batch, x_cat_batch, y_batch = [b.to(device) for b in batch]
            if X_cat_tensor is None:  # 범주형 데이터가 없는 경우
                x_cat_batch = None
            optimizer.zero_grad()  # 기울기 초기화
            transformer_output = ft_transformer(x_num_batch, x_cat_batch)  # FT Transformer 출력
            outputs = classifier(transformer_output)  # 출력 레이어로 클래스 예측
            loss = loss_fn(outputs, y_batch)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

        # 평가
        ft_transformer.eval()  # 평가 모드
        classifier.eval()
        y_pred_test_all = []
        with torch.no_grad():  # 기울기 계산 비활성화
            for batch in test_loader:
                x_num_batch, x_cat_batch, y_batch = [b.to(device) for b in batch]
                if X_test_cat_tensor is None:  # 범주형 데이터가 없는 경우
                    x_cat_batch = None
                test_transformer_output = ft_transformer(x_num_batch, x_cat_batch)  # 테스트 데이터로 FT Transformer 출력
                y_pred_test = classifier(test_transformer_output).argmax(dim=1).cpu().numpy()  # 테스트 예측
                y_pred_test_all.extend(y_pred_test)
        
        f1_test = f1_score(y_true_encoded, y_pred_test_all, average='weighted')  # 테스트 F1 점수
        print(f"에포크 {epoch+1}/{max_epochs}, 테스트 F1: {f1_test:.4f}")

        # 조기 종료
        if f1_test > best_f1:
            best_f1 = f1_test  # 최고 F1 점수 갱신
            patience_counter = 0  # 카운터 초기화
        else:
            patience_counter += 1  # 카운터 증가
            if patience_counter >= patience:
                print("조기 종료가 실행되었습니다.")
                break

    # 최종 예측 (훈련 데이터)
    ft_transformer.eval()  # 평가 모드
    classifier.eval()
    y_pred_train_all = []
    with torch.no_grad():
        for batch in train_loader:
            x_num_batch, x_cat_batch, y_batch = [b.to(device) for b in batch]
            if X_cat_tensor is None:
                x_cat_batch = None
            train_transformer_output = ft_transformer(x_num_batch, x_cat_batch)
            y_pred_train = classifier(train_transformer_output).argmax(dim=1).cpu().numpy()
            y_pred_train_all.extend(y_pred_train)

    f1_train_ft = f1_score(y_encoded, y_pred_train_all, average='weighted')  # 훈련 F1 점수
    f1_test_ft = f1_score(y_true_encoded, y_pred_test_all, average='weighted')  # 테스트 F1 점수
    print(f"FT Transformer - 훈련 F1: {f1_train_ft:.4f}, 테스트 F1: {f1_test_ft:.4f}")

except Exception as e:
    print(f"FT Transformer 실패: {e}")

# 메모리 정리
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

