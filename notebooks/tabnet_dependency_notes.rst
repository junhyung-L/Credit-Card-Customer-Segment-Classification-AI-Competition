.. code:: ipython3

    !pip uninstall scikit-learn numpy -y
    !pip install numpy scikit-learn --upgrade --force-reinstall
    


.. parsed-literal::

    Found existing installation: scikit-learn 1.4.2
    Uninstalling scikit-learn-1.4.2:
      Successfully uninstalled scikit-learn-1.4.2
    Found existing installation: numpy 2.2.4
    Uninstalling numpy-2.2.4:
      Successfully uninstalled numpy-2.2.4
    

.. code:: ipython3

    !pip install -U rtdl


.. parsed-literal::

    Requirement already satisfied: rtdl in c:\users\user\anaconda3\lib\site-packages (0.0.13)
    Requirement already satisfied: numpy<2,>=1.18 in c:\users\user\anaconda3\lib\site-packages (from rtdl) (1.26.4)
    INFO: pip is looking at multiple versions of rtdl to determine which version is compatible with other requirements. This could take a while.
    Collecting rtdl
      Using cached rtdl-0.0.13-py3-none-any.whl.metadata (1.0 kB)
      Using cached rtdl-0.0.12-py3-none-any.whl.metadata (1.0 kB)
      Using cached rtdl-0.0.10-py3-none-any.whl.metadata (1.0 kB)
      Using cached rtdl-0.0.9-py3-none-any.whl.metadata (1.0 kB)
    INFO: pip is still looking at multiple versions of rtdl to determine which version is compatible with other requirements. This could take a while.
      Using cached rtdl-0.0.8-py3-none-any.whl.metadata (1.0 kB)
      Using cached rtdl-0.0.7-py3-none-any.whl.metadata (934 bytes)
    INFO: This is taking longer than usual. You might need to provide the dependency resolver with stricter constraints to reduce runtime. See https://pip.pypa.io/warnings/backtracking for guidance. If you want to abort this run, press Ctrl + C.
      Using cached rtdl-0.0.6-py3-none-any.whl.metadata (934 bytes)
      Using cached rtdl-0.0.5-py3-none-any.whl.metadata (934 bytes)
      Using cached rtdl-0.0.4-py3-none-any.whl.metadata (934 bytes)
      Using cached rtdl-0.0.3-py3-none-any.whl.metadata (934 bytes)
      Using cached rtdl-0.0.2-py3-none-any.whl.metadata (995 bytes)
      Using cached rtdl-0.0.1-py3-none-any.whl.metadata (995 bytes)
    
    The conflict is caused by:
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.12 depends on torch<2 and >=1.7
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.10 depends on torch<2 and >=1.7
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.9 depends on torch<2 and >=1.7
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.8 depends on torch<2 and >=1.7
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.7 depends on torch<2 and >=1.6
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.6 depends on torch<2 and >=1.6
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.5 depends on torch<2 and >=1.6
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.4 depends on torch<2 and >=1.6
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.3 depends on torch<2 and >=1.6
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.2 depends on torch<2 and >=1.6
        rtdl 0.0.13 depends on torch<2 and >=1.7
        rtdl 0.0.1 depends on torch<2 and >=1.6
    
    To fix this you could try to:
    1. loosen the range of package versions you've specified
    2. remove package versions to allow pip attempt to solve the dependency conflict
    
    

.. parsed-literal::

    ERROR: Cannot install rtdl==0.0.1, rtdl==0.0.10, rtdl==0.0.12, rtdl==0.0.13, rtdl==0.0.2, rtdl==0.0.3, rtdl==0.0.4, rtdl==0.0.5, rtdl==0.0.6, rtdl==0.0.7, rtdl==0.0.8 and rtdl==0.0.9 because these package versions have conflicting dependencies.
    ERROR: ResolutionImpossible: for help visit https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts
    

.. code:: ipython3

    import pandas as pd
    import numpy as np
    import gc
    import os
    from sklearn.preprocessing import LabelEncoder

.. code:: ipython3

    train_sampled_df=pd.read_csv("C:/Users/user/Desktop/신용카드고객/train_sampled_df.csv")
    evaluation_df=pd.read_csv("C:/Users/user/Desktop/신용카드고객/evaluation_df.csv")


.. parsed-literal::

    C:\Users\user\AppData\Local\Temp\ipykernel_30332\639464490.py:2: DtypeWarning: Columns (385) have mixed types. Specify dtype option on import or set low_memory=False.
      evaluation_df=pd.read_csv("C:/Users/user/Desktop/신용카드고객/evaluation_df.csv")
    

.. code:: ipython3

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


.. parsed-literal::

    train_sampled_df after preprocessing:
       Unnamed: 0    기준년월            ID  남녀구분코드   연령 Segment  회원여부_이용가능  \
    0      599866  201808  TRAIN_199866       2  30대       E          1   
    1      427635  201808  TRAIN_027635       1  50대       E          0   
    2     1833852  201811  TRAIN_233852       2  40대       E          1   
    3      690821  201808  TRAIN_290821       1  40대       E          1   
    4     1098632  201809  TRAIN_298632       2  30대       D          1   
    
       회원여부_이용가능_CA  회원여부_이용가능_카드론  소지여부_신용  ...  변동률_RV일시불평잔  변동률_할부평잔  변동률_CA평잔  \
    0             1              0        1  ...     0.999998  1.987409  0.999998   
    1             0              0        1  ...     0.999998  0.999998  0.999998   
    2             1              1        1  ...     0.999998  0.999998  0.999998   
    3             1              1        1  ...     0.999998  0.904525  0.999998   
    4             1              1        1  ...     0.952195  0.604032  0.999998   
    
       변동률_RVCA평잔  변동률_카드론평잔  변동률_잔액_B1M  변동률_잔액_일시불_B1M  변동률_잔액_CA_B1M  \
    0    0.999998   0.999998    0.147471       -0.116887            0.0   
    1    0.999998   0.999998    0.000000        0.000000            0.0   
    2    0.999998   0.999998    0.000000        0.000000            0.0   
    3    0.999998   0.999998   -0.025178        0.102291            0.0   
    4    0.999998   0.999998    0.199584        0.085968            0.0   
    
       혜택수혜율_R3M  혜택수혜율_B0M  
    0   0.000000   0.000000  
    1   0.000000   0.000000  
    2   0.000000   0.000000  
    3   3.111331   3.220928  
    4  -0.084577   0.075305  
    
    [5 rows x 853 columns]
    evaluation_df after preprocessing:
       Unnamed: 0    기준년월            ID  남녀구분코드   연령 Segment  회원여부_이용가능  \
    0      140607  201807  TRAIN_140607       1  60대       E          1   
    1      615413  201808  TRAIN_215413       1  50대       E          1   
    2     2128921  201812  TRAIN_128921       1  50대       E          1   
    3      494497  201808  TRAIN_094497       2  40대       D          1   
    4     1814277  201811  TRAIN_214277       2  30대       E          1   
    
       회원여부_이용가능_CA  회원여부_이용가능_카드론  소지여부_신용  ...  변동률_RV일시불평잔  변동률_할부평잔  변동률_CA평잔  \
    0             1              1        1  ...     0.999998  0.999998  0.999998   
    1             1              0        1  ...     0.999998  0.901223  0.999998   
    2             1              1        1  ...     0.999998  0.999998  0.999998   
    3             1              0        1  ...     0.999998  1.990905  1.002733   
    4             1              1        1  ...     0.999998  0.000000  0.999998   
    
       변동률_RVCA평잔  변동률_카드론평잔  변동률_잔액_B1M  변동률_잔액_일시불_B1M  변동률_잔액_CA_B1M  \
    0    0.999998   0.999998    0.000000        0.000000        0.00000   
    1    0.999998   0.999998   -0.009084       -0.050106        0.00000   
    2    0.999998   0.999998    0.000000        0.000000        0.00000   
    3    0.999998   1.453043   -0.153078        0.000000       -0.07013   
    4    0.999998   0.999998   -0.025110       -0.140781        0.00000   
    
       혜택수혜율_R3M  혜택수혜율_B0M  
    0   0.000000   0.000000  
    1   4.216123   4.244498  
    2   0.000000   0.000000  
    3   0.000000   0.000000  
    4   1.732865   2.974604  
    
    [5 rows x 853 columns]
    

.. code:: ipython3

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
    


.. parsed-literal::

    Training TabNet...
    

.. parsed-literal::

    C:\Users\user\anaconda3\Lib\site-packages\pytorch_tabnet\abstract_model.py:82: UserWarning: Device used : cpu
      warnings.warn(f"Device used : {self.device}")
    

.. parsed-literal::

    epoch 0  | loss: 0.93222 | test_accuracy: 0.77705 |  0:00:23s
    epoch 1  | loss: 0.57087 | test_accuracy: 0.42175 |  0:00:50s
    epoch 2  | loss: 0.51032 | test_accuracy: 0.76045 |  0:01:04s
    epoch 3  | loss: 0.46622 | test_accuracy: 0.7983  |  0:01:18s
    epoch 4  | loss: 0.4302  | test_accuracy: 0.77005 |  0:01:32s
    epoch 5  | loss: 0.42396 | test_accuracy: 0.80905 |  0:01:51s
    epoch 6  | loss: 0.40947 | test_accuracy: 0.8293  |  0:02:05s
    epoch 7  | loss: 0.39964 | test_accuracy: 0.8361  |  0:02:19s
    epoch 8  | loss: 0.39249 | test_accuracy: 0.8378  |  0:02:32s
    epoch 9  | loss: 0.39234 | test_accuracy: 0.84085 |  0:02:46s
    epoch 10 | loss: 0.39019 | test_accuracy: 0.83985 |  0:03:01s
    epoch 11 | loss: 0.38356 | test_accuracy: 0.8408  |  0:03:15s
    epoch 12 | loss: 0.3766  | test_accuracy: 0.8423  |  0:03:30s
    epoch 13 | loss: 0.37444 | test_accuracy: 0.8405  |  0:03:43s
    epoch 14 | loss: 0.37173 | test_accuracy: 0.83625 |  0:03:59s
    epoch 15 | loss: 0.36827 | test_accuracy: 0.84145 |  0:04:14s
    epoch 16 | loss: 0.36452 | test_accuracy: 0.8391  |  0:04:31s
    epoch 17 | loss: 0.36625 | test_accuracy: 0.84135 |  0:04:46s
    epoch 18 | loss: 0.368   | test_accuracy: 0.84465 |  0:05:01s
    epoch 19 | loss: 0.36425 | test_accuracy: 0.84205 |  0:05:16s
    epoch 20 | loss: 0.3624  | test_accuracy: 0.84095 |  0:05:31s
    epoch 21 | loss: 0.35924 | test_accuracy: 0.84445 |  0:05:47s
    epoch 22 | loss: 0.35612 | test_accuracy: 0.84295 |  0:06:03s
    epoch 23 | loss: 0.35343 | test_accuracy: 0.8419  |  0:06:19s
    epoch 24 | loss: 0.3504  | test_accuracy: 0.83785 |  0:06:33s
    epoch 25 | loss: 0.34639 | test_accuracy: 0.8404  |  0:06:49s
    epoch 26 | loss: 0.34824 | test_accuracy: 0.82205 |  0:07:06s
    epoch 27 | loss: 0.34763 | test_accuracy: 0.84285 |  0:07:22s
    epoch 28 | loss: 0.34708 | test_accuracy: 0.8409  |  0:07:37s
    
    Early stopping occurred at epoch 28 with best_epoch = 18 and best_test_accuracy = 0.84465
    

.. parsed-literal::

    C:\Users\user\anaconda3\Lib\site-packages\pytorch_tabnet\callbacks.py:172: UserWarning: Best weights from best epoch are automatically used!
      warnings.warn(wrn_msg)
    

.. parsed-literal::

    TabNet - Train F1: 0.8345, Test F1: 0.8285
    

.. code:: ipython3

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


.. parsed-literal::

    FT Transformer 학습 시작...
    FT Transformer 실패: [enforce fail at alloc_cpu.cpp:114] data. DefaultCPUAllocator: not enough memory: you tried to allocate 12364800000 bytes.
    
