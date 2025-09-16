# 신용카드 고객 세그먼트 분류 AI 경진대회 (Dacon)  

![포스터](https://static.onoffmix.com/afv2/attach/2025/03/10/v3d3fe415a2850d703c6fda7d83499d87a.png)
> **목표**: 고객의 신용·거래·마케팅 데이터를 바탕으로 신용카드 고객 **세그먼트(Segment)**를 분류하는 AI 모델 개발  
> **성과**: Public **0.64636 (75등)** · Private **0.6251 (58등, 상위 25%)**  
> **내부 검증 F1 (Best)**: **0.8936 (Stacking, meta=LogReg)**  

---

## 프로젝트 개요
- **주최/주관**: 데이콘  
- **후원**: 한국지능정보사회진흥원(NIA)  
- **주제**: 일정 기간 동안의 신용·거래·채널·마케팅·성과 데이터를 활용한 **고객 세그먼트 분류**  
- **의의**:  
  - 카드사의 **고객 맞춤형 마케팅** 및 **CRM 전략** 고도화  
  - **리스크 관리(고객 이탈·부도 예측)** 기반 모델링 확장 가능성  

---

## 산업적 배경
- 신용카드사는 고객의 소비/신용 행태를 기반으로 마케팅 전략을 수립합니다.  
- **세그먼트 분류 모델**은 다음과 같은 실제 비즈니스 문제와 직결됩니다:  
  - 신규 카드 발급 고객 대상 **맞춤 혜택 추천**  
  - **이탈(Churn) 가능 고객** 조기 탐지  
  - **이상거래 탐지(Fraud Detection)**와의 연계 가능  
- 따라서 정확한 세그먼트 분류는 **마케팅 ROI 향상**, **리스크 관리 비용 절감**, **고객 만족도 제고**에 기여합니다.  

---

## 데이터 탐색 (EDA)
- **표본 수**: 약 240,000명  
- **컬럼 수**: 857개  
- **클래스 불균형 존재**  
  - Segment A: 소수(972) → 오버샘플링 필요  
  - Segment E: 다수(1,922,052) → 언더샘플링 필요  
- **결측치 분포**:  
  - 50% 이상 결측 → 제거  
  - 20~50% → 상관 변수 기반 대체  
  - 20% 이하 → 최빈값/Unknown 대체  
- **패턴 발견**  
  - `_순위업종` 결측 ↔ `업종이용금액=0` → 규칙형 대체  
  - `_1순위신용체크구분` ↔ `_2순위신용체크구분` → 상호작용 변수 생성 가능  

**시각화 활용**  
- **Missingno**: 결측치 구조 파악  
- **Seaborn**: 주요 변수 분포 비교  
- **히트맵**: 변수 간 상관관계 확인  

---

## 전처리 전략
- **결측치 처리**  
  - 50% 이상 → 컬럼 삭제 (예: `최종카드론_대출일자`)  
  - 20~50% → 패턴 기반 대체 (예: 혜택수혜율 R3M → R6M/R12M 활용)  
  - 20% 이하 → 최빈값, Unknown 처리  
- **스케일링**  
  - 로그 변환 + StandardScaler 적용  
- **인코딩**  
  - LabelEncoder 우선 (CatBoost, XGB, LightGBM 등 트리모델과 호환)  
- **대용량 연산**  
  - Colab 메모리 초과 → **Dask DataFrame** 사용  

---

## 모델링 접근법
### (1) 베이스라인
- **XGBoost baseline**  
- F1: **0.607 (Public)**  

### (2) 전처리 반영 XGB
- 결측치 패턴 변수를 활용한 XGB  
- F1: **0.625 (Public)**  

### (3) 주요 모델 비교 (내부 F1)
| Model                  | PCA 미적용 | PCA 95%  | PCA 99%  |
|------------------------|------------|----------|----------|
| **CatBoost**           | **0.8893** | 0.8444   | 0.8478   |
| Logistic Regression    | 0.8794     | 0.8539   | **0.8645** |
| XGBoost                | 0.8880     | 0.8367   | 0.8393   |
| Random Forest          | 0.8625     | 0.8050   | 0.7919   |
| SVM                    | 0.8488     | 0.8443   | 0.8467   |
| DNN                    | 0.8629     | 0.8525   | 0.8623   |
| MLP                    | 0.8645     | 0.8468   | 0.8592   |
| CNN                    | 0.8601     | 0.8314   | 0.8376   |

- **CatBoost (0.8893)**: 최고 성능, 다소 과적합  
- **Logistic Regression (PCA 99%) (0.8645)**: 안정적 성능, 과적합 완화  

---

## SOTA 모델 실험
### TabNet
- **Train F1: 0.8345, Test F1: 0.8285**  
- 트리 기반보다 낮은 성능  
- NumPy/Scikit-learn 버전 충돌 → 별도 환경 필요  

### FT-Transformer
- Colab GPU 환경에서도 **메모리 초과**  
- 미니배치+경량화 적용했으나 학습 불가  

### NODE (Neural Oblivious Decision Ensembles)
- 초기 실험 단계 (epochs=10, lr=1e-3)  
- 성능 개선 확인 못함  

**Negative Results**  
- 대형 Tabular 딥러닝 모델은 메모리/성능/튜닝 비용 대비 효과 낮음  
- 본 데이터셋은 여전히 **트리 기반 모델 + 선형 모델 앙상블**이 최적  

---

## 앙상블 (Stacking)
- **메타모델: Logistic Regression**  
  - CatBoost + Logistic Regression + MLP → **0.8936 (Best)**  
- **메타모델: CatBoost**  
  - CatBoost + Logistic Regression + MLP → 0.8911  

단일 모델 CatBoost(0.8893) 대비, 스태킹(meta=LogReg)으로 **성능 +0.0043p 개선**  

---

## 최종 결과
- **Public Score**: 0.64636 (75등)  
- **Private Score**: 0.6251 (58등, 상위 25%)  
- **내부 검증 F1 Best**: 0.8936 (Stacking, Logistic Regression 메타모델)  

---

## 실험 요약 테이블
| 구분 | 모델 | 특이사항 | F1-score |
|------|------|----------|----------|
| Baseline | XGBoost | 데이콘 기본 제공 | 0.607 (Public) |
| 개선 | XGBoost | 결측치 변수 반영 | 0.625 (Public) |
| 단일모델 | CatBoost | PCA 미적용 | 0.8893 |
| 단일모델 | Logistic Regression | PCA 99% 적용 | 0.8645 |
| 앙상블 | CatBoost+LogReg+MLP (meta=LogReg) | Stacking | **0.8936** |

---

## 모델 성능 비교 (내부 검증 F1)

![Model Performance](./model_f1_comparison.png)

---

## 인사이트 & 한계
- **인사이트**  
  - 결측치 패턴을 “정보”로 활용하면 성능 개선  
  - Tabular 딥러닝 모델은 효과 낮고, 트리 기반+앙상블이 여전히 강력  
- **한계**  
  - Colab 메모리 제한으로 전수 학습/CV 불가  
  - 일부 고결측 변수 제거 → 정보 손실 가능  

---

## Notebooks
- [Baseline XGBoost](./notebooks/baseline_xgb.ipynb)  
- [Preprocessing - Missing Features](./notebooks/preprocess_missing_features.ipynb)  
- [Preprocessing Overview](./notebooks/preprocess_overview.ipynb)  
- [Missing Mechanism Analysis](./notebooks/missing_mechanism_analysis.ipynb)  
- [Scaling: Log + Standard](./notebooks/scaling_log_standard.ipynb)  
- [Train & Evaluation (20k split)](./notebooks/train_eval_20k.ipynb)  
- [TabNet Experiments](./notebooks/tabnet_experiments.ipynb)  
- [NODE Experiments](./notebooks/node_experiments.ipynb)  
- [TabNet Dependency Notes](./notebooks/tabnet_dependency_notes.rst)  
