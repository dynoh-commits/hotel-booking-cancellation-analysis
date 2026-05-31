# Hotel Booking Cancellation Analysis

## 1. Project Overview

Predict hotel reservation cancellations using a two-phase analytical approach: unsupervised customer segmentation followed by supervised tree-based classification.

The project follows an end-to-end data analytics pipeline:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Customer Clustering (Phase 1)
- Limitation Analysis & Model Pivot
- Classification Modeling (Phase 2)
- Model Evaluation

Main objectives:

- Predict booking cancellations with high precision
- Identify customer behavior patterns through clustering
- Propose business strategies for hotel revenue optimization

### Research Flow & Model Transition

The project initially explored customer segmentation through unsupervised K-Means clustering. However, the cancellation rate variance across clusters (29% / 35% / 42%) was too narrow to serve as a reliable training boundary. To overcome this limitation, the strategy was pivoted to a unified tree-based supervised learning framework trained on the full dataset, achieving significantly higher performance.

---

## 2. Dataset

Dataset source: [Hotel Booking Demand Dataset (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

| Field | Details |
|---|---|
| Period | 2015–2017 |
| Raw size | 119,390 records / 32 columns |
| Target variable | `is_canceled` (0 = Not Cancelled, 1 = Cancelled) |
| Raw cancellation rate | 37.0% |

Dataset characteristics:

- Numerical + categorical mixed features
- Missing values (`children`, `country`, `agent`, `company`)
- Outliers in `adr`
- Highly right-skewed `lead_time` distribution

---

## 3. Pipeline

```
[Phase 1 — Clustering-based Approach]
Raw Data
→ Preprocessing (outlier removal, feature engineering, stratified sampling)
→ KMeans Clustering (K=3, Silhouette-validated)
→ Customer Segmentation & Behavioral Analysis
→ Limitation Analysis (narrow cancellation rate gap → model pivot)

[Phase 2 — Unified Tree-based Modeling]
Full Preprocessed Dataset
→ Random Forest / Boosting (RF, XGBoost, GBM, AdaBoost)
→ Threshold Optimization
→ Feature Importance Analysis
→ Final Evaluation (Precision 0.724 / ROC-AUC 0.879)
```

---

## 4. Preprocessing

Main preprocessing steps:

- Duplicate removal (119,390 → 87,396)
- `lead_time` outlier filtering (≤ 300 days → 85,110)
- ADR outlier removal (0 ≤ adr ≤ 500)
- Missing value imputation (`children` → 0, `country` → "Unknown")
- Peak season feature engineering
- Dynamic `lead_time_group` generation (season-aware thresholds)
- Domestic/international customer flag (`is_domestic`)
- Family type categorization (`family_type`)
- Binary flags: `room_mismatch`, `has_waiting_list`, `has_parking`
- Stratified sampling (30,000 rows, preserving `lead_time_group` ratio)
- One-Hot Encoding
- Robust Scaling

Engineered features:

| Feature | Description |
|---|---|
| `peak_season` | 1 if arrival month is April–October |
| `lead_time_group` | Impulsive / Planned / Long_plan (dynamic cut-off by season) |
| `is_domestic` | 1 if country of origin is Portugal (PRT) |
| `family_type` | Individual / Couple / Family / Friends |
| `room_mismatch` | 1 if reserved and assigned room types differ |
| `has_waiting_list` | 1 if days on waiting list > 0 |
| `has_parking` | 1 if car parking spaces requested > 0 |
| `has_children` | 1 if children > 0 or babies > 0 |
| `country_group` | Top-10 countries kept; others grouped as "Other" |

---

## 5. Phase 1 — Customer Clustering

Clustering method: KMeans

Optimal K selection:

- Elbow Method (Inertia)
- Silhouette Score → peak at K=3

Customer segments identified:

| Cluster | Label | Cancel Rate | ADR | Key Trait |
|---|---|---|---|---|
| 0 | Practical Individual (Low-risk) | ~29% | ~146 | High engagement, stable bookings |
| 1 | Long-stay Groups (Long-stay) | ~35% | ~95 | Operational buffer, moderate volatility |
| 2 | High-spending VIPs (High-risk) | ~42% | ~164 | Premium spend, high cancellation risk |

Visualization outputs → `results/phase1/`

---

## 6. Phase 1 — Limitation Analysis & Model Pivot

### Clustering Limitation

| Cluster | Cancellation Rate | Gap from Overall Avg (27%) |
|---|---|---|
| Low-risk | ~29% | +2%p |
| Long-stay | ~35% | +8%p |
| High-risk | ~42% | +15%p |

Although K-Means successfully segmented customers into 3 behavioral groups, the cancellation rate gap between clusters was too narrow to serve as a reliable decision boundary for separate model training. Training individual classifiers per cluster risks learning ambiguous boundaries, leading to degraded generalization.

### Phase 2 — Model Pivot

To overcome this limitation, the strategy shifted from cluster-wise separate training to a **single unified model** trained on the full dataset.

| Metric | Phase 1 (Cluster-wise RF) | Phase 2 (Unified RF) |
|---|---|---|
| F1-score | ~0.68 | ~0.71 |
| Precision | ~0.65 | 0.724 |
| ROC-AUC | ~0.85 | 0.879 |

---

## 7. Phase 2 — Classification (Final Solution)

Models evaluated:

- Random Forest (`class_weight=balanced`)
- XGBoost
- Gradient Boosting
- AdaBoost
- Bagging
- Soft Voting Ensemble
- Logistic Regression
- Decision Tree

Evaluation approach:

- Stratified 5-Fold Cross Validation
- Threshold optimization
- Feature importance analysis

### Final Model Performance

Algorithm: Random Forest | `class_weight=balanced` | `n_estimators=300` | `threshold=0.5`

| Metric | Not Cancelled | Cancelled |
|---|---|---|
| Precision | 0.89 | **0.724** |
| Recall | 0.81 | 0.568 |
| F1-Score | 0.85 | 0.637 |
| ROC-AUC | — | **0.879** |

> Precision was adopted as the primary metric because a False Positive (predicting cancellation for a guest who actually shows up) carries significant business cost in the hotel domain.

### Key Findings

| Finding | Detail |
|---|---|
| Lead Time (Rank #1 feature) | Cancellation rate rises monotonically with lead_time (0–300 days) |
| Domestic vs. International | Domestic guests cancel at 34% vs. 24% for international guests |
| Impulsive Bookings (0–14 days) | Cancellation rate only ~11%, well below the 27% average |
| Booking Channel (Agent) | Agent-mediated reservations show a distinct cancellation profile |
| Seasonality | Peak season carries slightly higher cancellation rate |

Visualization outputs → `results/phase2/`

---

## 8. Evaluation

Classification evaluation:

- Stratified 5-Fold CV
- Model comparison table
- Feature importance analysis (Bagging vs RF, AdaBoost vs GBM vs XGBoost)

Clustering evaluation:

- Inertia (Elbow Method)
- Silhouette Score
- PCA-based 2D visualization

---

## 9. Project Structure

```
hotel-booking-cancellation-analysis/
├── dataset/
│   └── hotel_bookings.csv
├── preprocessing/
│   ├── preprocessing.py       # Full preprocessing pipeline
│   │                          #   Outputs: hotel_bookings_pre_eda.csv
│   │                          #            hotel_bookings_sampling.csv  (→ clustering.py input)
│   │                          #            hotel_bookings_preprocessed.csv  (→ limitation_analysis.py input)
│   └── peak_season.py         # EDA: seasonal booking pattern analysis & peak/off-peak definition
├── phase1_clustering/
│   ├── clustering.py          # Phase 1: KMeans segmentation + visualization
│   │                          #   (Elbow, Silhouette, PCA scatter, cancellation rate, correlation charts)
│   │                          #   Input: hotel_bookings_sampling.csv
│   └── limitation_analysis.py # Phase 1: Cluster-wise classification + feature importance visualization
│                              #   (Pure Bagging vs RF, AdaBoost vs GBM vs XGBoost per cluster)
│                              #   → Limitation confirmed → triggers Phase 2 pivot
│                              #   Input: hotel_bookings_preprocessed.csv
├── phase2_modeling/
│   ├── final.ipynb            # Phase 2: Unified tree-based model (Main Solution)
│   │                          #   Input: hotel_bookings.csv (full pipeline inside)
│   └── hotel_analyzer.py      # Open Source SW Contribution (scikit-learn-style API)
├── results/
│   ├── phase1/                # Visualization outputs from phase1 scripts
│   │   ├── cancellation_patterns.png
│   │   ├── clustering_analysis.png
│   │   ├── lead_time_feature_engineering.png
│   │   ├── peak_season_analysis.png
│   │   ├── sampling_distribution_preservation.png
│   │   ├── bagging_rf_feature_importance.png
│   │   └── boosting_feature_importance.png
│   └── phase2/                # Visualization outputs from final.ipynb
│       ├── data_preprocessing_steps.png
│       ├── lead_time_distribution.png
│       ├── final_model_performance.png
│       ├── feature_importance_random_forest.png
│       ├── actual_vs_predicted_cancellation_rate_by_segment.png
│       ├── elbow_method.png
│       └── k-means_clustering.png
├── requirements.txt
└── README.md
```

---

## 10. How to Run

### Phase 1 — Clustering-based Approach

```bash
# Step 1: Preprocessing (generates intermediate CSV files)
python preprocessing/preprocessing.py

# Step 2: KMeans clustering + visualization
python phase1_clustering/clustering.py

# Step 3: Cluster-wise classification + limitation analysis
python phase1_clustering/limitation_analysis.py
```

### Phase 2 — Unified Tree-based Model (Main Solution)

Open and run `phase2_modeling/final.ipynb` in order.

### Open Source SW Contribution (`hotel_analyzer.py`)

```python
from hotel_analyzer import hotel_analyzer

results = hotel_analyzer(
    csv_path="hotel_bookings.csv",
    scalers=["robust", "standard", "minmax"],
    encoders=["onehot"],
    models=["rf", "xgb", "adaboost", "gbm", "bagging", "soft_voting", "lr", "dt"],
    cv=5,
    top_n=5,
    verbose=True
)

print(results["best_combination"])
```

---

## 11. Open Source SW Contribution

`hotel_analyzer.py` is a single top-level function designed in the style of scikit-learn's API. It runs the full end-to-end pipeline in one call: preprocessing → clustering → classification → top-N ranking.

| Parameter | Description |
|---|---|
| `scalers` | List of scaling methods: `"robust"`, `"standard"`, `"minmax"` |
| `encoders` | List of encoding methods: `"onehot"`, `"label"` |
| `models` | List of models: `"rf"`, `"xgb"`, `"gbm"`, `"adaboost"`, `"bagging"`, `"soft_voting"`, `"lr"`, `"dt"` |
| `cv` | Number of Stratified K-Fold splits (default: 5) |
| `top_n` | Number of top combinations to return (default: 5) |

**Best combination from our run:**

| Rank | Scaler | Encoder | Model | Mean F1 |
|---|---|---|---|---|
| #1 | robust | onehot | xgb | 70.71% |
| #2 | standard | onehot | xgb | 70.71% |
| #3 | robust | onehot | gbm | 70.66% |
| #4 | standard | onehot | gbm | 70.66% |
| #5 | robust | onehot | soft\_voting | 69.87% |
