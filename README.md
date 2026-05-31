# Hotel Booking Cancellation Analysis

## 1. Project Overview

Predict hotel reservation cancellations and analyze customer behavior patterns.

The project follows an end-to-end data analytics pipeline:

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Data Preprocessing
* Classification Modeling
* Customer Clustering
* Model Evaluation

Main objective:

* Predict booking cancellations
* Identify customer behavior patterns
* Propose business strategies for hotel revenue optimization

---

## 2. Dataset

Dataset source:
[Hotel Booking Demand Dataset (Kaggle)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

Dataset characteristics:

* Numerical + categorical features
* Missing values
* Outliers
* Highly skewed lead_time distribution

---

## 3. Pipeline

```text
[Phase 1 — Clustering-based Approach]
Preprocessing
→ Feature Engineering
→ Stratified Sampling
→ Encoding & Scaling
→ KMeans Clustering
→ Customer Segmentation & Behavioral Analysis
→ Limitation Analysis

[Phase 2 — Integrated Tree-based Modeling]
Full Dataset
→ Random Forest / Boosting
→ Threshold Optimization
→ Feature Importance Analysis
→ Final Evaluation
```

---

## 4. Preprocessing

Main preprocessing steps:

* Missing value handling
* ADR outlier removal
* Peak season feature engineering
* lead_time_group generation
* Domestic/foreign customer conversion
* Family type categorization
* Stratified sampling (30,000 rows)
* One-Hot Encoding
* Robust Scaling

Generated features:

* peak_season
* lead_time_group
* family_type
* room_mismatch
* has_waiting_list
* has_parking
* is_domestic

Additional EDA script:

- preprocessing/peak_season.py  
  Used to analyze seasonal booking patterns and define peak/off-peak season criteria.

---

## 5. Clustering

Clustering method:

* KMeans Clustering

Analysis methods:

* Elbow Method
* Silhouette Score
* PCA Visualization

Customer clusters:

* Low-risk
* Long-stay
* High-risk

---

## 6. Classification

Models used:

* Logistic Regression
* Decision Tree
* Random Forest
* Bagging
* AdaBoost
* Gradient Boosting
* XGBoost
* Soft Voting Ensemble

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Stratified 5-Fold Cross Validation

---

## 7. Limitation Analysis & Model Pivot

### Phase 1 Results — Clustering

| Cluster | Cancellation Rate | Gap from Overall Avg (27%) |
|---------|------------------|---------------------------|
| Low-risk | ~29% | +2%p |
| Long-stay | ~35% | +8%p |
| High-risk | ~42% | +15%p |

Although K-Means successfully segmented customers into 3 behavioral groups,
the cancellation rate gap between clusters was too narrow to serve as a
reliable boundary for separate model training.

Training individual classifiers per cluster risks learning ambiguous
decision boundaries, leading to degraded generalization.

### Phase 2 — Model Pivot

To overcome this limitation, the strategy was shifted from
cluster-wise separate training to a **single unified model**
trained on the full dataset.

Tree-based classifiers (Random Forest, Boosting series) were applied,
resulting in significant performance improvement:

| Metric | Phase 1 (Cluster-wise RF) | Phase 2 (Unified RF) |
|--------|--------------------------|----------------------|
| Precision | ~0.65 | **0.724** |
| ROC-AUC | ~0.85 | **0.879** |

---

## 8. Evaluation

Classification evaluation:

* Stratified 5-Fold CV
* Model comparison
* Feature importance analysis

Clustering evaluation:

* Inertia
* Silhouette Score
* PCA-based visualization

---

## 9. Project Structure

```
hotel-booking-cancellation-analysis/
├── dataset/
│   └── hotel_bookings.csv
├── preprocessing/
│   ├── preprocessing.py       # Full preprocessing pipeline
│   └── peak_season.py         # Seasonal pattern analysis
├── phase1_clustering/
│   ├── clustering.py          # Phase 1: KMeans segmentation
│   └── limitation_analysis.py     # Phase 1: Cluster-wise classification
│                              #   → Limitation found here
├── phase2_modeling/
│   ├── final.ipynb            # Phase 2: Unified tree-based model (Main)
│   └── hotel_analyzer.py     # Open Source SW Contribution
├── results/
│   ├── lead_time_feature_engineering.png
│   └── clustering_plot.png
├── hotel_analyzer.py          # Open Source SW Contribution (see Section 10)
├── requirements.txt
└── README.md
```

---

## 10. How to Run

### Phase 1 — Clustering-based Approach
```bash
python preprocessing/preprocessing.py
python phase1_clustering/clustering.py
python phase1_clustering/classification.py
```

### Phase 2 — Unified Tree-based Model (Main)
Open and run `phase2_modeling/final.ipynb` in order.

**Open Source SW Contribution (hotel_analyzer.py):**
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

`hotel_analyzer.py` is a single top-level function designed in the style of
scikit-learn's API. It runs the full end-to-end pipeline in one call:
preprocessing → exploratory clustering → tree-based classification → top-N ranking.

| Parameter | Description |
|-----------|-------------|
| `scalers` | List of scaling methods: `"robust"`, `"standard"`, `"minmax"` |
| `encoders` | List of encoding methods: `"onehot"`, `"label"` |
| `models` | List of models: `"rf"`, `"xgb"`, `"gbm"`, `"adaboost"`, `"bagging"`, `"soft_voting"`, `"lr"`, `"dt"` |
| `cv` | Number of Stratified K-Fold splits (default: 5) |
| `top_n` | Number of top combinations to return (default: 5) |

**Best combination from our run:**

| Rank | Scaler | Encoder | Model | Mean F1 |
|------|--------|---------|-------|---------|
| #1 | robust | onehot | xgb | 70.71% |
| #2 | standard | onehot | xgb | 70.71% |
| #3 | robust | onehot | gbm | 70.66% |
| #4 | standard | onehot | gbm | 70.66% |
| #5 | robust | onehot | soft_voting | 69.87% |

---

## Top 5 Model Combinations (hotel_analyzer.py 실행 결과)

| Rank | Scaler | Encoder | Model | Mean F1 (across clusters) |
|------|--------|---------|-------|--------------------------|
| #1 | robust | onehot | XGBoost | 70.71% |
| #2 | standard | onehot | XGBoost | 70.71% |
| #3 | robust | onehot | Gradient Boosting | 70.66% |
| #4 | standard | onehot | Gradient Boosting | 70.66% |
| #5 | robust | onehot | Soft Voting | 69.87% |

---
## Project Overview
The project initially explored customer segmentation through unsupervised clustering techniques. However, due to overlapping cancellation patterns between clusters, the final solution transitioned toward a unified tree-based supervised learning framework for more robust cancellation prediction.

![Lead Time](results/lead_time_feature_engineering.png)

![Clustering](results/clustering_plot.png)
