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
EDA
→ Peak/Off-peak season analysis
→ Feature Engineering
→ Stratified Sampling
→ Encoding & Scaling
→ Classification
→ Clustering
→ Evaluation
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

## 5. Classification

Models used:

* Logistic Regression
* Decision Tree
* Random Forest
* Bagging
* AdaBoost
* Gradient Boosting
* XGBoost

Evaluation metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Stratified 5-Fold Cross Validation

---

## 6. Clustering

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

## 7. Evaluation

Classification evaluation:

* Stratified 5-Fold CV
* Model comparison
* Feature importance analysis

Clustering evaluation:

* Inertia
* Silhouette Score
* PCA-based visualization

---

## 8. Project Structure
```text
- dataset/
- preprocessing/
- modeling/
- results/
```

---

## 9. How to Run

1. Run preprocessing/preprocessing.py
2. Run modeling/classification.py
3. Run modeling/clustering.py


![Lead Time](results/lead_time_feature_engineering.png)

![Clustering](results/clustering_plot.png)
