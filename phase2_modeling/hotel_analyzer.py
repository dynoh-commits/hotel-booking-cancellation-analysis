# -*- coding: utf-8 -*-
"""
hotel_analyzer.py
=================
A single top-level function that runs the full end-to-end hotel booking
cancellation analysis pipeline: preprocessing → clustering → classification.

Designed in the style of scikit-learn's API.

Usage
-----
    from hotel_analyzer import hotel_analyzer

    results = hotel_analyzer(
        csv_path="hotel_bookings.csv",
        scalers=["robust", "standard"],
        encoders=["onehot"],
        models=["rf", "xgb", "adaboost", "gbm", "bagging", "soft_voting", "lr", "dt"],
        cv=5,
        n_clusters=3,
        top_n=5,
        sample_size=30000,
        random_state=42,
        verbose=True
    )

    print(results["best_combination"])
    print(results["top_combinations"])

Parameters
----------
csv_path : str
    Path to the raw hotel bookings CSV file.
    Expected: Hotel Booking Demand Dataset
    (https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

scalers : list of str, default=["robust", "standard", "minmax"]
    Scaling methods to evaluate.
    Supported values:
        "robust"   → RobustScaler   (resistant to outliers; used for ADR)
        "standard" → StandardScaler (zero mean, unit variance)
        "minmax"   → MinMaxScaler   (scales to [0, 1])

encoders : list of str, default=["onehot"]
    Encoding methods to evaluate for categorical features.
    Supported values:
        "onehot" → One-Hot Encoding via pd.get_dummies (drop_first=True)
        "label"  → Label Encoding via sklearn LabelEncoder

models : list of str, default=["rf", "xgb", "adaboost", "gbm", "bagging", "soft_voting", "lr", "dt"]
    Classification models to evaluate per cluster.
    Supported values:
        "rf"          → RandomForestClassifier      (Bagging ensemble)
        "bagging"     → BaggingClassifier           (Pure Bagging with DecisionTree)
        "soft_voting" → VotingClassifier            (Soft Voting: LR + DT)
        "adaboost"    → AdaBoostClassifier          (Boosting with Decision Stumps)
        "gbm"         → GradientBoostingClassifier  (Gradient Boosting)
        "xgb"         → XGBClassifier               (eXtreme Gradient Boosting)

cv : int, default=5
    Number of folds for Stratified K-Fold Cross Validation.
    Stratification preserves the class ratio (is_canceled) in each fold.

n_clusters : int, default=3
    Number of customer segments for K-Means clustering.
    Optimal K is validated by Silhouette Score.

top_n : int, default=5
    Number of top-performing (scaler, encoder, model) combinations to return,
    ranked by mean F1-Score across all clusters.

sample_size : int, default=30000
    Number of rows to extract via Stratified Sampling on lead_time_group.
    Set to None to use the full dataset.

random_state : int, default=42
    Random seed for reproducibility across all stochastic components.

verbose : bool, default=True
    If True, prints progress logs at each major pipeline stage.

Returns
-------
dict with the following keys:

    "best_combination" : dict
        The single best (scaler, encoder, model) combination with scores.
        Keys: "scaler", "encoder", "model", "f1", "precision", "recall"

    "top_combinations" : list of dict
        Top-N combinations ranked by mean F1-Score.

    "cluster_results" : dict
        Per-cluster results for every combination evaluated.
        Structure: {cluster_id: [{scaler, encoder, model, f1, precision, recall}]}

    "preprocessed_df" : pd.DataFrame
        The final preprocessed and encoded DataFrame used for modeling.

    "clustered_df" : pd.DataFrame
        The sampling-stage DataFrame with cluster labels assigned.

Notes
-----
- Preprocessing follows the team's established pipeline:
    outlier removal → missing value imputation → feature engineering
    (peak_season, lead_time_group, is_domestic, family_type,
     room_mismatch, has_waiting_list, has_parking)
    → stratified sampling → encoding → scaling

- Clustering uses the sampling-stage DataFrame (before OHE/scaling)
  to preserve interpretability of cluster profiles.

- Classification is performed per cluster using Stratified K-Fold CV.

- For label encoding, high-cardinality columns (reserved_room_type,
  assigned_room_type) are encoded with a fitted LabelEncoder per column.

Examples
--------
    # Minimal usage
    results = hotel_analyzer("hotel_bookings.csv")

    # Custom configuration
    results = hotel_analyzer(
        csv_path="hotel_bookings.csv",
        scalers=["robust"],
        encoders=["onehot"],
        models=["xgb", "gbm", "rf"],
        cv=5,
        top_n=3,
        verbose=True
    )

    # Access best result
    best = results["best_combination"]
    print(f"Best model: {best['model']} | F1: {best['f1']:.2f}%")

    # Access top-N table
    import pandas as pd
    top_df = pd.DataFrame(results["top_combinations"])
    print(top_df)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    RobustScaler, StandardScaler, MinMaxScaler, LabelEncoder
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, precision_score, recall_score, silhouette_score
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


# ─────────────────────────────────────────────
# Internal helper: Preprocessing
# ─────────────────────────────────────────────

def _preprocess(df_raw, scaler_name, encoder_name, sample_size, random_state, verbose):
    """
    Internal preprocessing pipeline.
    Returns (df_for_clustering, df_for_modeling).
    df_for_clustering: sampled, before OHE/scaling (used for KMeans)
    df_for_modeling:   sampled, after encoding + scaling (used for classification)
    """
    df = df_raw.copy()

    # 1. Outlier removal
    df = df[(df['adults'] + df['children'].fillna(0) + df['babies']) > 0]
    df = df[(df['adr'] >= 0) & (df['adr'] <= 500)]

    # 2. Missing value imputation
    df['children'] = df['children'].fillna(0)
    df['country']  = df['country'].fillna('Unknown')

    # 3. Drop irrelevant columns
    drop_cols = [
        'arrival_date_week_number', 'arrival_date_day_of_month',
        'meal', 'reservation_status', 'reservation_status_date',
        'agent', 'company', 'distribution_channel',
        'previous_bookings_not_canceled', 'booking_changes',
        'arrival_date_year', 'deposit_type', 'customer_type',
    ]
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 4. Month → integer
    month_map = {
        'January':1,'February':2,'March':3,'April':4,
        'May':5,'June':6,'July':7,'August':8,
        'September':9,'October':10,'November':11,'December':12
    }
    df['arrival_date_month'] = df['arrival_date_month'].map(month_map)

    # 5. Feature engineering: peak_season
    peak_months = [4, 5, 6, 7, 8, 9, 10]
    df['peak_season'] = df['arrival_date_month'].isin(peak_months).astype(int)

    # 6. Feature engineering: lead_time_group (dynamic cut-off by season)
    def classify_lead_time(row):
        lt, peak = row['lead_time'], row['peak_season']
        if lt <= 7:
            return 'Impulsive'
        elif (peak == 1 and lt <= 90) or (peak == 0 and lt <= 30):
            return 'Planned'
        else:
            return 'Long_plan'

    df['lead_time_group'] = df.apply(classify_lead_time, axis=1)

    # 7. is_domestic (Portugal = domestic)
    df['is_domestic'] = (df['country'] == 'PRT').astype(int)

    # 8. family_type
    def classify_family(row):
        has_child = (row['children'] > 0) or (row['babies'] > 0)
        if has_child:           return 'Family'
        elif row['adults'] == 1: return 'Individual'
        elif row['adults'] == 2: return 'Couple'
        elif row['adults'] >= 3: return 'Friends'
        else:                    return 'Unknown'

    df['family_type'] = df.apply(classify_family, axis=1)

    # 9. Binary flags
    df['has_waiting_list'] = (df['days_in_waiting_list'] > 0).astype(int)
    df['has_parking']      = (df['required_car_parking_spaces'] > 0).astype(int)
    df['room_mismatch']    = (df['reserved_room_type'] != df['assigned_room_type']).astype(int)

    # 10. Drop source columns after feature engineering
    drop_after = [
        'country', 'adults', 'children', 'babies',
        'days_in_waiting_list', 'required_car_parking_spaces',
        'lead_time', 'arrival_date_month',
    ]
    df = df.drop(columns=[c for c in drop_after if c in df.columns])

    # 11. Stratified sampling
    if sample_size and len(df) > sample_size:
        df, _ = train_test_split(
            df, train_size=sample_size,
            stratify=df['lead_time_group'],
            random_state=random_state
        )
        df = df.reset_index(drop=True)

    # Save clustering-stage df (before encoding/scaling)
    df_cluster = df.copy()

    # 12. Encoding
    ohe_cols = ['hotel', 'market_segment', 'family_type',
                'reserved_room_type', 'assigned_room_type', 'lead_time_group']

    if encoder_name == 'onehot':
        df = pd.get_dummies(df, columns=[c for c in ohe_cols if c in df.columns], drop_first=True)

    elif encoder_name == 'label':
        for col in [c for c in ohe_cols if c in df.columns]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 13. Scaling
    scale_cols = ['adr', 'stays_in_weekend_nights', 'stays_in_week_nights']
    scale_cols = [c for c in scale_cols if c in df.columns]

    scaler_map = {
        'robust':   RobustScaler(),
        'standard': StandardScaler(),
        'minmax':   MinMaxScaler(),
    }
    scaler_obj = scaler_map[scaler_name]
    df[scale_cols] = scaler_obj.fit_transform(df[scale_cols])

    if verbose:
        print(f"    [Preprocessing] scaler={scaler_name}, encoder={encoder_name} | shape={df.shape}")

    return df_cluster, df


# ─────────────────────────────────────────────
# Internal helper: Build model object
# ─────────────────────────────────────────────

def _build_model(model_name, random_state):
    """Returns a sklearn-compatible classifier for the given model name."""
    if model_name == 'rf':
        return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)

    elif model_name == 'bagging':
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=7, random_state=random_state),
            n_estimators=100, random_state=random_state, n_jobs=-1
        )

    elif model_name == 'soft_voting':
        base = [
            ('lr', LogisticRegression(max_iter=1000, random_state=random_state)),
            ('dt', DecisionTreeClassifier(max_depth=6, random_state=random_state))
        ]
        return VotingClassifier(estimators=base, voting='soft')

    elif model_name == 'adaboost':
        # Decision Stumps (max_depth=1) as base estimators per lecture spec
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=random_state),
            n_estimators=100, random_state=random_state
        )

    elif model_name == 'gbm':
        return GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state
        )

    elif model_name == 'xgb':
        # XGBoost: eXtreme Gradient Boosting with L1/L2 regularization
        return XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            random_state=random_state, eval_metric='logloss'
        )
    
    elif model_name == 'lr':
        return LogisticRegression(max_iter=1000, random_state=random_state)

    elif model_name == 'dt':
        return DecisionTreeClassifier(max_depth=6, random_state=random_state)

    else:
        raise ValueError(f"Unknown model name: '{model_name}'. "
                         f"Supported: rf, bagging, soft_voting, adaboost, gbm, xgb")


# ─────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────

def hotel_analyzer(
    csv_path,
    scalers=None,
    encoders=None,
    models=None,
    cv=5,
    n_clusters=3,
    top_n=5,
    sample_size=30000,
    random_state=42,
    verbose=True
):
    """
    End-to-end hotel booking cancellation analysis pipeline.

    Runs preprocessing, customer clustering (K-Means), and cluster-wise
    classification under Stratified K-Fold Cross Validation for all
    combinations of scalers, encoders, and models.
    Returns the top-N best combinations ranked by mean F1-Score.

    See module docstring for full parameter and return value documentation.
    """

    # Default parameter values
    if scalers is None:
        scalers = ['robust', 'standard', 'minmax']
    if encoders is None:
        encoders = ['onehot']
    if models is None:
        models = ['rf', 'xgb', 'adaboost', 'gbm', 'bagging', 'soft_voting']

    # ── Step 1: Load raw data ──────────────────────────────
    if verbose:
        print("=" * 65)
        print("  hotel_analyzer  |  Hotel Booking Cancellation Pipeline")
        print("=" * 65)
        print(f"\n[Step 1] Loading data from '{csv_path}' ...")

    df_raw = pd.read_csv(csv_path)

    if verbose:
        print(f"  Raw data shape: {df_raw.shape}")

    # ── Step 2: Preprocessing (first pass with robust+onehot for clustering) ─
    if verbose:
        print("\n[Step 2] Preprocessing (for clustering stage) ...")

    df_cluster_base, _ = _preprocess(
        df_raw, scaler_name='robust', encoder_name='onehot',
        sample_size=sample_size, random_state=random_state, verbose=False
    )

    # ── Step 3: K-Means Clustering ────────────────────────
    if verbose:
        print(f"\n[Step 3] K-Means Clustering (n_clusters={n_clusters}) ...")

    cluster_features = [
        'adr', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'total_of_special_requests', 'peak_season', 'is_domestic',
        'has_waiting_list', 'has_parking', 'room_mismatch',
    ]
    cluster_features = [c for c in cluster_features if c in df_cluster_base.columns]

    X_cluster = df_cluster_base[cluster_features].copy()
    cluster_scaler = RobustScaler()
    X_scaled_cluster = cluster_scaler.fit_transform(X_cluster)

    # Validate K with Silhouette Score
    sil_scores = {}
    for k in range(2, min(9, len(df_cluster_base))):
        km_tmp = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels_tmp = km_tmp.fit_predict(X_scaled_cluster)
        sil_scores[k] = silhouette_score(
            X_scaled_cluster, labels_tmp, sample_size=5000, random_state=random_state
        )

    if verbose:
        for k, s in sil_scores.items():
            print(f"    K={k}: Silhouette={s:.4f}")

    km_final = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df_cluster_base['cluster'] = km_final.fit_predict(X_scaled_cluster)

    # Assign human-readable labels by cancellation rate (low → high)
    cancel_rank = (
        df_cluster_base.groupby('cluster')['is_canceled']
        .mean().sort_values().index.tolist()
    )
    label_map = {
        cancel_rank[0]: 'Low-risk',
        cancel_rank[1]: 'Long-stay',
        cancel_rank[2]: 'High-risk',
    }
    df_cluster_base['cluster_label'] = df_cluster_base['cluster'].map(label_map)

    if verbose:
        print(f"\n  Cluster label mapping: {label_map}")
        profile = (df_cluster_base.groupby('cluster_label')[cluster_features + ['is_canceled']]
                   .mean().round(3))
        profile['count'] = df_cluster_base.groupby('cluster_label').size()
        print(profile.to_string())

    # ── Step 4: Classification for all combinations ───────
    if verbose:
        print(f"\n[Step 4] Classification | {len(scalers)} scalers × "
              f"{len(encoders)} encoders × {len(models)} models × {cv}-Fold CV")
        print(f"  Total combinations: {len(scalers) * len(encoders) * len(models)}\n")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    all_results = []
    cluster_results = {label_map[cid]: [] for cid in df_cluster_base['cluster'].unique()}

    combo_idx = 0
    for scaler_name in scalers:
        for encoder_name in encoders:

            # Preprocess for this scaler/encoder combo
            _, df_model = _preprocess(
                df_raw, scaler_name=scaler_name, encoder_name=encoder_name,
                sample_size=sample_size, random_state=random_state, verbose=verbose
            )

            # Align cluster labels from clustering stage
            # (indices match because same sample_size + random_state)
            if len(df_model) == len(df_cluster_base):
                df_model['customer_cluster'] = df_cluster_base['cluster'].values
            else:
                # Fallback: re-cluster on model df
                X_tmp = df_model[[c for c in cluster_features
                                  if c in df_model.columns]].copy()
                df_model['customer_cluster'] = km_final.predict(
                    cluster_scaler.transform(X_tmp)
                )

            for model_name in models:
                combo_idx += 1
                combo_label = f"[{combo_idx}] {scaler_name}+{encoder_name}+{model_name}"

                if verbose:
                    print(f"  {combo_label}")

                cluster_f1s = []

                for cluster_id in sorted(df_model['customer_cluster'].unique()):
                    df_c = df_model[df_model['customer_cluster'] == cluster_id]
                    X_c = df_c.drop(columns=['is_canceled', 'customer_cluster'],
                                    errors='ignore').reset_index(drop=True)
                    y_c = df_c['is_canceled'].reset_index(drop=True)

                    f1_folds, prec_folds, rec_folds = [], [], []

                    for train_idx, val_idx in skf.split(X_c, y_c):
                        X_train = X_c.iloc[train_idx]
                        X_val   = X_c.iloc[val_idx]
                        y_train = y_c.iloc[train_idx]
                        y_val   = y_c.iloc[val_idx]

                        model_obj = _build_model(model_name, random_state)
                        model_obj.fit(X_train, y_train)
                        y_pred = model_obj.predict(X_val)

                        f1_folds.append(f1_score(y_val, y_pred) * 100)
                        prec_folds.append(precision_score(y_val, y_pred) * 100)
                        rec_folds.append(recall_score(y_val, y_pred) * 100)

                    avg_f1   = float(np.mean(f1_folds))
                    avg_prec = float(np.mean(prec_folds))
                    avg_rec  = float(np.mean(rec_folds))

                    cluster_results[label_map[cluster_id]].append({
                        'scaler':    scaler_name,
                        'encoder':   encoder_name,
                        'model':     model_name,
                        'f1':        round(avg_f1, 2),
                        'precision': round(avg_prec, 2),
                        'recall':    round(avg_rec, 2),
                    })
                    cluster_f1s.append(avg_f1)

                    if verbose:
                        print(f"    Cluster {label_map[cluster_id]} ({cluster_id}) → "
                              f"F1={avg_f1:.2f}% | Prec={avg_prec:.2f}% | Rec={avg_rec:.2f}%")

                mean_f1 = float(np.mean(cluster_f1s))
                all_results.append({
                    'scaler':    scaler_name,
                    'encoder':   encoder_name,
                    'model':     model_name,
                    'mean_f1':   round(mean_f1, 2),
                })

                if verbose:
                    print(f"    → Mean F1 across clusters: {mean_f1:.2f}%\n")

    # ── Step 5: Rank and return top-N ─────────────────────
    all_results_sorted = sorted(all_results, key=lambda x: x['mean_f1'], reverse=True)
    top_combinations   = all_results_sorted[:top_n]
    best_combination   = top_combinations[0] if top_combinations else {}

    if verbose:
        print("=" * 65)
        print(f"[Result] Top-{top_n} combinations by mean F1-Score")
        print("=" * 65)
        for rank, combo in enumerate(top_combinations, 1):
            print(f"  #{rank}: {combo['scaler']}+{combo['encoder']}+{combo['model']}"
                  f" → Mean F1={combo['mean_f1']:.2f}%")
        print(f"\n  Best: {best_combination}")

    return {
        'best_combination':  best_combination,
        'top_combinations':  top_combinations,
        'cluster_results':   cluster_results,
        'preprocessed_df':   df_model,
        'clustered_df':      df_cluster_base,
    }


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = hotel_analyzer(
        csv_path="hotel_bookings.csv",
        scalers=["robust", "standard"],
        encoders=["onehot"],
        models=["rf", "xgb", "adaboost", "gbm", "bagging", "soft_voting", "lr", "dt"],
        cv=5,
        n_clusters=3,
        top_n=5,
        sample_size=30000,
        random_state=42,
        verbose=True
    )

    print("\n[Best Combination]")
    print(results["best_combination"])

    print("\n[Top-5 Combinations]")
    top_df = pd.DataFrame(results["top_combinations"])
    print(top_df.to_string(index=False))
