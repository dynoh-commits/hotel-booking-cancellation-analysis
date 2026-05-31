import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# =========================================================
# 1. 데이터 로드
# =========================================================
df = pd.read_csv("hotel_bookings_sampling.csv")
print(f"데이터 shape: {df.shape}")

# =========================================================
# 2. 클러스터링 피처 선택
#    - previous_cancellations 제외: 극단 이상치 클러스터 유발
#    - is_canceled 제외: 타깃 변수는 클러스터링에 사용 안 함
#      (나중에 클러스터별 취소율 분석에만 활용)
# =========================================================
cluster_features = [
    'adr',
    'stays_in_weekend_nights',
    'stays_in_week_nights',
    'total_of_special_requests',
    'peak_season',
    'is_domestic',
    'has_waiting_list',
    'has_parking',
    'room_mismatch',
]

X_cluster = df[cluster_features].copy()
scaler    = RobustScaler()
X_scaled  = scaler.fit_transform(X_cluster)

# =========================================================
# 3. Elbow Method + Silhouette Score → 최적 K 선정
# =========================================================
print("\n===== Elbow Method + Silhouette Score =====")
k_range    = range(2, 9)
inertias   = []
sil_scores = []

for k in k_range:
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil    = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)
    sil_scores.append(sil)
    print(f"  K={k}: Inertia={km.inertia_:.0f}, Silhouette={sil:.4f}")

best_k = k_range.start + sil_scores.index(max(sil_scores))
print(f"\n최적 K: {best_k} (Silhouette={max(sil_scores):.4f})")

# =========================================================
# 4. K=3로 최종 클러스터링
#    가장 높은 Silhouette Score와 안정적인 클러스터 분포를 고려
# =========================================================
FINAL_K = 3
km_final   = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

# =========================================================
# 5. 클러스터 레이블 지정
# =========================================================
# 취소율 낮은 순으로 레이블 매핑
cancel_rank = (df.groupby('cluster')['is_canceled']
                 .mean().sort_values().index.tolist())
label_map = {
    cancel_rank[0]: 'Low-risk',       
    cancel_rank[1]: 'Long-stay',         
    cancel_rank[2]: 'High-risk',    
}
df['cluster_label'] = df['cluster'].map(label_map)

print("\n===== 클러스터별 프로파일 =====")
profile_cols = cluster_features + ['is_canceled', 'previous_cancellations']
profile_cols = [c for c in profile_cols if c in df.columns]
profile = df.groupby('cluster_label')[profile_cols].mean().round(3)
profile['count'] = df.groupby('cluster_label').size()
print(profile)

# =========================================================
# 6. PCA 2D 시각화용 좌표
# =========================================================
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]
print(f"\nPCA 설명력: {pca.explained_variance_ratio_.sum():.3f}")

# =========================================================
# 7. Feature Relationship Analysis
# =========================================================
print("\n===== Feature Relationship Analysis =====")
corr_cols = cluster_features + ['is_repeated_guest', 'previous_cancellations']
corr_cols = [c for c in corr_cols if c in df.columns]
corr = (df[corr_cols + ['is_canceled']]
          .corr()['is_canceled']
          .drop('is_canceled')
          .sort_values())

for feat, val in corr.items():
    direction = '취소↑' if val > 0 else '취소↓'
    print(f"  {feat:<30}: {val:>7.4f}  {direction}")

# =========================================================
# 8. 시각화
# =========================================================
colors = ['#4C72B0', '#55A868', '#C44E52']
label_order = list(label_map.values())

fig = plt.figure(figsize=(24, 16))
gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    height_ratios=[1, 1.2, 1.2],
    width_ratios=[1, 1, 1.4],
    hspace=0.55,
    wspace=0.45,
    left=0.05,
    right=0.98,
    top=0.92,
    bottom=0.06
)

# ── (행1-좌) Elbow + Silhouette ───────────────────────────
ax1   = fig.add_subplot(gs[0, 0])
ax1_r = ax1.twinx()
ax1.plot(list(k_range), inertias,   'b-o', linewidth=2, label='Inertia')
ax1_r.plot(list(k_range), sil_scores, 'r-s', linewidth=2, label='Silhouette')
ax1.axvline(x=FINAL_K, color='green', linestyle='--',
            linewidth=1.5, label=f'Final K={FINAL_K}')
ax1.set_xlabel('K'); ax1.set_ylabel('Inertia', color='blue')
ax1_r.set_ylabel('Silhouette Score', color='red')
ax1.set_title('Elbow Method + Silhouette\n→ K Selection', fontweight='bold')
lines1, lbl1 = ax1.get_legend_handles_labels()
lines2, lbl2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines1+lines2, lbl1+lbl2, fontsize=8)
ax1.yaxis.grid(True, alpha=0.4); ax1.set_axisbelow(True)

# ── (행1-중) PCA 2D Scatter ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
for i, label in enumerate(label_order):
    mask = df['cluster_label'] == label
    ax2.scatter(df.loc[mask, 'pca1'], df.loc[mask, 'pca2'],
                s=5, alpha=0.3, label=label.replace('\n',' '),
                color=colors[i])
ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax2.set_title('Customer Segments (PCA 2D)', fontweight='bold')
ax2.legend(fontsize=9, markerscale=4, loc='upper right', frameon=True)
ax2.yaxis.grid(True, alpha=0.4); ax2.set_axisbelow(True)

# ── (행1-우) 클러스터별 취소율 ───────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
cancel_by = (df.groupby('cluster_label')['is_canceled']
               .mean().reindex(label_order))
bars3 = ax3.barh(cancel_by.index, cancel_by.values,
                 color=colors, alpha=0.85)
for bar, val in zip(bars3, cancel_by.values):
    ax3.text(val+0.005, bar.get_y()+bar.get_height()/2,
             f'{val:.1%}', va='center', fontsize=10, fontweight='bold')
ax3.set_xlabel('Cancellation Rate')
ax3.set_title('Cancellation Rate by Cluster', fontweight='bold')
ax3.set_xlim(0, 0.70)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f'{v:.0%}'))
ax3.xaxis.grid(True, alpha=0.4); ax3.set_axisbelow(True)

# ── (행2-좌) ADR 비교 ────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
adr_vals = df.groupby('cluster_label')['adr'].mean().reindex(label_order)
bars4 = ax4.barh(adr_vals.index, adr_vals.values, color=colors, alpha=0.85)
for bar, val in zip(bars4, adr_vals.values):
    ax4.text(val+0.5, bar.get_y()+bar.get_height()/2,
             f'{val:.1f}', va='center', fontsize=9, fontweight='bold')
ax4.set_title('Cluster Profile — ADR', fontweight='bold')
ax4.set_xlabel('Average Daily Rate')
ax4.xaxis.grid(True, alpha=0.4); ax4.set_axisbelow(True)

# ── (행2-중) Special Requests 비교 ───────────────────────
ax5 = fig.add_subplot(gs[1, 1])
sr_vals = df.groupby('cluster_label')['total_of_special_requests'].mean().reindex(label_order)
bars5 = ax5.barh(sr_vals.index, sr_vals.values, color=colors, alpha=0.85)
for bar, val in zip(bars5, sr_vals.values):
    ax5.text(val+0.01, bar.get_y()+bar.get_height()/2,
             f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
ax5.set_title('Cluster Profile — Special Requests', fontweight='bold')
ax5.set_xlabel('Avg Special Requests')
ax5.xaxis.grid(True, alpha=0.4); ax5.set_axisbelow(True)

# ── (행2-우) 클러스터 요약 테이블 ────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_anchor('C')
ax6.axis('off')
tbl_cols  = ['adr', 'total_of_special_requests', 'peak_season',
             'is_domestic', 'is_canceled']
tbl_cols  = [c for c in tbl_cols if c in df.columns]
summary   = df.groupby('cluster_label')[tbl_cols].mean().round(2).reindex(label_order)
summary['count'] = df.groupby('cluster_label').size().reindex(label_order)
summary   = summary.reset_index()
summary = summary.rename(columns={
    'cluster_label': 'Cluster',
    'total_of_special_requests': 'SpecialReq',
    'peak_season': 'PeakSeason',
    'is_domestic': 'Domestic',
    'is_canceled': 'CancelRate'
})
tbl = ax6.table(
    cellText=summary.values,
    colLabels=summary.columns,
    loc='center',
    cellLoc='center',
    bbox=[0, 0, 1, 1]
)
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.4, 3.0)
ax6.set_title('Cluster Summary Table', fontweight='bold', pad=10)

# ── (행3) 상관계수 bar ───────────────────────────
ax7 = fig.add_subplot(gs[2, :])
bar_colors = ['#DD8452' if v > 0 else '#4C72B0' for v in corr.values]
bars7 = ax7.barh(corr.index, corr.values, color=bar_colors, alpha=0.85)
for bar, val in zip(bars7, corr.values):
    xpos = val+0.005 if val >= 0 else val-0.005
    ha   = 'left' if val >= 0 else 'right'
    ax7.text(xpos, bar.get_y()+bar.get_height()/2,
             f'{val:.3f}', va='center', ha=ha,
             fontsize=9, fontweight='bold')
ax7.axvline(x=0, color='black', linewidth=0.8)
ax7.set_xlabel('Correlation with is_canceled', fontsize=10)
ax7.set_title(
    'Feature Relationship with Cancellation\n'
    '(Orange: Higher values → More cancellations / Blue: Higher values → Fewer cancellations)',
    fontweight='bold')
ax7.xaxis.grid(True, alpha=0.4); ax7.set_axisbelow(True)

fig.suptitle('Customer Clustering & Customer Behavior Analysis',
             fontsize=15, fontweight='bold', y=0.97)
plt.show()

# =========================================================
# 9. Feature Relationship Summary
# =========================================================
print("\n===== Feature Relationship Summary =====")
print("취소율 증가 요인 (양의 상관):")
for feat, val in corr[corr > 0.05].sort_values(ascending=False).items():
    print(f"  {feat:<30}: +{val:.4f}")
print("\n취소율 감소 요인 (음의 상관):")
for feat, val in corr[corr < -0.05].items():
    print(f"  {feat:<30}: {val:.4f}")
