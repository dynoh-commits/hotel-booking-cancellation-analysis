import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("hotel_bookings.csv")

month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

monthly = (df.groupby('arrival_date_month')
             .agg(count=('is_canceled','count'),
                  avg_adr=('adr','mean'),
                  avg_lead_time=('lead_time','mean'))
             .reindex(month_order))

# 성수기 구분 (4~10월)
peak_months = ['April','May','June','July','August','September','October']
colors = ['#FF6B6B' if m in peak_months else '#4A90D9'
          for m in month_order]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── 예약 건수 ──
axes[0].bar(month_order, monthly['count'], color=colors)
axes[0].set_title('Booking Count by Month')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(month_order, rotation=45, ha='right')

# ── 평균 ADR ──
axes[1].bar(month_order, monthly['avg_adr'], color=colors)
axes[1].set_title('Average ADR by Month')
axes[1].set_ylabel('ADR')
axes[1].set_xticklabels(month_order, rotation=45, ha='right')

# ── 평균 lead_time ──
axes[2].bar(month_order, monthly['avg_lead_time'], color=colors)
axes[2].set_title('Average Lead Time by Month')
axes[2].set_ylabel('Mean Lead Time')
axes[2].set_xticklabels(month_order, rotation=45, ha='right')

# 범례
from matplotlib.patches import Patch
legend = [Patch(color='#FF6B6B', label='Peak (Apr–Oct)'),
          Patch(color='#4A90D9', label='Off-peak (Nov–Mar)')]
fig.legend(handles=legend, loc='upper right', fontsize=10)

fig.suptitle('Peak vs Off-peak Season Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()