import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("clean_cat.csv")

#--------------------------------------------------------------------------------------
# คำนวณคะแนนเฉลี่ยของแต่ละ packaging (5 มิติ)
#--------------------------------------------------------------------------------------

pkg_scores = []
for i in range(1, 11):
    avg = {
        'packaging': f'opt{i}',
        'buy':       df[f'opt{i}_buy'].mean(),
        'unique':    df[f'opt{i}_unique'].mean(),
        'premium':   df[f'opt{i}_premium'].mean(),
        'taste':     df[f'opt{i}_taste'].mean(),
        'personal':  df[f'opt{i}_personal'].mean()
    }
    pkg_scores.append(avg)

pkg_df = pd.DataFrame(pkg_scores).set_index('packaging')

print("คะแนนเฉลี่ยแต่ละ packaging:")
print(pkg_df.round(2))

#--------------------------------------------------------------------------------------
# จัดกลุ่มเป็น 3 กลุ่ม
#--------------------------------------------------------------------------------------

kmeans = KMeans(n_clusters=3, random_state=42)
pkg_df['cluster'] = kmeans.fit_predict(pkg_df)

print("\nผลการจัดกลุ่ม:")
for cluster in sorted(pkg_df['cluster'].unique()):
    pkgs = pkg_df[pkg_df['cluster'] == cluster].index.tolist()
    print(f"กลุ่ม {cluster}: {pkgs}")