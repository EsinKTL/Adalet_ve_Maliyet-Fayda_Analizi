import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scm_race import df_clean

cost_series = pd.Series({'LSAT': 1.0, 'UGPA': 1.0, 'TIER': 2.0, 'FAM_INC': 8.0})
features = ['LSAT', 'UGPA', 'TIER', 'FAM_INC']
X_candidates = df_clean[features]
y = df_clean['Y']

mi_scores = mutual_info_classif(X_candidates, y, random_state=42)
iv_series = pd.Series(mi_scores, index=features)
print("--- Information Value (Mutual Information) skorları ---")
print(iv_series.round(4))
print()

lambda_values = np.linspace(0.001, 0.01, 10)
survival_history = []
for lam in lambda_values:
	j_scores = iv_series - (lam * cost_series)
	survivors = j_scores[j_scores > 0].index.tolist()
	survival_history.append({'Lambda': round(lam, 4), 'Kalan_Degiskenler': survivors})

df_history = pd.DataFrame(survival_history)
print("--- Maliyet-Duyarlı Budama Senaryoları ---")
print(df_history.to_string(index=False))