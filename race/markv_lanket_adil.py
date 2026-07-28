"""
ADİL TARAF için IV/Maliyet analizi.
Aynı mantık, ama ham (kirli) özellikler yerine A'dan bağımsız U'lar kullanılıyor.

Önemli: Maliyet, VERİYİ TOPLAMANIN maliyetidir — U'yu hesaplamak (regresyon artığı
almak) ek bir maliyet getirmiyor, o yüzden her U, kendi ham karşılığıyla AYNI maliyeti taşıyor.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scm_race import df_clean

# Aynı maliyetler, sadece U_... sütunlarına eşlenmiş
cost_series = pd.Series({
	'U_LSAT': 1.0,
	'U_UGPA': 1.0,
	'U_TIER': 2.0,
	'U_FAMINC': 8.0,
})
features = ['U_LSAT', 'U_UGPA', 'U_TIER', 'U_FAMINC']
X_candidates = df_clean[features]
y = df_clean['Y']

mi_scores = mutual_info_classif(X_candidates, y, random_state=42)
iv_series = pd.Series(mi_scores, index=features)
print("--- ADİL TARAF: Information Value (Mutual Information) skorları ---")
print(iv_series.round(4))
print()

lambda_values = np.linspace(0.001, 0.01, 10)
survival_history = []
for lam in lambda_values:
	j_scores = iv_series - (lam * cost_series)
	survivors = j_scores[j_scores > 0].index.tolist()
	survival_history.append({'Lambda': round(lam, 4), 'Kalan_Degiskenler': survivors})

df_history = pd.DataFrame(survival_history)
print("--- ADİL TARAF: Maliyet-Duyarlı Budama Senaryoları ---")
print(df_history.to_string(index=False))