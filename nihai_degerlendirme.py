import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
orig_u_path = os.path.join(current_dir, "lsac_with_U_zengin.csv")
cf_path = os.path.join(current_dir, "lsac_counterfactual_sim_zengin.csv")

df_orig = pd.read_csv(orig_u_path)
df_cf = pd.read_csv(cf_path)

# --- 1. MODELLERİN KURULMASI ---

# A) ADİL OLMAYAN MODEL (Tüm gözlemlenebilen gerçek dünya verileri)
X_unfair = sm.add_constant(df_orig[['gender', 'FAM_INC', 'UGPA', 'LSAT', 'TIER', 'DECILE1']])
y = df_orig['Y']
model_unfair = sm.Logit(y, X_unfair).fit(disp=0)

# B) NEDENSEL ADİL MODEL (Sadece 5 adet saf, arındırılmış U değişkeni)
X_fair = sm.add_constant(df_orig[['U_FAMINC', 'U_LSAT', 'U_UGPA', 'U_TIER', 'U_DEC']])
model_fair = sm.Logit(y, X_fair).fit(disp=0)

# Orijinal Tahminler
preds_unfair = (model_unfair.predict(X_unfair) > 0.5).astype(int)
preds_fair = (model_fair.predict(X_fair) > 0.5).astype(int)

# Doğruluk Skorları
acc_unfair = accuracy_score(y, preds_unfair)
acc_fair = accuracy_score(y, preds_fair)


# --- 2. KARŞIOLGUSAL SİMÜLASYON İLE FLIP ORANI TESTİ ---

# Adil Olmayan Modele, zincirleme değişmiş yeni CF verilerini veriyoruz
X_unfair_cf = sm.add_constant(df_cf[['gender', 'FAM_INC_cf', 'UGPA_cf', 'LSAT_cf', 'TIER_cf', 'DECILE1_cf']])
X_unfair_cf.columns = ['const', 'gender', 'FAM_INC', 'UGPA', 'LSAT', 'TIER', 'DECILE1']
preds_unfair_cf = (model_unfair.predict(X_unfair_cf) > 0.5).astype(int)

# Adil modelin girdileri (U'lar) doğası gereği değişmediği için CF tahminleri sabittir
preds_fair_cf = preds_fair

# Flip Hesaplaması
flip_unfair = np.mean(preds_unfair != preds_unfair_cf) * 100
flip_fair = np.mean(preds_fair != preds_fair_cf) * 100

print("\n=== ZENGİNLEŞTİRİLMİŞ DAG: NİHAİ ADALET RAPORU ===")
print(f"Adil Olmayan Model Doğruluğu  : % {acc_unfair*100:.2f} | Flip Oranı: % {flip_unfair:.2f}")
print(f"Nedensel (Adil) Model Doğruluğu: % {acc_fair*100:.2f} | Flip Oranı: % {flip_fair:.2f}")