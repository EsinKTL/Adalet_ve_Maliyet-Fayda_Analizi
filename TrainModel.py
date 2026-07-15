import os
import pandas as pd
import statsmodels.api as sm
import numpy as np

# 1. Dosya yolunu otomatik ve dinamik olarak buluyoruz (Türkçe karakter hatası yaşanmaz)
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "lsac_clean.csv")

# Veriyi yüklüyoruz
try:
    df = pd.read_csv(file_path)
    print(f"Veri başarıyla yüklendi! Satır sayısı: {len(df)}")
except FileNotFoundError:
    # Eğer dosya adında gizli .csv varsa alternatif olarak dener
    file_path_alt = os.path.join(current_dir, "lsac_clean.csv.csv")
    df = pd.read_csv(file_path_alt)
    print(f"Veri (çift uzantılı) başarıyla yüklendi! Satır sayısı: {len(df)}")

# --- GÖREV 1 & 2: ABDUCTION (GELİŞMİŞ SCM VE GİZLİ 'U' HESAPLAMALARI) ---

# A) UGPA ve LSAT tahmini (Girdiler: Cinsiyet + Aile Geliri)
X_base = sm.add_constant(df[['gender', 'FAM_INC']])

model_ugpa = sm.OLS(df['UGPA'], X_base).fit()
df['U_UGPA'] = model_ugpa.resid

model_lsat = sm.OLS(df['LSAT'], X_base).fit()
df['U_LSAT'] = model_lsat.resid

# B) Fakülte Kalitesi (TIER) tahmini (Girdiler: Aile Geliri + UGPA + LSAT)
X_tier = sm.add_constant(df[['FAM_INC', 'UGPA', 'LSAT']])
model_tier = sm.OLS(df['TIER'], X_tier).fit()
df['U_TIER'] = model_tier.resid

# Doğrulama Testi (Korelasyonlar 0 olmalı)
print("\n--- Gelişmiş Model Doğrulama Testleri ---")
print("U_UGPA ile gender korelasyonu:", df['U_UGPA'].corr(df['gender']))
print("U_LSAT ile gender korelasyonu:", df['U_LSAT'].corr(df['gender']))
print("U_TIER ile gender korelasyonu:", df['U_TIER'].corr(df['gender']))

# Dosyayı kaydet
df.to_csv(os.path.join(current_dir, "lsac_with_U_gelismis.csv"), index=False)
print("\nGelişmiş U değişkenleri eklendi ve 'lsac_with_U_gelismis.csv' olarak kaydedildi!")


# --- GÖREV 3: MÜDAHALE (INTERVENTION) VE KARŞIOLGUSAL ÜRETİM ---

# Cinsiyeti tersine çeviriyoruz
df_counterfactual = df.copy()
df_counterfactual['gender'] = 1 - df_counterfactual['gender']

# Yeni cinsiyete göre UGPA ve LSAT simülasyonu
X_base_cf = sm.add_constant(df_counterfactual[['gender', 'FAM_INC']])
df_counterfactual['UGPA_counterfactual'] = model_ugpa.predict(X_base_cf) + df['U_UGPA']
df_counterfactual['LSAT_counterfactual'] = model_lsat.predict(X_base_cf) + df['U_LSAT']

# Zincirleme etkiyle yeni TIER simülasyonu
X_tier_cf = sm.add_constant(df_counterfactual[['FAM_INC', 'UGPA_counterfactual', 'LSAT_counterfactual']])
X_tier_cf.columns = ['const', 'FAM_INC', 'UGPA', 'LSAT'] # İsim eşitleme
df_counterfactual['TIER_counterfactual'] = model_tier.predict(X_tier_cf) + df['U_TIER']

df_counterfactual.to_csv(os.path.join(current_dir, "lsac_counterfactual_sim_gelismis.csv"), index=False)
print("Zincirleme karşıolgusal simülasyon tamamlandı ve kaydedildi!\n")


# --- GÖREV 4: GEÇİCİ MODELLER İLE FLIP ORANI VE SAPMA TESTLERİ ---

# Adil Olmayan Model
X_unfair_orig = sm.add_constant(df[['gender', 'FAM_INC', 'UGPA', 'LSAT', 'TIER']])
model_unfair = sm.OLS(df['Y'], X_unfair_orig).fit()

# Adil Model
X_fair_orig = sm.add_constant(df[['FAM_INC', 'U_UGPA', 'U_LSAT', 'U_TIER']])
model_fair = sm.OLS(df['Y'], X_fair_orig).fit()

# Tahminler (Kararlar)
pred_unfair_orig = (model_unfair.predict(X_unfair_orig) > 0.5).astype(int)
pred_fair_orig = (model_fair.predict(X_fair_orig) > 0.5).astype(int)

# Karşıolgusal Tahminler
X_unfair_cf = sm.add_constant(df_counterfactual[['gender', 'FAM_INC', 'UGPA_counterfactual', 'LSAT_counterfactual', 'TIER_counterfactual']])
X_unfair_cf.columns = ['const', 'gender', 'FAM_INC', 'UGPA', 'LSAT', 'TIER']
pred_unfair_cf = (model_unfair.predict(X_unfair_cf) > 0.5).astype(int)

pred_fair_cf = pred_fair_orig

# Flip Oranları
flip_rate_unfair = np.mean(pred_unfair_orig != pred_unfair_cf) * 100
flip_rate_fair = np.mean(pred_fair_orig != pred_fair_cf) * 100

print("--- FLIP ORANI (KARŞIOLGUSAL ADALET) SONUÇLARI ---")
print(f"Adil Olmayan Gelişmiş Modelin Flip Oranı : % {flip_rate_unfair:.2f}")
print(f"Nedensel (Adil) Gelişmiş Modelin Flip Oranı: % {flip_rate_fair:.2f}")

# Saf Skor Sapmaları (Gizli Ayrımcılığı Gösteren Kısım)
raw_unfair_orig = model_unfair.predict(X_unfair_orig)
raw_unfair_cf = model_unfair.predict(X_unfair_cf)
raw_fair_orig = model_fair.predict(X_fair_orig)
raw_fair_cf = raw_fair_orig 

score_diff_unfair = np.mean(np.abs(raw_unfair_orig - raw_unfair_cf))
score_diff_fair = np.mean(np.abs(raw_fair_orig - raw_fair_cf))

print("\n--- HAM PUAN DEĞİŞİMİ (SKOR SAPMASI) ---")
print(f"Adil Olmayan Modelin Ortalama Skor Sapması : {score_diff_unfair:.4f}")
print(f"Nedensel (Adil) Modelin Ortalama Skor Sapması: {score_diff_fair:.4f}")