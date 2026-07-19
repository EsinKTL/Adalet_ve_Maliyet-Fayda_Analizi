import os
import pandas as pd
import statsmodels.api as sm

# Dosya yolunu otomatik bul
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "lsac_clean.csv")

df = pd.read_csv(file_path)
print(f"Veri başarıyla yüklendi! Satır sayısı: {len(df)}")

# --- GÖREV 1: ABDUCTION (5 DENKLEM İÇİN 'U' HESAPLAMALARI) ---

# 1. FAM_INC = f(Gender)
model_fam = sm.OLS(df['FAM_INC'], sm.add_constant(df['gender'])).fit()
df['U_FAMINC'] = model_fam.resid

# 2 & 3. UGPA ve LSAT = f(Gender, FAM_INC)
X_base = sm.add_constant(df[['gender', 'FAM_INC']])
model_ugpa = sm.OLS(df['UGPA'], X_base).fit()
df['U_UGPA'] = model_ugpa.resid

model_lsat = sm.OLS(df['LSAT'], X_base).fit()
df['U_LSAT'] = model_lsat.resid

# 4. TIER = f(LSAT, UGPA, FAM_INC)
X_tier = sm.add_constant(df[['FAM_INC', 'UGPA', 'LSAT']])
model_tier = sm.OLS(df['TIER'], X_tier).fit()
df['U_TIER'] = model_tier.resid

# 5. DECILE1 = f(LSAT, UGPA, TIER)  -> Dikkat: A (Gender) veya FAM_INC direkt girmiyor!
X_dec = sm.add_constant(df[['LSAT', 'UGPA', 'TIER']])
model_dec = sm.OLS(df['DECILE1'], X_dec).fit()
df['U_DEC'] = model_dec.resid

# 5 Sütunlu Tam U Matrisini Kaydet
df.to_csv(os.path.join(current_dir, "lsac_with_U_zengin.csv"), index=False)
print("5 Değişkenli Zenginleştirilmiş U Matrisi kaydedildi!")


# --- GÖREV 2: INTERVENTION (ZİNCİRLEME KARŞIOLGUSAL SİMÜLASYON) ---

df_cf = df.copy()

# A) Cinsiyeti değiştir
df_cf['gender'] = 1 - df_cf['gender']

# B) Yeni Cinsiyet -> Yeni Aile Geliri (FAM_INC)
df_cf['FAM_INC_cf'] = model_fam.predict(sm.add_constant(df_cf['gender'])) + df['U_FAMINC']

# C) Yeni Cinsiyet + Yeni FAM_INC -> Yeni LSAT ve UGPA
X_cf_base = sm.add_constant(df_cf[['gender', 'FAM_INC_cf']])
X_cf_base.columns = ['const', 'gender', 'FAM_INC'] # İsimleri modelin beklediği hale getiriyoruz
df_cf['UGPA_cf'] = model_ugpa.predict(X_cf_base) + df['U_UGPA']
df_cf['LSAT_cf'] = model_lsat.predict(X_cf_base) + df['U_LSAT']

# D) Yeni FAM_INC + Yeni LSAT/UGPA -> Yeni Fakülte Kalitesi (TIER)
X_cf_tier = sm.add_constant(df_cf[['FAM_INC_cf', 'UGPA_cf', 'LSAT_cf']])
X_cf_tier.columns = ['const', 'FAM_INC', 'UGPA', 'LSAT']
df_cf['TIER_cf'] = model_tier.predict(X_cf_tier) + df['U_TIER']

# E) Yeni LSAT/UGPA + Yeni TIER -> Yeni Fakülte Başarısı (DECILE1)
X_cf_dec = sm.add_constant(df_cf[['LSAT_cf', 'UGPA_cf', 'TIER_cf']])
X_cf_dec.columns = ['const', 'LSAT', 'UGPA', 'TIER']
df_cf['DECILE1_cf'] = model_dec.predict(X_cf_dec) + df['U_DEC']

# Tamamlanan zinciri kaydet
df_cf.to_csv(os.path.join(current_dir, "lsac_counterfactual_sim_zengin.csv"), index=False)
print("Tam Zincirleme Karşıolgusal Simülasyon tamamlandı ve kaydedildi!\n")