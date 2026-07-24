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

import matplotlib.pyplot as plt
import seaborn as sns

# --- ADIM 4: SONUÇLARIN PROFESYONEL OLARAK GÖRSELLEŞTİRİLMESİ ---

# Grafiğin stilini daha profesyonel (akademik) bir temaya ayarlıyoruz
sns.set_theme(style="whitegrid")

# Çizilecek verileri hazırlıyoruz
modeller = ['Adil Olmayan (Klasik) Model', 'Nedensel (Adil) Model']
dogruluk_oranlari = [acc_unfair * 100, acc_fair * 100]
flip_oranlari = [flip_unfair, flip_fair]

x = np.arange(len(modeller))  # X eksenindeki etiketlerin konumu
genislik = 0.35  # Sütun genişliği

# Şekil ve eksenleri oluşturuyoruz
fig, ax = plt.subplots(figsize=(10, 6))

# Doğruluk Sütunları (Yeşil tonları)
sutun1 = ax.bar(x - genislik/2, dogruluk_oranlari, genislik, label='Doğruluk / Accuracy (%)', color='#2ca02c', edgecolor='black')

# Flip Oranı Sütunları (Kırmızı tonları - Tehlike/Adaletsizlik vurgusu)
sutun2 = ax.bar(x + genislik/2, flip_oranlari, genislik, label='Flip Oranı / Adaletsizlik (%)', color='#d62728', edgecolor='black')

# Grafiğe başlık ve etiketler ekliyoruz
ax.set_ylabel('Yüzde (%)', fontsize=12, fontweight='bold')
ax.set_title('Model Performansı ve Karşıolgusal Adalet Karşılaştırması', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(modeller, fontsize=12)
ax.legend(fontsize=11)

# Sütunların tam üzerine değerleri sayısal olarak yazdırıyoruz
ax.bar_label(sutun1, padding=3, fmt='%.2f', fontsize=11, fontweight='bold')
ax.bar_label(sutun2, padding=3, fmt='%.2f', fontsize=11, fontweight='bold')

# Y eksenini 0 ile 105 arasına sıkıştırıyoruz ki yazılar grafiğin üstünden taşmasın
ax.set_ylim(0, 110)

# Çıktıyı klasöre yüksek çözünürlüklü olarak kaydediyoruz
grafik_yolu = os.path.join(current_dir, "Model_Karsilastirma_Grafikleri.png")
fig.tight_layout()
plt.savefig(grafik_yolu, format="PNG", dpi=300)

print(f"\nProfesyonel performans grafiği '{grafik_yolu}' adıyla klasöre kaydedildi!")

# Grafiği ekranda göster
plt.show()