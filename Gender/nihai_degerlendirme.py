import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

# --- 3. CONFUSION MATRIX HESAPLAMALARI ---
# Matris değerlerini doğrudan modelin kendi çıktısından otomatik alıyoruz
cm_base = confusion_matrix(y, preds_unfair)
cm_fair = confusion_matrix(y, preds_fair)

print("Adil Olmayan Matris:\n", cm_base)
print("Adil Model Matrisi:\n", cm_fair)


# --- 4. GÖRSELLEŞTİRME 1: CONFUSION MATRIX ISI HARİTALARI ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Adil Olmayan Model Isı Haritası
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=False, annot_kws={"size": 14})
axes[0].set_title(f'Adil Olmayan Model (Baseline)\nAccuracy: %{acc_unfair*100:.2f}', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tahmin Edilen (Predicted)', fontsize=12)
axes[0].set_ylabel('Gerçek (Actual)', fontsize=12)
axes[0].set_xticklabels(['Kalan (0)', 'Geçen (1)'])
axes[0].set_yticklabels(['Kalan (0)', 'Geçen (1)'])

# Adil Model Isı Haritası
sns.heatmap(cm_fair, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False, annot_kws={"size": 14})
axes[1].set_title(f'Karşıolgusal Adil Model (Fair)\nAccuracy: %{acc_fair*100:.2f}', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tahmin Edilen (Predicted)', fontsize=12)
axes[1].set_ylabel('Gerçek (Actual)', fontsize=12)
axes[1].set_xticklabels(['Kalan (0)', 'Geçen (1)'])
axes[1].set_yticklabels(['Kalan (0)', 'Geçen (1)'])

plt.tight_layout()
plt.show()


# --- 5. GÖRSELLEŞTİRME 2: ADALET VERGİSİ (TRADE-OFF) BAR GRAFİĞİ ---
labels = ['Adil Olmayan (Baseline)', 'Adil (Fair Model)']

# Rakamları terminalden çekip otomatik yuvarlıyoruz
accuracy_scores = [round(acc_unfair * 100, 2), round(acc_fair * 100, 2)]
cc_scores = [round(100 - flip_unfair, 2), round(100 - flip_fair, 2)]

x = np.arange(len(labels))
width = 0.35

fig2, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Doğruluk (Accuracy) %', color='#2c3e50')
rects2 = ax.bar(x + width/2, cc_scores, width, label='Karşıolgusal Tutarlılık (CC) %', color='#e74c3c')

ax.set_ylabel('Yüzde (%)', fontsize=12, fontweight='bold')
ax.set_title('Modeller Arası Performans ve Adalet Ödünleşimi (Trade-off)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12, loc='lower right')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

ax.set_ylim(0, 115)
plt.tight_layout()
plt.show()