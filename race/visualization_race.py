import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Daha önce hesapladığımız Confusion Matrix değerleri (Senin çıktılarından alındı)
cm_base = np.array([[147, 68],
                    [987, 3002]])

cm_fair = np.array([[145, 70],
                    [1342, 2647]])

# ---------------------------------------------------------
# 1. GRAFİK: CONFUSION MATRIX ISI HARİTALARI (HEATMAPS)
# ---------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Adil Olmayan Model Isı Haritası
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Reds', ax=axes[0],
            cbar=False, annot_kws={"size": 14})
axes[0].set_title('Adil Olmayan Model (Baseline)\nAccuracy: %74.90', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Tahmin Edilen (Predicted)', fontsize=12)
axes[0].set_ylabel('Gerçek (Actual)', fontsize=12)
axes[0].set_xticklabels(['Kalan (0)', 'Geçen (1)'])
axes[0].set_yticklabels(['Kalan (0)', 'Geçen (1)'])

# Adil Model Isı Haritası
sns.heatmap(cm_fair, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            cbar=False, annot_kws={"size": 14})
axes[1].set_title('Karşıolgusal Adil Model (Fair)\nAccuracy: %66.41', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Tahmin Edilen (Predicted)', fontsize=12)
axes[1].set_ylabel('Gerçek (Actual)', fontsize=12)
axes[1].set_xticklabels(['Kalan (0)', 'Geçen (1)'])
axes[1].set_yticklabels(['Kalan (0)', 'Geçen (1)'])

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 2. GRAFİK: ADALET VERGİSİ (COST OF FAIRNESS) BAR GRAFİĞİ
# ---------------------------------------------------------
labels = ['Adil Olmayan (Baseline)', 'Adil (Fair Model)']
accuracy_scores = [74.90, 66.41]
cc_scores = [40.20, 100.00]

x = np.arange(len(labels))  # Etiketlerin X eksenindeki konumları
width = 0.35  # Sütun genişliği

fig2, ax = plt.subplots(figsize=(10, 6))

# Sütunları çizdiriyoruz
rects1 = ax.bar(x - width/2, accuracy_scores, width, label='Doğruluk (Accuracy) %', color='#2c3e50')
rects2 = ax.bar(x + width/2, cc_scores, width, label='Karşıolgusal Tutarlılık (CC) %', color='#e74c3c')

# Eksen ve Başlık Ayarları
ax.set_ylabel('Yüzde (%)', fontsize=12, fontweight='bold')
ax.set_title('Modeller Arası Performans ve Adalet Ödünleşimi (Trade-off)', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12, loc='upper left')

# Sütunların üzerine rakamları yazdırmak için fonksiyon
def autolabel(rects):
	for rect in rects:
		height = rect.get_height()
		ax.annotate(f'{height}%',
		            xy=(rect.get_x() + rect.get_width() / 2, height),
		            xytext=(0, 3),  # 3 points vertical offset
		            textcoords="offset points",
		            ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.show()