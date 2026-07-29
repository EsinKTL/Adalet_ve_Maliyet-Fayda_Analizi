import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. VERIYI YUKLEME ---
current_dir = os.path.dirname(os.path.abspath(__file__))
clean_path = os.path.join(current_dir, "lsac_clean.csv")
df = pd.read_csv(clean_path)

print("="*50)
print("   LSAC TEMIZ VERI SETI: GENEL OZET RAPORU")
print("="*50)

# --- 2. TERMINAL ICIN TEMEL ISTATISTIKLER VE ORANLAR ---
toplam_ogrenci = len(df)
print(f"Toplam Ogrenci Sayisi: {toplam_ogrenci}\n")

print("--- Temel Ozelliklerin Ortalamalari ---")
print(f"Ortalama LSAT Puani : {df['LSAT'].mean():.2f}")
print(f"Ortalama UGPA (Not) : {df['UGPA'].mean():.2f}")
print(f"Ortalama Aile Geliri: {df['FAM_INC'].mean():.2f} (1-5 Olcegi)")
print(f"Ortalama Okul Kademesi: {df['TIER'].mean():.2f} (1-6 Olcegi)\n")

print("--- Baro Basari (Y) Orani ---")
basari_oranlari = df['Y'].value_counts(normalize=True) * 100
print(f"Baroyu Gecenler (1): % {basari_oranlari.get(1, 0):.2f}")
print(f"Barodan Kalanlar (0): % {basari_oranlari.get(0, 0):.2f}\n")

print("--- Aile Geliri (FAM_INC) Dagilimi ---")
gelir_oranlari = df['FAM_INC'].value_counts(normalize=True).sort_index() * 100
for gelir_seviyesi, oran in gelir_oranlari.items():
    print(f"Gelir Dilimi {gelir_seviyesi}: % {oran:.2f}")
print("="*50)


# --- 3. GORSELLESTIRME: GENEL VERI SETI DAGILIM GRAFIKLERI ---
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('LSAC Temiz Veri Seti (lsac_clean.csv) Genel Ozellik Dagilimlari', fontsize=18, fontweight='bold', y=0.98)

# Renk paleti (Kurumsal ve nötr bir mavi tonu)
base_color = "#34495e"

# 1. LSAT Dağılımı (Histogram)
sns.histplot(df['LSAT'], bins=20, kde=True, color=base_color, ax=axes[0, 0])
axes[0, 0].set_title('LSAT Puan Dagilimi', fontweight='bold')
axes[0, 0].set_xlabel('LSAT Puani')
axes[0, 0].set_ylabel('Ogrenci Sayisi')

# 2. UGPA Dağılımı (Histogram)
sns.histplot(df['UGPA'], bins=20, kde=True, color=base_color, ax=axes[0, 1])
axes[0, 1].set_title('Lisans Not Ortalamasi (UGPA)', fontweight='bold')
axes[0, 1].set_xlabel('UGPA')
axes[0, 1].set_ylabel('')

# 3. Baro Başarısı Y (Bar Çizimi - Uyarılar düzeltildi)
sns.countplot(x='Y', data=df, hue='Y', palette=[base_color, "#2ecc71"], legend=False, ax=axes[0, 2])
axes[0, 2].set_title('Baro Sinavi Basarisi (Y)', fontweight='bold')
axes[0, 2].set_xticks([0, 1])
axes[0, 2].set_xticklabels(['Kalan (0)', 'Gecen (1)'])
axes[0, 2].set_xlabel('')
axes[0, 2].set_ylabel('')

# 4. Aile Geliri Dağılımı (FAM_INC)
sns.countplot(x='FAM_INC', data=df, color=base_color, ax=axes[1, 0])
axes[1, 0].set_title('Aile Gelir Dilimleri (FAM_INC)', fontweight='bold')
axes[1, 0].set_xlabel('Gelir Seviyesi (1: En Dusuk, 5: En Yuksek)')
axes[1, 0].set_ylabel('Ogrenci Sayisi')

# 5. Fakülte Kademesi (TIER)
sns.countplot(x='TIER', data=df, color=base_color, ax=axes[1, 1])
axes[1, 1].set_title('Fakulte Prestij Kademesi (TIER)', fontweight='bold')
axes[1, 1].set_xlabel('Kademe (1: Dusuk, 6: Yuksek)')
axes[1, 1].set_ylabel('')

# 6. Demografik Dağılım (Sadece Cinsiyet olarak güncellendi)
demo_df = pd.DataFrame({
    'Kategori': ['Kadin (0)', 'Erkek (1)'],
    'Yuzde': [
        (df['gender'] == 0).mean() * 100,
        (df['gender'] == 1).mean() * 100
    ]
})
sns.barplot(x='Yuzde', y='Kategori', data=demo_df, hue='Kategori', palette=["#95a5a6", "#7f8c8d"], legend=False, ax=axes[1, 2])
axes[1, 2].set_title('Veri Seti Cinsiyet Dagilimi (%)', fontweight='bold')
axes[1, 2].set_xlabel('Yuzde (%)')
axes[1, 2].set_ylabel('')

# Düzenleme ve Kaydetme
plt.tight_layout(rect=[0, 0, 1, 0.95])
grafik_yolu = os.path.join(current_dir, "Genel_Veri_Seti_Dagilimi.png")
plt.savefig(grafik_yolu, dpi=300)
print(f"\nGenel veri seti dagilim grafigi '{grafik_yolu}' adiyla klasore kaydedildi.")
plt.show()