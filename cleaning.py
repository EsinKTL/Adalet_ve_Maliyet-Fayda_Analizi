"""
LSAC Veri Temizleme
Notebook'taki kludge yerine, DAG'ımıza uygun minimal ve net bir temizlik.

Nedensel değişken haritamız (Kusner ve ark. 2017 çerçevesine yakın):
    A (race): LSAT, UGPA          (sistemik yanlılık: hazırlık fırsatı farkı)
    LSAT, UGPA: DECILE1 (1.yıl hukuk fakültesi performansı)
    DECILE1: PASS_BAR (Y)
"""
import pandas as pd
import numpy as np

df = pd.read_csv('bar_pass_prediction.csv')

# Sadece DAG'ımızda ihtiyacımız olan sütunlar
cols = ['race1', 'gender', 'lsat', 'ugpa', 'decile1', 'fam_inc', 'tier', 'pass_bar']
d = df[cols].copy()

# Eksik veri: sadece ilgili sütunlarda satır bazlı temiz veri.
before = d.shape[0]
d = d.dropna(subset=['lsat','ugpa','decile1','pass_bar','fam_inc','tier'])
print(f"Satır: {before} -> {d.shape[0]} (eksik veri temizlendi)")

d['gender'] = d['gender'].map({'male':1, 'female':0})
d = d.dropna(subset=['gender'])

d = d.rename(columns={'lsat':'LSAT','ugpa':'UGPA','decile1':'DECILE1',
                      'fam_inc':'FAM_INC','tier':'TIER','pass_bar':'Y'})

print("\nA (ırk) dağılımı: 0=white, 1=black")
print(d['A'].value_counts())
print("\nGrup bazında Y (pass_bar) oranı:")
print(d.groupby('A')['Y'].mean())
print("\nGrup bazında ortalama LSAT / UGPA / DECILE1:")
print(d.groupby('A')[['LSAT','UGPA','DECILE1']].mean())

d.to_csv('lsac_clean.csv', index=False)
print("\nKaydedildi: lsac_clean.csv, shape:", d.shape)
