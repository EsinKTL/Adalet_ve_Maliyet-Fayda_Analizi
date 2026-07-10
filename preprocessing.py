import pandas as pd
import numpy as np

df = pd.read_csv('bar_pass_prediction.csv')

# Kopya feature silme

silinecek = ["sex", "male", "race", "race2" , "other", "asian", "black", "hisp",
             "gpa","decile1b","parttime","bar_passed","bar1", "bar1_yr", "bar2", "bar2_yr","bar",
             "dnn_bar_pass_prediction","cluster", "DOB_yr", "age","ID",
             "indxgrp2", "index6040", "indxgrp", "zfygpa", "zgpa","decile3", "Dropout"]

for i in silinecek:
	df = df.drop(i, axis=1)
	

# Satır silme null değerlere göre

null_arama = ['lsat', 'ugpa', 'pass_bar', 'gender', 'race1', 'grad', 'fulltime', 'tier', 'fam_inc', 'decile1']

df[null_arama] = df[null_arama].replace(r'^\s*$', np.nan, regex=True)

df_clean = df.dropna(subset=null_arama, how='any')

print(f"Orijinal satır sayısı: {len(df)}")
print(f"Temizlenmiş satır sayısı: {len(df_clean)}")

df_clean['A'] = (df_clean['race1'] != 'white').astype(int) # race white temelli encoding

df_clean = df_clean.rename(columns={'lsat':'LSAT','ugpa':'UGPA','decile1':'DECILE1',
                      'fam_inc':'FAM_INC','tier':'TIER','pass_bar':'Y'})

df_clean['grad'] = df_clean['grad'].map({'Y':1, 'X':0, 'O':0}) # grade encoding
df_clean['gender'] = df_clean['gender'].map({'male':1, 'female':0}) # gender encoding
df_clean['fulltime'] = df_clean['fulltime'].map({1:1, 2:0}) # fulltime encoding

df_clean.to_csv('lsac_clean.csv', index=False)

print("\nA (ırk) dağılımı: 0=white, 1=other")
print(df_clean['A'].value_counts())
print("\nGrup bazında Y (pass_bar) oranı:")
print(df_clean.groupby('A')['Y'].mean())
print("\nGrup bazında ortalama LSAT / UGPA / DECILE1:")
print(df_clean.groupby('A')[['LSAT','UGPA','DECILE1']].mean())

print("\nKaydedildi: lsac_clean.csv, shape:", df_clean.shape)