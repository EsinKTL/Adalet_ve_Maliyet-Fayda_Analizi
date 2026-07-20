import numpy as np
import statsmodels.api as sm
import pandas as pd

"""
nedensel çıkarım (causal inference) yapmak, parametrelerin standart hatalarını hesaplamak ve hipotez testleri (t-testi, F-testi) gerçekleştirmektir.

statsmodels, klasik istatistiksel modelleme, ekonometri ve zaman serisi analizi için Python'daki endüstri standartıdır. Bir modeli "kara kutu" olarak eğitmez;
size katsayıların istatistiksel olarak anlamlı olup olmadığını gösteren kapsamlı bir özet tablosu sunar.
"""

df_clean = pd.read_csv('lsac_clean.csv')

m_faminc = sm.OLS(df_clean['FAM_INC'], sm.add_constant(df_clean[['A']])).fit()
print(m_faminc.params)   # A'nın katsayısı negatif çıkmalı (~-0.3 ile -0.5 arası bekliyoruz)

# FAM_INC = const + A × (katsayı) + U_FAMINC

U_FAMINC = m_faminc.resid # = U_FAMINC = df_clean['FAM_INC'] - m_faminc.predict(sm.add_constant(df_clean[['A']]))

''' U_FAMINC = her öğrenci için: "gerçek gelir dilimi" − "kendi ırk grubunun ortalama geliri"

Örnek: Beyaz bir öğrencinin geliri 5 ise → U_FAMINC = 5 - 3.549 = 1.45 (grubunun ortalamasından zengin)
Beyaz olmayan bir öğrencinin geliri 2 ise → U_FAMINC = 2 - 3.077 = -1.08 (grubunun ortalamasından fakir)

İşte bu U_FAMINC, ırktan bağımsız, "kişiye özel" gelir sinyali — Abduction'ın amacı tam olarak bu: her öğrencinin ırkının etkisini çıkarıp, geriye kalan "gerçek/kişisel"
kısmı elde etmek. Bir sonraki adımda (LSAT denklemi), bu U_FAMINC'i kullanmayacağız aslında; FAM_INC'in kendisini LSAT denklemine bir girdi olarak koyacağız,
U_FAMINC'i ise en sona, "adil model"i kurarken kullanacağız.
'''

print(np.corrcoef(U_FAMINC, df_clean['A'])[0,1]) # Bu sayı ~0'a çok yakın çıkmalı (örneğin 0.00X gibi). Çünkü U_FAMINC, tanımı gereği, A'nın açıkladığı kısım çıkarılmış geriye kalan — yani A ile hiç ilişkili olmaması matematiksel bir zorunluluk, bu bir "başarı" değil, regresyonun doğası.

#LSAT = f(A, FAM_INC) + U_LSAT
m_lsat = sm.OLS(df_clean['LSAT'], sm.add_constant(df_clean[['A', 'FAM_INC']])).fit()
print(m_lsat.params)

U_LSAT = m_lsat.resid
print(np.corrcoef(U_LSAT, df_clean['A'])[0,1])
print(np.corrcoef(U_LSAT, df_clean['FAM_INC'])[0,1])

#UGPA = f(A, FAM_INC) + U_UGPA
m_ugpa = sm.OLS(df_clean['UGPA'], sm.add_constant(df_clean[['A', 'FAM_INC']])).fit()
print(m_ugpa.params)

U_UGPA = m_ugpa.resid
print(np.corrcoef(U_UGPA, df_clean['A'])[0,1])
print(np.corrcoef(U_UGPA, df_clean['FAM_INC'])[0,1])

# TIER = f(LSAT, UGPA, FAM_INC) + U_TIER
m_tier = sm.OLS(df_clean['TIER'], sm.add_constant(df_clean[['LSAT', 'UGPA','FAM_INC']])).fit()
print(m_tier.params)

U_TIER = m_tier.resid
print(np.corrcoef(U_TIER, df_clean['LSAT'])[0,1])
print(np.corrcoef(U_TIER, df_clean['UGPA'])[0,1])
print(np.corrcoef(U_TIER, df_clean['FAM_INC'])[0,1])

# DECILE1 = f(LSAT, UGPA, TIER)
m_decile1 = sm.OLS(df_clean['DECILE1'], sm.add_constant(df_clean[['LSAT', 'UGPA','TIER']])).fit()
print(m_decile1.params)

U_DECILE1 = m_decile1.resid
print(np.corrcoef(U_DECILE1, df_clean['LSAT'])[0,1])
print(np.corrcoef(U_DECILE1, df_clean['UGPA'])[0,1])
print(np.corrcoef(U_DECILE1, df_clean['TIER'])[0,1])

# Y = f(DECILE1, FAM_INC)
m_y = sm.OLS(df_clean['Y'], sm.add_constant(df_clean[['DECILE1', 'FAM_INC']])).fit()
print(m_y.params)

U_Y = m_y.resid
print(np.corrcoef(U_Y, df_clean['LSAT'])[0,1])
print(np.corrcoef(U_Y, df_clean['UGPA'])[0,1])
print(np.corrcoef(U_Y, df_clean['TIER'])[0,1])