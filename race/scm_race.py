import numpy as np
import statsmodels.api as sm
import pandas as pd

"""
nedensel çıkarım (causal inference) yapmak, parametrelerin standart hatalarını hesaplamak ve hipotez testleri (t-testi, F-testi) gerçekleştirmektir.

statsmodels, klasik istatistiksel modelleme, ekonometri ve zaman serisi analizi için Python'daki endüstri standartıdır. Bir modeli "kara kutu" olarak eğitmez;
size katsayıların istatistiksel olarak animport numpy as np
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




'''
1. FAM_INC_cf = m_faminc.params['const'] + m_faminc.params['A']×a_yeni + U_FAMINC(bu öğrencinin)

2. LSAT_cf = m_lsat.params['const'] + m_lsat.params['A']×a_yeni
             + m_lsat.params['FAM_INC']×FAM_INC_cf + U_LSAT(bu öğrencinin)

3. UGPA_cf = m_ugpa.params['const'] + m_ugpa.params['A']×a_yeni
             + m_ugpa.params['FAM_INC']×FAM_INC_cf + U_UGPA(bu öğrencinin)

4. TIER_cf = m_tier.params['const'] + m_tier.params['LSAT']×LSAT_cf
             + m_tier.params['UGPA']×UGPA_cf + m_tier.params['FAM_INC']×FAM_INC_cf + U_TIER(bu öğrencinin)

5. DECILE1_cf = m_decile1.params['const'] + m_decile1.params['LSAT']×LSAT_cf
                + m_decile1.params['UGPA']×UGPA_cf + m_decile1.params['TIER']×TIER_cf + U_DECILE1(bu öğrencinin)
'''

def karsiolgusal_uret(row, a_yeni):
	faminc_cf = m_faminc.params['const'] + m_faminc.params['A']*a_yeni + row['U_FAMINC']
	lsat_cf   = m_lsat.params['const'] + m_lsat.params['A']*a_yeni + m_lsat.params['FAM_INC']*faminc_cf + row['U_LSAT']
	ugpa_cf   = m_ugpa.params['const'] + m_ugpa.params['A']*a_yeni + m_ugpa.params['FAM_INC']*faminc_cf + row['U_UGPA']
	tier_cf   = m_tier.params['const'] + m_tier.params['LSAT']*lsat_cf + m_tier.params['UGPA']*ugpa_cf + m_tier.params['FAM_INC']*faminc_cf + row['U_TIER']
	decile1_cf = m_decile1.params['const'] + m_decile1.params['LSAT']*lsat_cf + m_decile1.params['UGPA']*ugpa_cf + m_decile1.params['TIER']*tier_cf + row['U_DECILE1']
	return faminc_cf, lsat_cf, ugpa_cf, tier_cf, decile1_cf

df_clean['U_FAMINC'] = m_faminc.resid
df_clean['U_LSAT'] = m_lsat.resid
df_clean['U_UGPA'] = m_ugpa.resid
df_clean['U_TIER'] = m_tier.resid
df_clean['U_DECILE1'] = m_decile1.resid

ornek = df_clean.iloc[0]
print("Gerçek A:", ornek['A'], "Gerçek FAM_INC:", ornek['FAM_INC'])
sonuc = karsiolgusal_uret(ornek, 1 - ornek['A'])
print("Karşıolgusal:", sonuc)

kf_sonuclari = df_clean.apply(lambda row: karsiolgusal_uret(row, 1 - row['A']), axis=1)

# AŞAMA 3: KARŞIOLGUSAL (COUNTERFACTUAL) ÜRETİM
# Abduction adımında bulduğumuz U (artık) değerlerini sabit tutuyoruz.

# 1. Müdahale (Intervention): A'yı tamamen tersine çevir.
# (Eğer A ikiliyse (0 ve 1), 1-A işlemi 0'ı 1, 1'i 0 yapar.)
df_clean['A_cf'] = 1 - df_clean['A']

# 2. Zincirleme İleri Besleme (Forward Pass)
# Her adımda bir önceki adımın karşıolgusal sonucunu ve orijinal U'yu kullanıyoruz.

# FAM_INC_cf = f(A_cf) + U_FAMINC
X_faminc_cf = sm.add_constant(df_clean[['A_cf']].rename(columns={'A_cf': 'A'}), has_constant='add')
df_clean['FAM_INC_cf'] = m_faminc.predict(X_faminc_cf) + U_FAMINC

# LSAT_cf = f(A_cf, FAM_INC_cf) + U_LSAT
X_lsat_cf = sm.add_constant(pd.DataFrame({
	'A': df_clean['A_cf'],
	'FAM_INC': df_clean['FAM_INC_cf']
}), has_constant='add')
df_clean['LSAT_cf'] = m_lsat.predict(X_lsat_cf) + U_LSAT

# UGPA_cf = f(A_cf, FAM_INC_cf) + U_UGPA
X_ugpa_cf = sm.add_constant(pd.DataFrame({
	'A': df_clean['A_cf'],
	'FAM_INC': df_clean['FAM_INC_cf']
}), has_constant='add')
df_clean['UGPA_cf'] = m_ugpa.predict(X_ugpa_cf) + U_UGPA

# TIER_cf = f(LSAT_cf, UGPA_cf, FAM_INC_cf) + U_TIER
X_tier_cf = sm.add_constant(pd.DataFrame({
	'LSAT': df_clean['LSAT_cf'],
	'UGPA': df_clean['UGPA_cf'],
	'FAM_INC': df_clean['FAM_INC_cf']
}), has_constant='add')
df_clean['TIER_cf'] = m_tier.predict(X_tier_cf) + U_TIER

# DECILE1_cf = f(LSAT_cf, UGPA_cf, TIER_cf) + U_DECILE1
X_decile1_cf = sm.add_constant(pd.DataFrame({
	'LSAT': df_clean['LSAT_cf'],
	'UGPA': df_clean['UGPA_cf'],
	'TIER': df_clean['TIER_cf']
}), has_constant='add')
df_clean['DECILE1_cf'] = m_decile1.predict(X_decile1_cf) + U_DECILE1

# Sütunların veri setine doğru eklendiğini kontrol et
print("\n--- Karşıolgusal Üretim Tamamlandı ---")
print(df_clean[['A', 'A_cf', 'FAM_INC', 'FAM_INC_cf', 'LSAT', 'LSAT_cf']].head())
lamlı olup olmadığını gösteren kapsamlı bir özet tablosu sunar.
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
