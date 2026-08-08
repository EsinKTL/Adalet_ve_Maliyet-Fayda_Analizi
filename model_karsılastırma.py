import pandas as pd, numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

d = pd.read_csv('race/lsac_clean.csv').copy()
d['race_A']  = d['A']
d['gender_G'] = d['gender']
print(f"Temizlenmiş veri kullanılıyor: {len(d)} öğrenci, {d.shape[1]} sütun\n")


def eksen_degerlendir(veri, a_kolonu, eksen_adi, seed=0):

	v = veri.copy()
	A = a_kolonu

	m_lsat = sm.OLS(v['LSAT'], sm.add_constant(v[[A]])).fit()
	m_ugpa = sm.OLS(v['UGPA'], sm.add_constant(v[[A]])).fit()
	m_dec  = sm.OLS(v['DECILE1'], sm.add_constant(v[['LSAT','UGPA',A]])).fit()

	v['U_LSAT'] = m_lsat.resid
	v['U_UGPA'] = m_ugpa.resid
	v['U_DEC']  = m_dec.resid
	
	def karsiolgusal(row, a_yeni):
		lsat_cf = m_lsat.params['const'] + m_lsat.params[A]*a_yeni + row['U_LSAT']
		ugpa_cf = m_ugpa.params['const'] + m_ugpa.params[A]*a_yeni + row['U_UGPA']
		dec_cf  = (m_dec.params['const'] + m_dec.params['LSAT']*lsat_cf
		           + m_dec.params['UGPA']*ugpa_cf + m_dec.params[A]*a_yeni + row['U_DEC'])
		return lsat_cf, ugpa_cf, dec_cf
	
	tr, te = train_test_split(v, test_size=0.3, random_state=seed, stratify=v['Y'])
	unfair = LogisticRegression(max_iter=1000, class_weight='balanced').fit(
		tr[['LSAT','UGPA','DECILE1',A]], tr['Y'])
	fair = LogisticRegression(max_iter=1000, class_weight='balanced').fit(
		tr[['U_LSAT','U_UGPA','U_DEC']], tr['Y'])
	
	acc_u = unfair.score(te[['LSAT','UGPA','DECILE1',A]], te['Y'])
	acc_f = fair.score(te[['U_LSAT','U_UGPA','U_DEC']], te['Y'])

	def flip_orani(model, veri_test, tur):
		flips = 0
		for idx, row in veri_test.iterrows():
			lsat_cf, ugpa_cf, dec_cf = karsiolgusal(row, 1 - int(row[A]))
			if tur == 'unfair':
				orijinal = model.predict([[row['LSAT'], row['UGPA'], row['DECILE1'], row[A]]])[0]
				cf = model.predict([[lsat_cf, ugpa_cf, dec_cf, 1 - int(row[A])]])[0]
			else:
				orijinal = model.predict([[row['U_LSAT'], row['U_UGPA'], row['U_DEC']]])[0]
				cf = orijinal
			if orijinal != cf:
				flips += 1
		return flips / len(veri_test)
	
	fr_u = flip_orani(unfair, te, 'unfair')
	fr_f = flip_orani(fair, te, 'fair')
	
	return {
		"Eksen": eksen_adi,
		"Model": "Adil OLMAYAN",
		"Doğruluk": round(acc_u, 3),
		"Flip Oranı": round(fr_u, 3),
	}, {
		"Eksen": eksen_adi,
		"Model": "ADİL (U tabanlı)",
		"Doğruluk": round(acc_f, 3),
		"Flip Oranı": round(fr_f, 3),
	}

sonuclar = []
for satir in eksen_degerlendir(d, 'race_A', 'race'):
	sonuclar.append(satir)
for satir in eksen_degerlendir(d, 'gender_G', 'gender'):
	sonuclar.append(satir)

tablo = pd.DataFrame(sonuclar)
print("=" * 60)
print("NİHAİ KARŞILAŞTIRMA TABLOSU")
print("=" * 60)
print(tablo.to_string(index=False))

tablo.to_csv("karsilastirma_tablosu.csv", index=False)
print("\nKaydedildi: karsilastirma_tablosu.csv")