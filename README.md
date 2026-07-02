# Adalet_ve_Maliyet-Fayda_Analizi

# Yol Haritası ve Çalışma Planı
### Karşıolgusal Adalet ve Maliyet-Fayda Analizi
**Ekip:** 3 kişi · **Süre:** Temmuz 2026 (4 hafta) · **Raporlama:** Haftalık

---

## 0. Projenin Hedefi

Ay sonunda elimizde şunlar olmalı:

1. Veri seti için doğrulanmış bir **nedensel DAG**.
2. Verilere uydurulmuş bir **Yapısal Nedensel Model (SCM)** ve her birey için çıkarsanmış gizli değişkenler **(U)**.
3. **Karşıolgusal olarak adil** bir sınıflandırıcı + karşılaştırma için "adil olmayan" baz model.
4. **Maliyet duyarlı özellik seçimi** (Markov Blanket üzerinden düğüm budama, IV/Maliyet oranı).
5. Adalet–doğruluk–maliyet üçgenini gösteren **değerlendirme raporu** ve tekrar üretilebilir kod deposu.

**Başarı kriteri:** Adil model, cinsiyet/ırk değiştirildiğinde tahminini "flip" etmemeli (karşıolgusal tutarlılık), doğrulukta kabul edilebilir bir kayıpla, seçilen özellikler hem adil hem yüksek IV/Maliyet oranına sahip olmalı.

---

## 1. Teknik Yığın ve Kurulum

| Amaç | Araç |
|---|---|
| Nedensel keşif (PC / FCI / GES) | `causal-learn` |
| Nedensel çıkarım & çürütme (refutation) | `DoWhy` |
| DAG işleme, Markov Blanket, düğüm budama | `networkx`, `pgmpy` |
| SCM / abduction (lineer-Gauss başlangıç) | `numpy`, `statsmodels` (ileri: `pyro`/`numpyro`) |
| Sınıflandırma & baz model | `scikit-learn` |
| Görselleştirme (DAG, sonuçlar) | `graphviz`, `matplotlib` |

> Ortak repo (Git), tek `requirements.txt`/`environment.yml`, `README`, ve her hafta çalıştırılabilir notebook + `src/` modülleri. Kod herkeste aynı ortamda çalışmalı.

## Veri Seti

- **Birincil öneri: Law School (LSAC)** — Karşıolgusal adalet literatürünün (Kusner ve ark., 2017) kanonik veri seti. Hassas öznitelik: ırk/cinsiyet; özellikler: LSAT, lisans notu; hedef: başarı. Metodolojiyle birebir örtüşür.
- **Alternatif: German Credit (UCI)** — belgedeki "kredi onayı" örneğine en yakın olanı.

> Not: Maliyet duyarlı seçim için veri setinde gerçek "özellik edinme maliyeti" yoksa, her özelliğe **sentetik ama gerekçeli bir maliyet** atanır (ör. kolay erişilen alanlar ucuz, dış kaynak gerektirenler pahalı).

---

## 2. 4 Haftalık Yol Haritası

### Hafta 1: Temel, Veri ve Aşama 1 (DAG Keşfi)
- Repo/ortam kurulumu, veri hattı, EDA.
- Hassas öznitelik (A), hedef (Y), özellikler (X) netleştirilir.
- Literatür: Kusner ve ark. 2017 + Pearl SCM/do-calculus özeti.
- **Aşama 1:** PC/FCI ile DAG keşfi + domain knowledge ile okların düzeltilmesi.
- **Rapor 1 çıktısı:** Seçilen veri seti, gerekçeli nedensel DAG, problem çerçevesi.

### Hafta 2: Aşama 2 (SCM + Abduction)
- Yapısal denklemler `f_i` uydurulur (başlangıç: lineer-Gauss / additive noise).
- **Aşama 2:** Her birey için gizli değişken `U` çıkarsanır (`Û = f⁻¹(X, Λ)`).
- SCM doğrulaması (artık analizi, uyum kontrolü).
- **Rapor 2 çıktısı:** Uydurulmuş SCM, çıkarsanmış U, 2-3 örnek üzerinde abduction gösterimi.

### Hafta 3: Aşama 3 (Karşıolgusal Adalet + Maliyet Duyarlı Seçim)
- Karşıolgusallar `X_{A←a'}` üretilir (abduction → action → prediction).
- Adil sınıflandırıcı: A'nın ardılı olmayan değişkenler / U üzerine eğitilir.
- Karşılaştırma için adil olmayan baz model.
- **Maliyet katmanı:** Özellik maliyetleri atanır, Information Value hesaplanır, Markov Blanket üzerinde graf tabanlı düğüm budama uygulanır.
- **Rapor 3 çıktısı:** Karşıolgusal adalet ölçümü + maliyet duyarlı seçilmiş özellik alt kümesi.

### Hafta 4: Değerlendirme, Maliyet-Fayda ve Yazım
- Metrikler: karşıolgusal tutarlılık (A değişince tahmin flip ediyor mu?), doğruluk, IV/Maliyet.
- Adil vs adil olmayan model: adalet–doğruluk–maliyet ödünleşimi.
- DoWhy ile çürütme/robustluk testleri.
- Final rapor, sunum, dokümantasyon, tekrar üretilebilirlik.
- **Rapor 4 / Final çıktısı:** Bütünleşik sonuçlar + sunum.

---

## 3. Rol Dağılımı (3 Kişi)

Roller *birincil sahiplik* içindir; işbirliği ve eşleştirme (pair-work) beklenir. Lider tüm teknik entegrasyonu ve incelemeyi üstlenir.

| Kişi | Birincil sorumluluk |
|---|---|
| **Ben** | Nedensel mimari: DAG tasarımı, SCM yapısal denklemleri, abduction implementasyonu, entegrasyon ve teknik inceleme. İşin matematiksel çekirdeği. |
| **Üye A** | Veri & Keşif: veri hattı, EDA, PC/FCI keşfi, DAG doğrulama/görselleştirme + maliyet & Information Value modeli. |
| **Üye B** | Adalet & Modelleme: karşıolgusal üretim, adil sınıflandırıcı, baz model, değerlendirme metrikleri ve maliyet-fayda analizi. |

---

## 4. Haftalık Rapor Şablonu

> Her hafta aynı formatı kullanın; hocaya/danışmana tutarlı görünür ve ilerleme ölçülebilir olur.

```
Hafta N — Rapor
1. Bu hafta hedeflenen çıktı:
2. Tamamlananlar (kişi bazında):
   - Ben:
   - Üye A:
   - Üye B:
3. Ana bulgu / karar (ör. DAG'de X→Y yönü, seçilen mekanizma):
4. Karşılaşılan sorun ve çözüm:
5. Sonraki haftaya devir / riskler:
6. Ek: grafik, DAG, metrik tablosu
```

---

## 5. Risk ve Tuzaklar (Baştan Bilinsin)

- **Nedensel keşif oynak olur:** PC/FCI sonuçları kırılgandır. Domain knowledge'ı asıl alın, keşfi destekleyici kullanın.
- **Abduction için tersine çevrilebilirlik şart:** Lineer-Gauss/additive noise ile başlayın; U = artık olur, hesap analitik kalır. Karmaşık modele ancak zaman kalırsa geçin.
- **Adaleti "inşa ederek" garantileyin:** Tam karşıolgusal üretim yerine, A'nın ardılı olmayan değişkenler/U üzerine eğitmek en güvenli yol.
- **Kapsam kontrolü (1 ay!):** Önce uçtan uca çalışan basit bir hat kurun, sonra iyileştirin. "Mükemmel SCM"e takılıp bitiremeyin.
- **Maliyet verisi yoksa:** Sentetik ama gerekçeli maliyetler atayın; metodoloji yine geçerli kalır.
