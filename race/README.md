# ⚖️ Yapısal Nedensel Model (SCM) ile Algoritmik Adalet ve Karşıolgusal Analiz

> **Proje Özeti:** Makine öğrenmesi tahmin modellerinde, cinsiyet gibi hassas niteliklerin yarattığı dolaylı ayrımcılığı (bias), Yapısal Nedensel Modeller (SCM) ve karşıolgusal (counterfactual) simülasyonlar kullanarak tespit eden ve gideren bir algoritmik adalet (causal fairness) analiz projesidir.

---

## 📖 Detaylı Açıklama

Bu proje, makine öğrenimi modellerinde **Algoritmik Adalet (Algorithmic Fairness)** sağlamak amacıyla geliştirilmiştir. Sistem, öğrencilerin; aile geliri (`FAM_INC`), sınav puanları (`LSAT`), lisans not ortalamaları (`UGPA`) ve hukuk fakültesindeki performansları (`DECILE1`) gibi faktörlerin, hassas bir öznitelik olan **ırk (A)** ile nasıl nedensel bir ilişki içinde olduğunu inceler.

**Projenin Çözdüğü Temel Sorunlar:**
- **Nedensel DAG (Directed Acyclic Graph) Modellemesi:** Değişkenler arası nedensellik akışını ve hiyerarşisini görselleştirir ve analiz eder.
- **Karşıolgusal (Counterfactual) Üretim:** `statsmodels` OLS regresyonu kullanılarak her bireyin kendine has yetenek ve çaba sinyalleri (Residual / U değerleri) izole edilir (Abduction).
- **Adil Değerlendirme:** Bireylerin profillerinde ırk değişkeni sanal olarak tersine çevrilerek (Intervention) kişinin "eğer farklı bir grupta olsaydı performansı ne olurdu?" sorusuna matematiksel bir cevap (Forward Pass) aranır.

---

## 🚀 Kullanılan Teknolojiler

Proje tamamen **Python** ekosistemi üzerinde geliştirilmiş olup, nedensellik ve veri analizi için endüstri standardı kütüphaneler kullanmaktadır:

- **Python 3.x**
- **Statsmodels:** İstatistiksel regresyonlar, katsayı analizi ve kalıntı (residual) hesaplamaları için.
- **Pandas & NumPy:** Veri işleme, manipülasyon ve matris işlemleri için.
- **NetworkX:** Nedensel ilişkilerin (DAG) matematiksel olarak kurulması için.
- **Matplotlib:** DAG tablolarının görselleştirilmesi için.

---

## ⚙️ Gereksinimler & Kurulum

Projeyi yerel makinenizde çalıştırmak için öncelikle gerekli kütüphanelerin yüklü olduğundan emin olun. Aşağıdaki adımları takip ederek projeyi hemen ayağa kaldırabilirsiniz.

**1. Repoyu klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADINIZ/Adalet_ve_Maliyet-Fayda_Analizi.git
cd Adalet_ve_Maliyet-Fayda_Analizi/race
```

**2. Gerekli kütüphaneleri yükleyin:**
```bash
pip install pandas numpy statsmodels networkx matplotlib
```

> **Not:** Projenin çalışabilmesi için çalışma dizininde `lsac_clean.csv` veri setinin bulunması gerekmektedir.

---

## 💻 Kullanım ve Çıktıların Yorumlanması

Proje iki temel betikten (script) oluşmaktadır. Çalıştırma sırası, komutlar ve elde edilen çıktıların ne anlama geldiği aşağıda açıklanmıştır:

### 1. Nedensel Modelin Görselleştirilmesi (`dag.py`)
Nedensel akışı (`A -> FAM_INC -> LSAT` vb.) anlamak ve görselleştirmek için kullanılır.

**Komut:**
```bash
python dag.py
```

**Çıktı ve Yorumlanması:**
- **`dag_tablo.csv`:** Değişkenlerin rollerini (Kök, Ara değişken, Hedef) ve birbirleriyle olan ebeveyn-çocuk ilişkilerini özetleyen tablodur.
- **`dag_gorsel_v2.png`:** Değişkenler arası akışı soldan sağa doğru gösteren grafiktir. Bu grafik, ırkın (`A`) sadece doğrudan değil, aynı zamanda aile geliri (`FAM_INC`) üzerinden dolaylı yoldan sınav puanlarını (`LSAT`, `UGPA`) nasıl etkilediğini teorik olarak kanıtlar.

---

### 2. Yapısal Nedensel Model ve Karşıolgusal Analiz (`scm_race.py`)
Ana analiz dosyasıdır. Veriyi okur, regresyonları çalıştırır ve "ırkın etkisi çıkarılmış" yeni bir veri seti oluşturur.

**Komut:**
```bash
python scm_race.py
```

**Çıktılar ve Detaylı Yorumlanması:**

* **OLS Regresyon Katsayıları:**
  Script çalıştığında ekrana `FAM_INC`, `LSAT`, `UGPA`, `TIER` ve `DECILE1` için ayrı ayrı OLS (En Küçük Kareler) regresyon katsayıları yazdırılır.
  * *Yorum:* Örneğin `FAM_INC` (Aile Geliri) modelinde `A` (Irk) değişkeninin katsayısının negatif çıkması beklenir. Bu, veri setindeki dezavantajlı ırk grubunda olmanın, tarihsel eşitsizlikler nedeniyle ortalama geliri düşürdüğünü matematiksel olarak gösterir. Aynı şekilde `LSAT` ve `UGPA` modellerindeki katsayılar da gelirin ve ırkın akademik başarı üzerindeki net ağırlıklarını ortaya koyar.

* **Korelasyon Değerleri (Artıklar ve Girdi Değişkenleri Arasında):**
  Ekrana `np.corrcoef` ile hesaplanan çeşitli korelasyon değerleri basılır.
  * *Yorum:* Hesaplanan $U$ (Residual/Artık) değerleri ile model girdileri (örn: `A` veya `FAM_INC`) arasındaki korelasyonun matematiksel bir zorunluluk olarak $0$'a çok yakın ($0.00X$ gibi) çıkması gerekir. Örneğin $U_{FAM\_INC}$, bir öğrencinin "kendi ırk grubunun ortalamasına göre ne kadar zengin veya fakir" olduğunu gösteren kişisel çaba ve durum sinyalidir. Bu artık değerinin ($U$) ırktan bağımsız olması, modelin ırkın yapısal etkisini başarılı bir şekilde izole ettiğini (**Abduction** aşamasının başarısını) kanıtlar.

* **Karşıolgusal (Counterfactual) Veriler (`A_cf`, `FAM_INC_cf`, `LSAT_cf` vb.):**
  Scriptin sonunda, orijinal veriler ile nedensel akış üzerinden yeniden üretilen karşıolgusal (`_cf`) verilerin karşılaştırıldığı bir tablo ekrana yazdırılır.
  * *Yorum:* Sistem her öğrenci için paralel bir evren (karşıolgusal senaryo) kurgular. Eğer dezavantajlı gruptaki bir öğrenci ($A=1$), avantajlı grupta ($A=0$) olsaydı:
    * Aile geliri (`FAM_INC_cf`) ne kadar artardı?
    * Aile gelirindeki bu artış ve ırkın değişmesi LSAT sınav puanını (`LSAT_cf`) ne kadar yükseltirdi?
  * Elde edilen bu `_cf` uzantılı değerler, öğrencinin sistemsel ve tarihsel dezavantajlardan arındırılmış, gerçek potansiyelini yansıtan **"Adil"** değerlerdir. Hedef model, kişinin baro sınavı ($Y$) başarısını tahmin ederken önyargılı ham veriler yerine bu arındırılmış karşıolgusal değerleri kullanarak adil (*fair*) sonuçlar üretir.
