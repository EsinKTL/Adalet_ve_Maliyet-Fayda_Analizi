# ⚖️ Yapısal Nedensel Model (SCM) ile Algoritmik Adalet ve Karşıolgusal Analiz

> **Proje Özeti:** Makine öğrenmesi tahmin modellerinde, cinsiyet gibi hassas niteliklerin yarattığı dolaylı ayrımcılığı (bias), Yapısal Nedensel Modeller (SCM) ve karşıolgusal (counterfactual) simülasyonlar kullanarak tespit eden ve gideren bir algoritmik adalet (causal fairness) analiz projesidir.

---

## 📖 Detaylı Açıklama

Günümüzde algoritmik karar alma sistemleri, veri setlerindeki tarihsel önyargıları öğrenerek ayrımcı sonuçlar üretebilmektedir. Bu proje, hukuk fakültesi öğrenci verilerini temel alarak **Baro Başarısını** tahmin ederken, **Cinsiyet** değişkeninin diğer değişkenler (Aile Geliri, UGPA, LSAT, Fakülte Kalitesi vb.) üzerindeki dolaylı ve haksız etkilerini ortadan kaldırmayı amaçlar. 

Proje, **Nedensel Çıkarım (Causal Inference)** ilkelerine dayanarak şu süreçleri yönetir:

1. **Abduction (Açımlama):** Bireyin performansını etkileyen cinsiyet dışı saf ve gözlemlenemeyen ($U$) yetenek/efor faktörlerinin denklemler yoluyla hesaplanması.
2. **Intervention (Müdahale/Simülasyon):** Kişinin cinsiyeti farklı olsaydı hayatındaki diğer değişkenlerin zincirleme olarak nasıl şekilleneceğinin karşıolgusal (counterfactual) olarak simüle edilmesi.
3. **Evaluation (Değerlendirme):** Saf $U$ faktörleriyle eğitilen **Adil Model** ile gerçek hayat verisiyle eğitilen **Klasik Modelin**, Doğruluk (Accuracy) ve Adaletsizlik (Flip Oranı) metrikleri üzerinden kıyaslanması.

---

## 🛠 Kullanılan Teknolojiler

Proje tamamen **Python** ekosistemi üzerinde geliştirilmiş olup, nedensellik ve veri analizi için endüstri standardı kütüphaneler kullanmaktadır:

- **Python 3.x**
- **Statsmodels:** İstatistiksel regresyonlar, katsayı analizi ve kalıntı (residual) hesaplamaları için.
- **Pandas & NumPy:** Veri işleme, manipülasyon ve matris işlemleri için.
- **NetworkX:** Nedensel ilişkilerin (DAG) matematiksel olarak kurulması için.
- **Matplotlib:** DAG tablolarının görselleştirilmesi için.

---

## ⚙️ Gereksinimler & Kurulum

Projeyi yerel makinenizde çalıştırmak için ilgili Python ortamınızda aşağıdaki kütüphanelerin yüklü olması gerekmektedir. Bağımlılıkları terminalinizde tek bir komutla kurabilirsiniz:

```bash
# Gerekli Python kütüphanelerini kurun
pip install pandas numpy statsmodels scikit-learn networkx matplotlib seaborn
```

> **Not:** Projeyi bilgisayarınıza indirdikten sonra, çalışmak için gereken temizlenmiş ham verinin (`lsac_clean.csv`) çalışma dizininde bulunduğundan emin olun.

---

## 🚀 Kullanım ve Dosyaların Anlamsal Çıkarımları

Proje içerisindeki üç ana Python script'i mantıksal bir sıralama ile çalıştırılmalıdır. Sistemdeki her bir dosyanın projedeki rolü ve ürettiği sonuçlar şöyledir:

### 1. Nedensel Haritanın Çıkarılması (`DAG.py`)
Değişkenler arası nedensel bağların şematik haritasını (Yönlü Döngüsüz Graf - DAG) oluşturur.

**Çalıştırma Komutu:**
```bash
python DAG.py
```
* **Mantıksal Çıkarım:** Veri setindeki Cinsiyet, Aile Geliri, Notlar ve Baro Başarısı arasındaki etki-tepki zincirini matematiksel bir yapıya (graf) döker. Sistemin hangi yönde aktığını görselleştirerek modelin teorik altyapısını temellendirir.
* **Çıktı:** `Gelişmiş_DAG_Haritası.png` görselini üretir.

---

### 2. Modellerin Eğitilmesi ve Simülasyon (`TrainModel.py`)
Kalıntı değişkenleri hesaplar ve paralel evren (karşıolgusal) verilerini türetir.

**Çalıştırma Komutu:**
```bash
python TrainModel.py
```
* **Mantıksal Çıkarım:** Sistem öncelikle, kişinin başarılarına etki eden cinsiyet ve gelir gibi dış etkenleri izole edip, kişinin sadece "kendi çabasını/genel faktörleri" temsil eden saf kalıntı değerlerini ($U$ matrisleri) bulur. Ardından *"Bu kişilerin cinsiyeti farklı olsaydı, notları ve fakülte başarıları nasıl değişirdi?"* sorusunun simülasyonunu yapar.
* **Çıktı:** Adil bir öğrenme için arındırılmış `lsac_with_U_zengin.csv` ve karşıolgusal `lsac_counterfactual_sim_zengin.csv` veri setlerini oluşturarak sisteme kaydeder.

---

### 3. Modellerin Karşılaştırılması ve Nihai Raporlama (`nihai_degerlendirme.py`)
Modelleri rekabet ettirir ve adaletsizlik (bias) ölçümü yapar.

**Çalıştırma Komutu:**
```bash
python nihai_degerlendirme.py
```
* **Mantıksal Çıkarım:** Arındırılmış faktörlerle eğitilen "Nedensel Adil Model" ile ayrımcı olabilecek dış etkenleri içeren "Klasik Model", cinsiyetin tersine çevrildiği veriler üzerinde teste tabi tutulur. Model doğruluğunun (Accuracy) yanı sıra, cinsiyet değiştiğinde verilen baro kararının ne kadar değiştiğini gösteren Adaletsizlik (Flip) oranı ölçülür.
* **Çıktı:** Sonuçlar `Model_Karsilastirma_Grafikleri.png` ile görselleştirilir ve klasik modellerin kırılganlığına karşın, adil modellerin yapısal istikrarı bilimsel bir profesyonellikte kanıtlanmış olur.
