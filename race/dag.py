"""
DAG Tablosu ve Görseli — race (A) ekseni için genişletilmiş nedensel model
A -> FAM_INC -> LSAT, UGPA -> TIER -> DECILE1 -> Y
"""
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1) DAG TABLOSU

dag_tablo = pd.DataFrame([
	{"Değişken": "A",       "Rol": "Kök (hassas öznitelik)", "Ebeveynleri": "—",
	 "Açıklama": "Irk (white=0, diğer=1). Karşıolgusal analizin odağı."},
	{"Değişken": "FAM_INC", "Rol": "Ara değişken",            "Ebeveynleri": "A",
	 "Açıklama": "Aile geliri dilimi. A'dan etkileniyor (tarihsel eşitsizlik)."},
	{"Değişken": "LSAT",    "Rol": "Ara değişken",            "Ebeveynleri": "A, FAM_INC",
	 "Açıklama": "Sınav puanı. A doğrudan + FAM_INC üzerinden dolaylı etkileniyor."},
	{"Değişken": "UGPA",    "Rol": "Ara değişken",            "Ebeveynleri": "A, FAM_INC",
	 "Açıklama": "Lisans not ortalaması. LSAT ile aynı mantık."},
	{"Değişken": "TIER",    "Rol": "Ara değişken",            "Ebeveynleri": "LSAT, UGPA, FAM_INC",
	 "Açıklama": "Okul prestij kademesi. Giriş puanları ve gelirle belirleniyor."},
	{"Değişken": "DECILE1", "Rol": "Ara değişken",            "Ebeveynleri": "LSAT, UGPA, TIER",
	 "Açıklama": "1. yıl hukuk fakültesi performansı."},
	{"Değişken": "Y",       "Rol": "Hedef değişken",          "Ebeveynleri": "DECILE1, FAM_INC",
	 "Açıklama": "Bar sınavını geçme durumu (pass_bar)."},
])

print("=" * 100)
print("DAG TABLOSU")
print("=" * 100)
print(dag_tablo.to_string(index=False))
dag_tablo.to_csv("dag_tablo.csv", index=False)
print("\nKaydedildi: dag_tablo.csv")

# 2) DAG GÖRSELİ

edges = [
	("A", "FAM_INC"),
	("A", "LSAT"), ("FAM_INC", "LSAT"),
	("A", "UGPA"), ("FAM_INC", "UGPA"),
	("LSAT", "TIER"), ("UGPA", "TIER"), ("FAM_INC", "TIER"),
	("LSAT", "DECILE1"), ("UGPA", "DECILE1"), ("TIER", "DECILE1"),
	("DECILE1", "Y"), ("FAM_INC", "Y"),
]
G = nx.DiGraph(edges)

# Topolojik sıraya göre katmanlı yerleşim (soldan sağa akış)
pos = {
	"A":        (0.0, 0.5),
	"FAM_INC":  (1.0, 0.9),
	"LSAT":     (2.0, 1.0),
	"UGPA":     (2.0, 0.2),
	"TIER":     (3.0, 0.9),
	"DECILE1":  (4.0, 0.6),
	"Y":        (5.0, 0.5),
}

renkler = {
	"A": "#f4a6a6",        # hassas öznitelik - kırmızımsı
	"FAM_INC": "#f4d4a6",  # ara değişken - turuncu
	"LSAT": "#a6d4f4", "UGPA": "#a6d4f4",   # gözlenen özellikler - mavi
	"TIER": "#f4d4a6",
	"DECILE1": "#c9a6f4",  # ara/performans - mor
	"Y": "#a6f4b4",        # hedef - yeşil
}
node_colors = [renkler[n] for n in G.nodes()]

plt.figure(figsize=(11, 6))
nx.draw(
	G, pos, with_labels=True, node_color=node_colors, node_size=2600,
	font_size=10, font_weight="bold", arrowsize=22, edge_color="#555",
	connectionstyle="arc3,rad=0.08"  # bazı okları hafif kavisli çiz, çakışmayı azalt
)
plt.title("Nedensel DAG — Race (A) Ekseni Karşıolgusal Adalet Modeli", fontsize=13)
plt.tight_layout()
plt.savefig("dag_gorsel_v2.png", dpi=130, bbox_inches="tight")
print("Kaydedildi: dag_gorsel_v2.png")