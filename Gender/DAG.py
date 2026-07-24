import os
import networkx as nx
import matplotlib.pyplot as plt

# Dosya yollarını dinamik olarak ayarlıyoruz
current_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(current_dir, "Gelişmiş_DAG_Haritası.png")

# 1. Yönlü graf oluşturuyoruz
G = nx.DiGraph()

# 2. Düğümleri ekliyoruz
nodes = [
    'Cinsiyet (A)', 
    'Aile Geliri\n(FAM_INC)', 
    'UGPA', 
    'LSAT', 
    'Fakülte Kalitesi\n(TIER)', 
    'Baro Başarısı (Y)'
]
G.add_nodes_from(nodes)

# 3. Nedensel Bağları ekliyoruz
edges = [
    ('Cinsiyet (A)', 'UGPA'),
    ('Cinsiyet (A)', 'LSAT'),
    ('Aile Geliri\n(FAM_INC)', 'UGPA'),
    ('Aile Geliri\n(FAM_INC)', 'LSAT'),
    ('Aile Geliri\n(FAM_INC)', 'Fakülte Kalitesi\n(TIER)'),
    ('UGPA', 'Fakülte Kalitesi\n(TIER)'),
    ('LSAT', 'Fakülte Kalitesi\n(TIER)'),
    ('UGPA', 'Baro Başarısı (Y)'),
    ('LSAT', 'Baro Başarısı (Y)'),
    ('Fakülte Kalitesi\n(TIER)', 'Baro Başarısı (Y)')
]
G.add_edges_from(edges)

# 4. Görselleştirme Ayarları
plt.figure(figsize=(10, 7))

pos = {
    'Cinsiyet (A)': (-1, 2),
    'Aile Geliri\n(FAM_INC)': (1, 2),
    'UGPA': (-1, 1),
    'LSAT': (1, 1),
    'Fakülte Kalitesi\n(TIER)': (0, 0),
    'Baro Başarısı (Y)': (0, -1)
}

nx.draw(
    G, pos, 
    with_labels=True, 
    node_size=5000, 
    node_color='#DDEAF6', 
    edgecolors='#5b9bd5',
    linewidths=2,
    font_size=11, 
    font_weight='bold', 
    arrows=True, 
    arrowsize=25,
    edge_color='#7f7f7f',
    width=1.5
)

plt.title("Gelişmiş Yapısal Nedensel Model (SCM) Haritası", fontsize=15, pad=20)
plt.savefig(save_path, format="PNG", dpi=300, bbox_inches='tight')
print(f"Yeni DAG görseli '{save_path}' adıyla başarıyla kaydedildi!")
plt.show()