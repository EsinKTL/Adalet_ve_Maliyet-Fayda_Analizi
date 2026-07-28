import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# SADECE df_clean'i import ediyoruz. Tüm '_cf' uzantılı sütunlar zaten bunun içinde.
from scm_race import df_clean

# --- 1. VERİ HAZIRLIĞI ---
y = df_clean['Y']

# Baz Model (Adil Olmayan) Girdileri: Ham notlar ve Irk
X_base = df_clean[['LSAT', 'UGPA', 'A']]

# Karşıolgusal Adil Model Girdileri: Sadece U (Arındırılmış) değerleri
X_fair = df_clean[['U_LSAT', 'U_UGPA']]

# --- 2. TRAIN/TEST AYRIMI ---
# Aynı satırların denk gelmesi için random_state sabitliyoruz
# stratify=y parametresi ile sınıf oranlarını train ve test setlerine eşit dağıtıyoruz
X_base_train, X_base_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42, stratify=y)
X_fair_train, X_fair_test, _, _ = train_test_split(X_fair, y, test_size=0.2, random_state=42, stratify=y)

# Test seti için Karşıolgusal verileri (Paralel Evren) hazırlıyoruz
# df_cf yerine, df_clean içindeki '_cf' sütunlarını çağırıyoruz
X_base_cf_test = df_clean.loc[X_base_test.index, ['LSAT_cf', 'UGPA_cf', 'A_cf']]
X_base_cf_test.columns = ['LSAT', 'UGPA', 'A'] # İsimleri modelin tanıması için eşitliyoruz

# --- 3. MODELLERİN EĞİTİMİ ---
clf_base = LogisticRegression(random_state=42, class_weight='balanced')
clf_base.fit(X_base_train, y_train)

clf_fair = LogisticRegression(random_state=42, class_weight='balanced')
clf_fair.fit(X_fair_train, y_train)

# --- 4. ACCURACY (DOĞRULUK) HESAPLAMASI ---
preds_base = clf_base.predict(X_base_test)
preds_fair = clf_fair.predict(X_fair_test)

acc_base = accuracy_score(y_test, preds_base)
acc_fair = accuracy_score(y_test, preds_fair)

# --- 5. FAIRNESS (KARŞIOLGUSAL TUTARLILIK) HESAPLAMASI ---
# Baz modelin paralel evren verisindeki tahminleri
preds_base_cf = clf_base.predict(X_base_cf_test)

# Kaç tanesinin kararı DEĞİŞMEDİ? (Tutarlılık Skoru)
cc_base = np.mean(preds_base == preds_base_cf)

# Adil modelde girdiler (U değerleri) ırktan bağımsız olduğu için değişmez
cc_fair = 1.0

# --- 6. SONUÇ RAPORU ---
print("--- MODEL KARŞILAŞTIRMA RAPORU ---")
print(f"1. Adil Olmayan Model (Baseline):")
print(f"   - Doğruluk (Accuracy)     : %{acc_base*100:.2f}")
print(f"   - Karşıolgusal Tutarlılık : %{cc_base*100:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds_base))
print("\nClassification Report:")
print(classification_report(y_test, preds_base))
print()
print()
print(f"2. Karşıolgusal Adil Model (Fair Model):")
print(f"   - Doğruluk (Accuracy)     : %{acc_fair*100:.2f}")
print(f"   - Karşıolgusal Tutarlılık : %{cc_fair*100:.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds_fair))
print("\nClassification Report:")
print(classification_report(y_test, preds_fair))