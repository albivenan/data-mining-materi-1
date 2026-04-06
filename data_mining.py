import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tabulate import tabulate

def cetak_tabel(df, judul, jumlah=3):
    """Fungsi helper untuk mencetak tabel cantik dengan garis sel"""
    print(f"\n>>> [TABEL INSPEKSI] {judul}")
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)
    print(tabulate(df.head(jumlah), headers='keys', tablefmt='grid', showindex=False))

# ==========================================
# 1. MENGIMPOR DATASET
# ==========================================
dataset = pd.read_csv('bank.csv', sep=';')
dataset.replace('unknown', np.nan, inplace=True)
print("1. Dataset berhasil dimuat.")
print(f"   -> Total data: {len(dataset)} baris.")
cetak_tabel(dataset, "DATA AWAL DARI CSV", 3)

# ==========================================
# 2. MEMISAHKAN FITUR (X) DAN TARGET (y)
# ==========================================
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print("2. Fitur (X) dan Target (y) berhasil dipisahkan.")
cetak_tabel(X, "MATRIKS X (FITUR)", 2)
print(f"   -> Cuplikan Target y (10 data): {y[:10]}")

# ==========================================
# 3. MENANGANI MISSING VALUE
# ==========================================
# Menghitung awal data kosong untuk penjelasan
nan_count = pd.DataFrame(X).isna().sum().sum()
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
X = imputer.fit_transform(X)
print("3. Data kosong (NaN) berhasil diisi.")
print(f"   -> Sebanyak {nan_count} data 'unknown' telah ditambal dengan nilai modus.")
cetak_tabel(X, "X SETELAH IMPUTASI (TIDAK ADA UNKNOWN)", 2)

# ==========================================
# 4. ENCODING DATA KATEGORIKAL
# ==========================================
# Simpan jumlah kolom awal untuk penjelasan
kolom_awal = X.shape[1]
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(sparse_output=False), [1, 2, 3, 4, 6, 7, 8, 10, 15])], 
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X))
le = LabelEncoder()
y = le.fit_transform(y)
print("4. Data teks berhasil diubah menjadi format angka (Encoding).")
print(f"   -> Fitur X bertambah kolom (One-Hot): {kolom_awal} menjadi {X.shape[1]}")
cetak_tabel(X[:, :10], "X SETELAH ENCODING (10 KOLOM PERTAMA)", 2)
print(f"   -> Target y ter-encode: {y[:10]} (1=yes, 0=no)")

# ==========================================
# 5. MEMBAGI DATASET
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(f"5. Data dibagi: {len(X_train)} data training & {len(X_test)} data testing.")

# ==========================================
# 6. FEATURE SCALING
# ==========================================
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("6. Normalisasi data (Feature Scaling) selesai.")
cetak_tabel(X_train[:, :10], "X_TRAIN SETELAH SCALING (10 KOLOM PERTAMA)", 2)

# ==========================================
# 7. MELATIH MODEL (RANDOM FOREST)
# ==========================================
classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=1)
classifier.fit(X_train, y_train)
print("7. Model Random Forest berhasil dilatih.\n")

# ==========================================
# OUTPUT EVALUASI (HASIL AKHIR)
# ==========================================
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("="*30)
print("HASIL EVALUASI MODEL")
print("="*30)
print(f"Akurasi Model: {acc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=['Tidak Langganan (No)', 'Langganan (Yes)']))
