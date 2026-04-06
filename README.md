# Bank Marketing Analysis - Data Mining Project

Proyek ini bertujuan untuk menganalisis data pemasaran bank menggunakan teknik data mining untuk memprediksi apakah seorang nasabah akan berlangganan deposito berjangka (fitur `y`) berdasarkan berbagai atribut profil nasabah.

## 📁 Struktur Proyek
- `data_mining.py`: Skrip Python utama yang berisi alur kerja *machine learning* (Preprocessing, Training, dan Evaluasi).
- `bank.csv`: Dataset yang berisi data profil nasabah dan riwayat kampanye pemasaran (menggunakan pemisah `;`).

## ⚙️ Prasyarat (Requirements)
Sebelum menjalankan program, pastikan Anda telah menginstal bahasa pemrograman Pyhton dan pustaka berikut:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## 🚀 Cara Menjalankan
1. Letakkan `data_mining.py` dan `bank.csv` dalam folder yang sama.
2. Jalankan skrip menggunakan Python:
```bash
python data_mining.py
```

## 🔍 Hubungan Antar File
Skrip `data_mining.py` dirancang untuk membaca file `bank.csv` secara otomatis sebagai sumber data utama. Hubungan teknisnya adalah sebagai berikut:
1. **Pemuatan Data**: Skrip memanggil `pd.read_csv('bank.csv', sep=';')`. Pastikan nama file tetap `bank.csv` agar tidak terjadi error.
2. **Preprocessing**: Data mentah dari CSV akan diolah (mengisi nilai kosong, mengubah teks menjadi angka, dan standarisasi) sebelum dimasukkan ke model.
3. **Prediksi**: Model *Random Forest* akan belajar dari pola yang ada di `bank.csv` untuk kemudian diuji akurasinya.

## 📊 Alur Kerja Skrip
1. **Import Dataset**: Membaca data dari `bank.csv`.
2. **Handling Missing Values**: Mengganti nilai 'unknown' dengan nilai yang paling sering muncul (modus).
3. **Encoding**: Mengubah data kategorikal (teks) menjadi format numerik menggunakan *One Hot Encoding* (untuk fitur) dan *Label Encoding* (untuk target).
4. **Splitting Data**: Membagi data menjadi 80% untuk pelatihan dan 20% untuk pengujian.
5. **Feature Scaling**: Menyamakan skala seluruh angka agar model bekerja lebih optimal.
6. **Training**: Melatih model menggunakan algoritma **Random Forest Classifier**.
7. **Evaluation**: Menampilkan Akurasi, *Confusion Matrix*, dan *Classification Report*.

---
*Dibuat untuk keperluan analisis data mining.*
