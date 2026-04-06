# 🏦 Bank Marketing Analysis - Data Mining Project

Proyek ini bertujuan untuk menganalisis data pemasaran bank menggunakan teknik **machine learning (Random Forest)** untuk memprediksi apakah seorang nasabah akan berlangganan deposito berjangka (`y`) berdasarkan profil dan riwayat interaksi nasabah.

---

## 📁 Struktur Proyek

* `data_mining.py` → Skrip utama yang berisi proses *machine learning* (preprocessing, training, dan evaluasi)
* `bank.csv` → Dataset berisi data nasabah dan riwayat kampanye pemasaran (delimiter `;`)

---

## ⚙️ Requirements

Pastikan telah menginstal Python dan library berikut:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 📊 Deskripsi Dataset

Dataset terdiri dari **17 atribut** yang merepresentasikan karakteristik nasabah dan aktivitas marketing:

| Atribut   | Deskripsi                              |
| --------- | -------------------------------------- |
| age       | Umur nasabah                           |
| job       | Pekerjaan                              |
| marital   | Status pernikahan                      |
| education | Tingkat pendidikan                     |
| default   | Status gagal bayar kredit              |
| balance   | Saldo rekening                         |
| housing   | Kepemilikan pinjaman rumah (KPR)       |
| loan      | Kepemilikan pinjaman pribadi           |
| contact   | Jenis kontak                           |
| day       | Hari kontak                            |
| month     | Bulan kontak                           |
| duration  | Durasi panggilan                       |
| campaign  | Jumlah kontak selama kampanye          |
| pdays     | Hari sejak kontak terakhir             |
| previous  | Jumlah kontak sebelumnya               |
| poutcome  | Hasil kampanye sebelumnya              |
| y         | Target (berlangganan deposito: yes/no) |

---

## 🚀 Cara Menjalankan

1. Pastikan file `data_mining.py` dan `bank.csv` berada dalam folder yang sama
2. Jalankan perintah berikut:

```bash
python data_mining.py
```

---

## 🔍 Alur Machine Learning

Proyek ini menggunakan tahapan berikut:

1. **Data Loading**
   Membaca dataset menggunakan `pandas`

2. **Data Preprocessing**

   * Mengganti nilai `unknown` dengan modus
   * Membersihkan dan menyiapkan data

3. **Encoding**

   * *One Hot Encoding* untuk fitur kategorikal
   * *Label Encoding* untuk target

4. **Data Splitting**

   * 80% data training
   * 20% data testing

5. **Feature Scaling**
   Menyamakan skala fitur numerik agar model optimal

6. **Model Training**
   Menggunakan algoritma **Random Forest Classifier**

7. **Model Evaluation**

   * Accuracy
   * Confusion Matrix
   * Classification Report (*precision, recall, f1-score*)

---

## 🤖 Model yang Digunakan

Model utama dalam proyek ini adalah:

* **Random Forest Classifier**
  Algoritma ensemble yang menggabungkan banyak decision tree untuk meningkatkan akurasi dan mengurangi overfitting.

---

## 🎯 Tujuan Proyek

* Memprediksi peluang nasabah berlangganan deposito
* Membantu pengambilan keputusan dalam strategi pemasaran
* Menganalisis faktor-faktor yang memengaruhi keputusan nasabah

---

## 📌 Catatan

* Pastikan nama file dataset tetap `bank.csv`
* Gunakan delimiter `;` saat membaca data
* Dataset bersifat supervised learning (memiliki label `y`)

---

## ✍️ Penulis

Repository ini dibuat untuk keperluan pembelajaran **data mining & machine learning**.

---
