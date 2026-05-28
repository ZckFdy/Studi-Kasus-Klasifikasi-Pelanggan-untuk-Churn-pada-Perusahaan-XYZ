# Proyek Prediksi Churn Nasabah Bank: Implementasi ML Workflow

Repositori ini berisi implementasi praktis dari *Machine Learning Workflow* menggunakan dataset perilaku nasabah bank untuk memprediksi risiko *churn* (nasabah meninggalkan bank / `Exited`). Proyek ini membandingkan 5 algoritma klasifikasi populer di Python untuk menemukan model terbaik.

---

## Detail Dataset & Studi Kasus
Dataset yang digunakan memuat informasi profil dan aktivitas transaksi nasabah bank. Target variabel yang ingin diprediksi adalah `Exited` (1 = Churn, 0 = Tetap Bertahan).

### Langkah 1: Pengumpulan & Pemuatan Data (Data Collecting & Loading)
* Data diunduh langsung dari Google Drive menggunakan `pd.read_csv()` dan dimuat ke dalam struktur data Pandas DataFrame.
* Pemeriksaan awal dilakukan dengan `data.info()` dan `data.isnull().sum()` untuk memantau keberadaan nilai kosong (*missing values*).

### Langkah 2: Eksplorasi Data (Exploratory Data Analysis - EDA)
Proses investigasi awal dilakukan untuk memahami pola distribusi dan hubungan antar variabel melalui visualisasi:
* **Univariate Analysis:** Menggunakan histogram untuk melihat distribusi fitur numerik (seperti `CreditScore`, `Age`, `Balance`, `EstimatedSalary`) dan `countplot` untuk melihat sebaran variabel kategorikal (`Geography`, `Gender`).
* **Multivariate Analysis:** Menggunakan *heatmap* matriks korelasi untuk mengidentifikasi kekuatan hubungan antar fitur numerik, serta `pairplot` untuk melihat sebaran data secara multidimensi.
* **Analisis Target:** Visualisasi distribusi variabel `Exited` guna mengidentifikasi apakah terdapat ketidakseimbangan kelas (*class imbalance*).

### Langkah 3: Pra-pemrosesan Data (Data Preprocessing)
Sebelum data diberikan kepada algoritma, langkah transformasi berikut diterapkan:
1.  **Drop Kolom Tidak Relevan:** Menghapus kolom identitas seperti `RowNumber`, `CustomerId`, dan `Surname` karena tidak memiliki pengaruh terhadap pola *churn*.
2.  **Encoding Variabel Kategorikal:** Mengubah fitur teks (`Geography`, `Gender`) menjadi representasi numerik menggunakan `LabelEncoder`.
3.  **Feature Scaling:** Menyamakan skala nilai fitur menggunakan `StandardScaler` atau `MinMaxScaler` agar algoritma jarak (seperti KNN dan SVM) tidak terpengaruh oleh perbedaan satuan angka yang ekstrem.

### Langkah 4: Pembagian Data (Data Splitting)
Dataset dibagi secara acak menjadi dua subset terpisah menggunakan `train_test_split`:
* **Training Set ($X_{train}, y_{train}$):** Digunakan sepenuhnya untuk melatih kelima algoritma klasifikasi agar mampu mempelajari pola perilaku nasabah yang *churn*.
* **Testing Set ($X_{test}, y_{test}$):** Disimpan terpisah sebagai data uji independen untuk mengevaluasi kemampuan generalisasi model akhir.

---

## Eksplorasi Algoritma Klasifikasi (Model Selection)

Proyek ini melakukan eksperimen pengujian performa awal (*baseline model*) pada **5 algoritma klasifikasi sekaligus** menggunakan konfigurasi standar bawaan library `Scikit-Learn`:

1.  **K-Neighbors Classifier (KNN):** Algoritma klasifikasi berbasis kedekatan jarak antar data tetangga terdekat.
2.  **Decision Tree Classifier:** Algoritma berbasis struktur pohon keputusan yang memecah data berdasarkan aturan/kondisi tertentu.
3.  **Random Forest Classifier:** Metode *ensemble* yang menggabungkan banyak *decision tree* secara acak untuk meminimalkan *overfitting* dan meningkatkan akurasi.
4.  **Support Vector Machine (SVM / SVC):** Algoritma yang mencari *hyperplane* pembatas optimal untuk memisahkan kelas data secara maksimal.
5.  **Gaussian Naive Bayes (GaussianNB):** Klasifikasi berbasis teorema probabilitas Bayes yang mengasumsikan setiap fitur bersifat independen.

---

## Evaluasi Model (Model Evaluation)

Setiap model diuji menggunakan data pengujian (*Testing Set*) dan dievaluasi secara komprehensif menggunakan metrik klasifikasi dari fungsi `evaluate_model` yang mengukur:

* **Confusion Matrix:** Tabel pemetaan performa klasifikasi nyata vs prediksi untuk melihat detail salah klasifikasi:
    * *True Positive (TP)* & *True Negative (TN)*
    * *False Positive (FP)* & *False Negative (FN)*
* **Accuracy Score:** Mengukur persentase prediksi total yang benar secara keseluruhan.
* **Precision:** Mengukur ketepatan model dalam memprediksi nasabah yang benar-benar akan *churn* dari total nasabah yang ditebak *churn* (meminimalkan kerugian salah sasaran promo).
* **Recall (Sensitivity):** Mengukur seberapa banyak nasabah *churn* aktual yang berhasil dideteksi oleh model (sangat penting untuk meminimalkan nasabah *churn* yang lolos dari deteksi).
* **F1-Score:** Rata-rata harmonik antara *precision* dan *recall* untuk mengevaluasi model pada kondisi data yang tidak seimbang.

---

## Library Python yang Digunakan
* `pandas` & `numpy` - Manipulasi struktur data tabel dan operasi vektor numerik.
* `matplotlib.pyplot` & `seaborn` - Visualisasi visual grafik distribusi dan korelasi data.
* `sklearn` (Scikit-Learn) - *Splitting*, *preprocessing*, *modeling* klasifikasi, hingga evaluasi metrik performa.

---
