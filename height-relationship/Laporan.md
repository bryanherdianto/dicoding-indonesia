# Laporan Proyek Machine Learning - Bryan Herdianto

## Domain Proyek
Pada akhir abad ke-19, Sir Francis Galton, seorang polymath terkenal asal Inggris, melakukan studi yang sekarang dikenal sebagai dasar dari ilmu genetika dan statistik modern. Studi Galton pada tahun 1886 memfokuskan pada hubungan antara tinggi badan orang tua dan anak-anak mereka. Penelitian ini mencakup pengamatan terhadap 934 anak dan 205 keluarga, dengan tujuan utama untuk mengidentifikasi apakah ada korelasi antara tinggi badan anak-anak dengan tinggi badan orang tua mereka, dan apakah ada hubungan yang signifikan antara tinggi badan suami dan istri.

Penelitian Galton mengenai tinggi badan keluarga memiliki implikasi luas dalam bidang genetika, biostatistika, dan sosiologi. Beberapa alasan mengapa masalah ini penting untuk diselesaikan adalah:
1. Pemahaman tentang Hereditas: Mengetahui sejauh mana karakteristik fisik seperti tinggi badan diturunkan dari orang tua ke anak-anak memberikan wawasan penting tentang mekanisme hereditas.
2. Perencanaan Kesehatan: Memahami pola hereditas dapat membantu dalam perencanaan kesehatan dan intervensi medis di masa depan, terutama dalam menangani kondisi yang diwariskan secara genetik.

Galton menemukan bahwa tinggi badan anak-anak cenderung mendekati rata-rata tinggi badan populasi umum daripada ekstrem tinggi atau pendek dari orang tua mereka, fenomena yang dikenal sebagai "regresi menuju rata-rata". Ini adalah penemuan kunci dalam statistik dan genetika kuantitatif.

Referensi:
Galton, F. (1886). Regression Towards Mediocrity in Hereditary Stature. The Journal of the Anthropological Institute of Great Britain and Ireland, 15, 246-263. Galton's seminal paper where he introduced the concept of regression.
Fisher, R. A. (1918). The Correlation Between Relatives on the Supposition of Mendelian Inheritance. Transactions of the Royal Society of Edinburgh, 52, 399-433. Fisher extended Galton's work and laid the foundation for modern quantitative genetics.
Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

## Business Understanding

### Problem Statements
1. Apakah terdapat hubungan yang signifikan antara tinggi badan anak-anak dengan tinggi badan orang tua mereka? Penelitian ini ingin mengetahui apakah tinggi badan anak dapat diprediksi berdasarkan tinggi badan ayah dan ibu.
2. Apakah ada korelasi antara tinggi badan suami dan istri? Penelitian ini juga ingin mengetahui apakah pemilihan pasangan dalam pernikahan menunjukkan adanya hubungan dalam hal tinggi badan.
3. Bagaimana pengaruh tinggi badan orang tua terhadap distribusi tinggi badan anak-anak mereka dalam konteks regresi menuju rata-rata yang ditemukan oleh Galton?

### Goals
1. Menentukan dan mengukur tingkat korelasi antara tinggi badan orang tua dan tinggi badan anak-anak mereka, serta membuat model prediktif yang dapat memperkirakan tinggi badan anak berdasarkan tinggi badan orang tua. Tujuannya adalah untuk memberikan pemahaman kuantitatif tentang seberapa kuat faktor genetik mempengaruhi tinggi badan anak.

2. Menganalisis data untuk menemukan apakah terdapat korelasi yang signifikan antara tinggi badan suami dan istri. Tujuannya adalah untuk memahami pola sosial atau biologis dalam pemilihan pasangan berdasarkan tinggi badan.

3. Mengkaji fenomena regresi menuju rata-rata dalam konteks tinggi badan keluarga. Tujuannya adalah untuk memverifikasi dan memahami konsep ini dalam data Galton, serta bagaimana fenomena ini mempengaruhi interpretasi data hereditas tinggi badan.

## Data Understanding
Dataset yang digunakan pada laporan ini adalah dataset tinggi orangtua dan tinggi anak-anak. Dataset tersebut bersumber dari kaggle dan link disediakan sebagai berikut.

Link: [Kaggle](https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data/data)

### Variabel-variabel pada dataset adalah sebagai berikut:
- `family` : id dari masing-masing keluarga
- `father` : tinggi dari ayah
- `mother` : tinggi dari ibu
- `midparentHeight` : tinggi rata-rata dari ayah dan ibu
- `children` : jumlah anak di keluarga
- `childNum` : id dari masing-masing anak di keluarga itu
- `childHeight` : tinggi dari anak
- `gender` : jenis kelamin dari anak

## Data Preparation
Dalam proyek ini, kami menerapkan beberapa teknik data preparation untuk memastikan bahwa data siap digunakan dalam proses analisis dan pemodelan. Proses data preparation melibatkan beberapa langkah yang esensial untuk mengubah data mentah menjadi format yang sesuai untuk algoritma machine learning. Berikut adalah teknik-teknik yang digunakan beserta alasan mengapa tahapan tersebut diperlukan:

1. **Label Encoding**  
Menggunakan LabelEncoder dari sklearn untuk mengganti data string menjadi integer. Banyak algoritma machine learning memerlukan input numerik. Dengan mengganti data string menjadi integer, kita membuat data dapat diproses oleh algoritma tersebut.
2. **Data Splitting**  
Dengan train_test_split dari sklearn, data dibagi ke dalam set pelatihan dan pengujian. Dengan memisahkan data ke dalam set pelatihan dan pengujian penting untuk mengukur performa model secara objektif. Dengan memiliki set pengujian yang tidak pernah dilihat oleh model selama pelatihan, kita dapat mengevaluasi bagaimana model akan bekerja pada data yang belum pernah ditemui sebelumnya.

## Modeling
Laporan ini mempertimbangkan beberapa algoritma machine learning seperti Linear Regression, AdaBoost Regressor, MLP Regressor, Gradient Boosting Regressor, dan Random Forest Regressor. Berikut kelebihan dan kekurangan masing-masing algoritma.

### 1. Linear Regression
- **Kelebihan**:
  - Sederhana dan mudah diinterpretasi.
  - Cepat dalam proses pelatihan dan prediksi.
- **Kekurangan**:
  - Kurang fleksibel dalam menangani pola data yang kompleks.
  - Rentan terhadap outliers.

### 2. AdaBoost Regressor
- **Kelebihan**:
  - Mampu meningkatkan performa dengan mengurangi bias.
  - Tidak terlalu rentan terhadap overfitting.
- **Kekurangan**:
  - Sensitif terhadap noise dan outliers dalam data.

### 3. MLP Regressor
- **Kelebihan**:
  - Mampu menangani hubungan yang kompleks antara fitur dan target.
  - Bisa melakukan learning dari data non-linear.
- **Kekurangan**:
  - Memerlukan tuning yang hati-hati terhadap banyaknya hyperparameter.
  - Rentan terhadap overfitting jika tidak diatur dengan baik.

### 4. Gradient Boosting Regressor
- **Kelebihan**:
  - Mampu menghasilkan model yang sangat akurat.
  - Menangani berbagai jenis data, termasuk data non-linear.
- **Kekurangan**:
  - Memerlukan waktu yang lebih lama dalam proses pelatihan dibandingkan dengan beberapa model lainnya.
  - Rentan terhadap overfitting jika tidak diatur dengan tepat.

### 5. Random Forest Regressor
- **Kelebihan** (mengapa memilih model ini sebagai terbaik):
  - Mengurangi overfitting dengan rata-rata dari banyak pohon yang berbeda.
  - Tidak memerlukan tuning hyperparameter yang ekstensif seperti halnya beberapa model lain.
- **Kekurangan**:
  - Cenderung sulit untuk diinterpretasi dibandingkan dengan model linear.
  - Membutuhkan sumber daya komputasi yang cukup besar dibandingkan dengan model linear.

Setelah mendapatkan hasil performa masing-masing algoritma, ditemukan bahwa Random Forest Regressor adalah model terbaik. Model ini menghasilkan nilai MSE 4,70 dan nilai R2 sebesar 0,665. Setelah dilakukan hyperparameter tuning, Random Forest Regressor memberikan nilai R2 yang paling tinggi dibandingkan dengan model lainnya serta skor MSE yang paling kecil dari model lainnya. Hal ini disebabkan model Random Forest Regressor terdiri dari ensemble banyak decision tree yang membuatnya lebih stabil terhadap noise dalam data dan mengurangi overfitting.

## Evaluation
Proyek ini mengangkat masalah regresi sehingga digunakan nilai metrik berupa MSE dan R2. 

- **MSE (Mean Squared Error)**: MSE mengukur rata-rata dari kuadrat kesalahan atau perbedaan antara nilai yang diprediksi oleh model dengan nilai sebenarnya. Nilai MSE yang lebih rendah menunjukkan model yang lebih baik dalam memprediksi data.

- **R2 (R-squared)**: R2 mengukur proporsi variabilitas dalam data yang dapat dijelaskan oleh model. Nilai R2 berkisar antara 0 dan 1, dengan nilai yang lebih tinggi menunjukkan model yang lebih baik dalam menjelaskan variabilitas data.

| Model                                                     | MSE      | R2 Score |
|-----------------------------------------------------------|----------|----------|
| Linear Regression                                         | 4.892332 | 0.651570 |
| Random Forest Regressor with Hyperparameter Tuning        | 4.702618 | 0.665081 |
| MLP Regressor                                             | 5.164442 | 0.632190 |
| Gradient Boosting Regressor with Hyperparameter Tuning    | 4.812859 | 0.657230 |
| AdaBoost Regressor with Hyperparameter Tuning             | 4.769155 | 0.660342 |