# Menyelesaikan Permasalahan Perusahaan

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. Walaupun telah menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. Hal ini berimbas tingginya *attrition rate* (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%. Untuk mencegah hal ini semakin parah, manajer departemen HR ingin meminta bantuan Anda mengidentifikasi berbagai faktor yang mempengaruhi tingginya *attrition rate* tersebut. Selain itu, ia juga meminta Anda untuk membuat business dashboard untuk membantunya memonitori berbagai faktor tersebut.

### Permasalahan Bisnis

Permasalahan bisnis yang dihadapi oleh Jaya Jaya Maju adalah tingginya tingkat atrisi karyawan. Beberapa faktor yang mempengaruhi atrisi ini perlu diidentifikasi agar perusahaan bisa mengambil tindakan yang tepat.

Apabila tidak cepat diselesaikan, permasalahan tingkat atrisi karyawan yang tinggi dapat berdampak negatif pada perusahaan dalam beberapa cara:
1. **Biaya Rekrutmen**: Perusahaan harus mengeluarkan biaya tambahan untuk merekrut dan melatih karyawan baru.
2. **Penurunan Produktivitas**: Karyawan baru memerlukan waktu untuk beradaptasi, yang dapat menurunkan produktivitas secara keseluruhan.
3. **Citra Perusahaan**: Atrisi yang tinggi dapat menciptakan persepsi negatif di luar perusahaan, mempengaruhi reputasi dan kemampuan untuk menarik talenta baru.
4. **Kualitas Produk atau Layanan**: Dapat terjadi penurunan kualitas produk atau layanan karena kurangnya kontinuitas dalam tim.

Oleh karena itu, penting bagi Jaya Jaya Maju untuk mengidentifikasi faktor-faktor yang menyebabkan atrisi dan mengambil tindakan untuk mengatasinya, seperti meningkatkan kondisi kerja, memberikan peluang karir, dan menciptakan lingkungan kerja yang mendukung.

### Cakupan Proyek

Untuk mengatasi masalah tingkat atrisi karyawan, akan dilakukan upaya mengembangkan model machine learning menggunakan metode k-means clustering untuk mengidentifikasi faktor-faktor yang berkontribusi terhadap atrisi. Selanjutnya, akan dibuat dashboard visualisasi data dan laporan analisis data yang mendalam. Dashboard akan menggunakan *Tableau Public* untuk menunjukkan hasil analisis dari model machine learning.

Analisis akan mencakup pertanyaan-pertanyaan berikut:
1. Apakah terdapat korelasi antara usia dan tingkat atrisi? Apakah karyawan yang lebih muda lebih sering mengundurkan diri?
2. Bagaimana pengaruh kepuasan kerja (rendah, sedang, tinggi) terhadap tingkat atrisi? Apakah karyawan yang merasa tidak puas lebih cenderung mengundurkan diri?
3. Kaji apakah karyawan dengan masa kerja yang lebih pendek lebih cenderung mengundurkan diri.
4. Apakah beban kerja lembur mempengaruhi tingkat atrisi? Apakah karyawan yang sering lembur lebih cenderung mengundurkan diri?
5. Selidiki apakah ada perbedaan tingkat atrisi berdasarkan tingkat pendidikan yang lebih tinggi, seperti sarjana atau doktor.
6. Apakah jarak tempuh dari rumah ke tempat kerja berpengaruh terhadap tingkat atrisi? Apakah karyawan dengan waktu perjalanan yang lebih lama lebih cenderung mengundurkan diri?

Dengan menjawab pertanyaan-pertanyaan ini, diharapkan dapat menemukan solusi efektif untuk mengurangi tingkat atrisi di perusahaan Jaya Jaya Maju.

### Persiapan

Sumber data: https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee

Setup Environment - Anaconda:
```
conda create --name main-ds python=3.11
conda activate main-ds
pip install -r requirements.txt
```

Setup Environment - Shell/Terminal:
```
mkdir employee-attrition
cd employee-attrition
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Business Dashboard

Dalam *business dashboard* yang telah dibuat, dilakukan upaya untuk menjawab pertanyaan-pertanyaan kunci yang ada dalam lingkup proyek kami. Proyek ini menggunakan metode *k-means clustering*, yang telah mengungkapkan empat klaster berbeda—dilabeli sebagai Klaster 1, 2, 3, dan 4. Keputusan untuk membentuk empat klaster didasarkan pada Metode Elbow. Setelah meneliti karakteristik masing-masing klaster, kami menemukan bahwa Klaster 3 berkaitan dengan karyawan yang mengalami atrisi. Klaster 1 mewakili karyawan tanpa atrisi, sementara Klaster 2 dan 4 mengandung campuran keduanya dengan mayoritas karyawan tanpa atrisi. Untuk mengeksplorasi tingkat atrisi di berbagai klaster, silakan gunakan filter di kanan atas dashboard. Filter ini sangat penting untuk mengungkap atribut spesifik dari masing-masing klaster.

Link: https://public.tableau.com/shared/MHYBQ9KJJ?:display_count=n&:origin=viz_share_link

## Conclusion

Upaya untuk mencari tahu penyebab terjadi *attrition* pada karyawan dilakukan dengan *k-means clustering*. Lewat upaya tersebut, ditemukan 4 klaster. 
- Klaster 1 adalah klaster yang mana karyawannya tidak terjadi atrisi. Klaster 0 ditandai dengan usia rata-rata 30 tahun, jarak dari rumah ke kantor sebesar 7 km, pengalaman kerja selama 7 tahun, dan tidak sering kerja lembur.
- Klaster 2 terdiri dari karyawan yang sebagian besar tidak terjadi atrisi yang ditandai dengan usia rata-rata 40 tahun, jarak dari rumah ke kantor sebesar 7 km, pengalaman kerja selama 14 tahun, dan tidak sering kerja lembur.
- Klaster 3 terdiri dari karyawan yang sebagian besar terjadi atrisi yang ditandai dengan usia rata-rata 29 tahun, jarak dari rumah ke kantor sebesar 9 km, pengalaman kerja selama 7 tahun, dan sering kerja lembur.
- Klaster 4 terdiri dari karyawan yang sebagian besar tidak terjadi atrisi yang ditandai dengan usia rata-rata 45 tahun, jarak dari rumah ke kantor sebesar 6 km, pengalaman kerja selama 16 tahun, dan tidak sering kerja lembur.

Dari ciri-ciri klaster di atas, bisa disimpulkan faktor jarak dari rumah ke tempat kerja, bekerja secara lembur, dan usia muda adalah faktor penyebab atrisi. Oleh karena itu, diharapkan penemuan ini dapat bermanfaat bagi HR untuk memilih karyawannya dengan lebih baik lagi.

### Rekomendasi Action Items

Berikut beberapa rekomendasi action items yang harus dilakukan perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.

- Mencari karyawan yang memiliki tempat tinggal yang dekat, maksimal 10 km. Jarak tempat tinggal yang dekat dengan tempat kerja dapat mengurangi stres dan kelelahan akibat perjalanan panjang.
- Mengurangi pekerjaan yang bersifat lembur. Pekerjaan lembur yang berlebihan dapat menyebabkan kelelahan, stres, dan ketidakpuasan kerja. Karyawan yang sering lembur mungkin merasa kewalahan dan kurang dihargai.
- Mencari karyawan dengan usia tua, yaitu di rentang 35-50 tahun. Karyawan yang lebih tua cenderung memiliki pengalaman dan stabilitas yang lebih tinggi. Mereka mungkin lebih berkomitmen pada pekerjaan mereka dan kurang cenderung untuk berpindah pekerjaan dibandingkan dengan karyawan yang lebih muda.
