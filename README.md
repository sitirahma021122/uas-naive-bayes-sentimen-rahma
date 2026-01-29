# KLASIFIKASI SENTIMEN TEKS MENGGUNAKAN ALGORITMA NAIVE BAYES

## Identitas Mahasiswa
Nama : Siti Rahmah  
NIM  : 24146075  

## Deskripsi Project
Project ini merupakan project Ujian Akhir Semester (UAS) yang bertujuan untuk membangun
sebuah sistem klasifikasi sentimen teks komentar menggunakan algoritma Naive Bayes.
Sistem ini digunakan untuk mengklasifikasikan komentar ke dalam tiga kategori sentimen,
yaitu positif, negatif, dan netral.

## Dataset
Dataset yang digunakan adalah file `dataset_sentimen.csv` yang berisi dua kolom utama:
- review_text : teks komentar pengguna
- sentiment   : label sentimen  
  (2 = Positif, 1 = Netral, 0 = Negatif)

Dataset digunakan sebagai data latih dan data uji dalam proses klasifikasi sentimen.

## Tahapan Sistem
1. Preprocessing teks (case folding, tokenizing, stopword removal, stemming)
2. Pembagian dataset menjadi 80% data latih dan 20% data uji
3. Ekstraksi fitur menggunakan TF-IDF
4. Klasifikasi menggunakan algoritma Multinomial Naive Bayes
5. Evaluasi model menggunakan classification report dan confusion matrix

## Struktur File
- rahma.py : file utama untuk menjalankan program
- preprocessing_rahma.py : preprocessing teks
- train_model_rahma.py : proses training dan evaluasi model
- dataset_sentimen.csv : dataset komentar

## Cara Menjalankan Program
Jalankan perintah berikut pada terminal:
