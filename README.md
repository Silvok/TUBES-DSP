# Tugas Besar Mata Kuliah Digital Processing Signal (IF3024)

## Dosen Pengampu: **Martin Clinton Tosima Manullang, S.T., M.T..**

# **Sistem Pemantauan Sinyal Respirasi & rPPG**
---

## **Anggota Kelompok**

| **Nama**                    | **NIM**   | **ID GITHUB**                                                               |
| --------------------------- | --------- | --------------------------------------------------------------------------- |
| Lois Novel E Gurning        | 122140098 | <a href="https://github.com/crngidlrey</a> |
| Silva Oktaria Putri         | 122140085 | <a href="https://github.com/Silvok</a>                     |

---

## **Deskripsi Proyek**

Proyek ini bertujuan untuk mengembangkan Sistem Pemantauan Sinyal Respirasi dan rPPG menggunakan teknik pemrosesan sinyal digital dan pemrograman Python. Sistem ini dirancang untuk memantau dan menganalisis sinyal respirasi dengan memanfaatkan analisis video melalui teknologi Remote Photoplethysmography (rPPG). Berikut adalah langkah-langkah utama yang dilakukan dalam proyek ini:

Pengumpulan Data: Menggunakan kamera untuk menangkap video yang diperlukan dalam analisis sinyal respirasi.
Pemrosesan Video: Mengimplementasikan teknik deteksi wajah dan landmark menggunakan library Mediapipe untuk mengekstraksi titik-titik penting dari wajah, yang diperlukan untuk menghitung perubahan warna darah.
Analisis Sinyal: Melakukan analisis sinyal respirasi dengan memfilter dan memproses data yang diperoleh dari video, menggunakan teknik pemfilteran digital.
Visualisasi Data: Membuat antarmuka grafis untuk menampilkan sinyal respirasi secara real-time, sehingga pengguna dapat monitor kondisi respirasi dengan lebih mudah.
Pengujian dan Evaluasi: Melakukan pengujian sistem untuk memastikan akurasi dan efektivitas dalam memantau sinyal respirasi.
Dengan metode ini, sistem diharapkan dapat memberikan informasi yang akurat dan real-time mengenai kondisi respirasi pengguna.

---

## **Teknologi yang Digunakan**

Berikut adalah teknologi dan alat yang digunakan dalam proyek ini:

| Logo                                                                                                                           | Nama Teknologi | Fungsi                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------- | -------------------------------------------------------------------------------- |
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="60">            | Python         | Bahasa pemrograman utama untuk pengembangan filter.                              |
| <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" alt="VS Code Logo" width="60"> |
VS Code        | Editor teks untuk mengedit skrip secara efisien dengan dukungan ekstensi Python.

---

## **Library yang Digunakan**

Berikut adalah daftar library Python yang digunakan dalam proyek ini, beserta fungsinya:

| **Library**                | **Fungsi**                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `cv2`                      | Digunakan untuk menangkap gambar dari kamera dan memproses gambar secara langsung.                 |
| `mediapipe`                | Digunakan untuk mendeteksi landmark wajah, seperti posisi hidung, untuk mendeteksi gerakan kepala. |
| `jupyternotebook`          | Digunakan sebagai media explorasi model dan landmark yang sesuai.                                  |
| `Scipy`, `numpy`           | Digunakan untuk bahan oprasi pembuatan program                                                     |

---
