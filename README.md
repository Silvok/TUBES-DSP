---

# Tugas Besar Mata Kuliah Digital Signal Processing (IF3024)

## Dosen Pengampu: **Martin Clinton Tosima Manullang, S.T., M.T.**

# **Sistem Pemantauan Sinyal Respirasi & rPPG**
---

## **Anggota Kelompok**

| **Nama**                    | **NIM**   | **ID GitHub**                            |
| --------------------------- | --------- | ------------------------------------------ |
| Lois Novel E Gurning        | 122140098 | [crngidlrey](https://github.com/crngidlrey)|
| Silva Oktaria Putri         | 122140085 | [Silvok](https://github.com/Silvok)        |

---

## **Deskripsi Proyek**

Proyek ini bertujuan untuk mengembangkan **Sistem Pemantauan Sinyal Respirasi dan rPPG** menggunakan teknik pemrosesan sinyal digital dan pemrograman Python. Sistem ini dirancang untuk memantau serta menganalisis sinyal respirasi secara real-time melalui teknologi *Remote Photoplethysmography* (rPPG), yang memanfaatkan analisis video.

### Langkah-langkah Utama dalam Proyek:
- **Pengumpulan Data**: Menggunakan kamera untuk menangkap video yang menjadi dasar analisis sinyal respirasi.
- **Pemrosesan Video**: Mendeteksi wajah dan landmark menggunakan library *MediaPipe*, untuk mengekstraksi titik-titik penting pada wajah guna mendeteksi perubahan warna darah.
- **Analisis Sinyal**: Memfilter dan memproses data video menggunakan teknik pemfilteran digital untuk menghasilkan sinyal respirasi.
- **Visualisasi Data**: Membuat antarmuka grafis untuk menampilkan sinyal respirasi secara real-time sehingga memudahkan pengguna dalam memantau kondisi pernapasan.
- **Pengujian dan Evaluasi**: Melakukan pengujian sistem untuk memastikan akurasi dan efektivitas dalam mendeteksi sinyal respirasi.

Dengan pendekatan ini, sistem diharapkan dapat memberikan informasi yang akurat dan real-time terkait kondisi pernapasan pengguna.

---

## **Teknologi yang Digunakan**

Berikut adalah teknologi dan alat utama yang digunakan dalam proyek ini:

| Logo                                                                                                                           | Nama Teknologi | Fungsi                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------ | -------------- | -------------------------------------------------------------------------------- |
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" alt="Python Logo" width="60">            | Python         | Bahasa pemrograman utama untuk pengembangan filter dan analisis sinyal.          |
| <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" alt="VS Code Logo" width="60"> | VS Code        | Editor teks untuk pengembangan skrip secara efisien dengan dukungan ekstensi Python. |

---

## **Library yang Digunakan**

Berikut adalah daftar library Python yang digunakan dalam proyek ini beserta fungsinya:

| **Library**                | **Fungsi**                                                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `cv2`                      | Menangkap gambar dari kamera dan melakukan pemrosesan gambar secara langsung.                     |
| `mediapipe`                | Mendeteksi landmark wajah seperti posisi hidung untuk membantu deteksi gerakan kepala.             |
| `numpy`, `scipy`           | Digunakan untuk operasi matematika dan pembuatan program pengolahan sinyal.                        |

---
