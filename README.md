---
# **Sistem Pemantauan Sinyal Respirasi & Detak Jantung dengan rPPG**
Tugas Besar Mata Kuliah Digital Signal Processing (IF3024)
---

## **Anggota Kelompok**

| **Nama**                    | **NIM**   | **ID GitHub**                            |
| --------------------------- | --------- | ------------------------------------------ |
| Lois Novel E Gurning        | 122140098 | [crngidlrey](https://github.com/crngidlrey)|
| Silva Oktaria Putri         | 122140085 | [Silvok](https://github.com/Silvok)        |

## **Deskripsi Proyek**

<div align="justify">
Proyek ini bertujuan untuk mengembangkan Sistem Pemantauan Sinyal Respirasi dan Detak Jantung dengan rPPG menggunakan teknik pemrosesan sinyal digital dan bahasa pemrograman Python. Sistem ini dirancang untuk memantau sinyal respirasi dan detak jantung secara real-time melalui teknologi *Remote Photoplethysmography* (rPPG) berbasis webcam, dengan deteksi landmark wajah dan tubuh menggunakan MediaPipe. Hasil estimasi BPM (detak jantung) dan BRPM (pernapasan) divisualisasikan secara langsung dan dapat diekspor ke file CSV.
</div>

## **Tools Utama yang Digunakan**

| Logo                                                                                                                          | Nama Teknologi | Fungsi                            |
|-------------------------------------------------------------------------------------------------------------------------------|----------------|-----------------------------------|
| <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg" width="40">                             | Python         | Bahasa pemrograman utama          |
| <img src="https://upload.wikimedia.org/wikipedia/commons/9/9a/Visual_Studio_Code_1.35_icon.svg" width="40">                   | VS Code        | Editor kode utama                 |

## **Library Utama yang Digunakan**

| **Library**      | **Fungsi**                                                                                  |
|------------------|--------------------------------------------------------------------------------------------|
| opencv-python    | Akuisisi video dari webcam, pemrosesan gambar                                              |
| mediapipe        | Deteksi landmark wajah (face mesh) & tubuh (pose)                                          |
| numpy            | Operasi array, statistik, dan numerik                                                      |
| scipy            | Filter sinyal, deteksi puncak, pemrosesan sinyal digital                                   |
| PyQt5            | Membuat antarmuka grafis (GUI)                                                             |
| pyqtgraph        | Plotting sinyal secara real-time                                                           |

## **Struktur Proyek**

```plaintext
TUBES/
├── __pycache__/
│   ├── main_gui.cpython-312.pyc
│   └── signal_processing.cpython-312.pyc
├── code/
│   ├── __pycache__/
│   │   ├── __init__.cpython-312.pyc
│   │   ├── main_gui.cpython-312.pyc
│   │   └── signal_processing.cpython-312.pyc
│   ├── __init__.py
│   ├── main_gui.py
│   ├── signal_processing.py
│   └── app.py
├── README.md
├── requirements.txt
└── signal_log.csv
```

## **Logbook Mingguan**

| Minggu  | Tanggal                    | Kegiatan                                                                     |
|---------|----------------------------|------------------------------------------------------------------------------|
| 1       | 27 April - 03 Mei 2025     | Mencari Referensi, brainstorming ide.                                        |
| 2       | 04 - 10 Mei 2025           | Membuat repository github, mempelajari latex, fiksasi ide.                   |
| 3       | 11 - 17 Mei 2025           | Mulai menggarap code.                                                        |
| 4       | 18 - 24 Mei 2025           | Mulai menyusun laporan, revisi code.                                         |
| 5       | 25 - 31 Mei                | Revisi code, finalisasi laporan, finalisasi code, submit tugas di gform.     |

## **Panduan Instalasi**

1. **Clone repository**  
   ```bash
   git clone https://github.com/Silvok/TUBES-DSP
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## **Panduan Penggunaan**

1. Pastikan device yang digunakan telah terhubung dengan webcam.

2. Jalankan aplikasi:

   ```bash
   python app.py
   ```

3. Pastikan wajah dan bahu terlihat jelas di kamera.
4. Pastikan terdapat pencahayaan yang cukup (cahaya alami matahari lebih disarankan).
5. Setelah aplikasi dimulai, butuh sedikit waktu untuk aplikasi dapat memproses dan mengolah sinyal.
6. Aplikasi akan menampilkan video yang ditangkap oleh webcam, nilai BRPM (breath per minute) dan BPM (beat per minute), beserta visualisasi sinyal keduanya.
7. Setiap BPM dan BRPM yang berhasil ditangkap, akan disimpan di ```signal_log.csv``` secara realtime.
8. Untuk menutup aplikasi, tekan tombol X di pojok kanan atas.

## **Catatan Tambahan**

1. Jika webcam tidak terdeteksi, ubah index pada ```cv2.VideoCapture(1)``` di ```main_gui.py``` menjadi ```cv2.VideoCapture(0)```.
2. Program sudah diuji dan berjalan dengan lancar di sistem operasi Windows 11, serta versi Python 3.12.9
3. Jika ada error dependency, update pip dan install ulang requirements.

   ```bash
   python -m pip install --upgrade pip
   ```
---