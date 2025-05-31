"""
Aplikasi GUI monitoring realtime rPPG (detak jantung dari wajah) dan respirasi (pernapasan dari bahu & dada)
berbasis PyQt5 dan MediaPipe. Sinyal diambil dari webcam, diproses, divisualisasikan, dan dicatat ke file CSV.
"""

import sys  # Modul untuk sistem Python
import os   # Modul untuk operasi file dan path
import cv2  # OpenCV untuk akses webcam dan pemrosesan gambar
import numpy as np  # Untuk operasi numerik dan array
import mediapipe as mp  # Untuk deteksi landmark wajah dan pose tubuh
import csv  # Untuk pencatatan data ke file CSV
import time  # Untuk pengaturan waktu dan timestamp
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout  # Komponen GUI PyQt5
from PyQt5.QtCore import QTimer, Qt  # Timer dan alignment dari PyQt5
from PyQt5.QtGui import QImage, QPixmap  # Untuk menampilkan gambar di PyQt5
import pyqtgraph as pg  # Untuk plotting sinyal secara real-time

from code.signal_processing import clean_signal, estimate_bpm_peaks  # Fungsi pemrosesan sinyal

def extract_pos_signal(rgb_buffer):
    """
    Mengimplementasikan algoritma POS (Plane-Orthogonal-to-Skin) untuk menghasilkan sinyal rPPG mentah.
    Parameter:
        rgb_buffer (np.ndarray): Array shape (N, 3) berisi rata-rata RGB ROI dahi untuk N frame.
    Return:
        pos_signal (np.ndarray): Sinyal rPPG mentah hasil POS.
    """
    # 1. Normalisasi setiap kanal warna
    X = rgb_buffer.T  # shape (3, N)
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True)
    Xn = (X - X_mean) / (X_std + 1e-8)  # Normalisasi per kanal

    # 2. Proyeksikan ke dua vektor orthogonal
    S = np.vstack([
        Xn[0, :] - Xn[1, :],  # R - G
        Xn[0, :] + Xn[1, :] - 2 * Xn[2, :]  # R + G - 2B
    ])  # shape (2, N)

    # 3. Standarisasi S
    std1 = np.std(S[0, :])
    std2 = np.std(S[1, :])
    if std2 == 0:
        std2 = 1e-8
    h = S[0, :] + (std1 / std2) * S[1, :]

    # 4. Hilangkan mean (zero-mean)
    h = h - np.mean(h)
    return h

class SignalMonitor(QWidget):
    """
    Kelas utama untuk aplikasi monitoring sinyal rPPG dan respirasi secara real-time.
    Menampilkan video webcam, plot sinyal, serta menghitung dan mencatat BPM/BRPM.
    """

    def __init__(self):
        """
        Inisialisasi objek SignalMonitor, setup GUI, MediaPipe, webcam, timer, buffer sinyal, dan file log.
        """
        super().__init__()  # Inisialisasi QWidget
        self.setWindowTitle("Realtime rPPG & Respirasi Monitor")  # Judul window
        self.setGeometry(100, 100, 1200, 700)  # Ukuran dan posisi window
        self.init_ui()  # Setup tampilan GUI
        self.init_mediapipe()  # Setup model MediaPipe

        self.cap = cv2.VideoCapture(1)  # Membuka webcam (ganti ke 0 jika webcam utama)
        self.timer = QTimer()  # Timer untuk update frame video
        self.timer.timeout.connect(self.update_frame)  # Koneksi timer ke fungsi update_frame
        self.timer.start(30)  # Update setiap 30 ms (~33 fps)

        self.rgb_buffer = []  # Buffer untuk rata-rata RGB ROI dahi
        self.rppg_buffer = []  # Buffer untuk sinyal rPPG (hasil POS)
        self.resp_buffer = []  # Buffer untuk sinyal respirasi (bahu & dada)
        self.max_buffer = 600  # Maksimal panjang buffer (20 detik jika 30 fps)
        self.last_rgb = None  # Simpan nilai RGB terakhir jika ROI tidak terdeteksi

        self.start_time = time.time()  # Waktu awal untuk logging
        self.log_interval = 60  # Interval logging ke CSV (detik)

        log_filename = 'signal_log.csv'  # Nama file log
        file_exists = os.path.isfile(log_filename)  # Cek apakah file sudah ada
        self.csv_file = open(log_filename, 'a', newline='')  # Buka file log (append mode)
        self.csv_writer = csv.writer(self.csv_file)  # Writer untuk file CSV
        if not file_exists:
            self.csv_writer.writerow(['Timestamp', 'HeartRate_BPM', 'Respiration_BPM'])  # Header jika file baru

    def init_ui(self):
        """
        Membuat dan mengatur layout GUI, label video, serta plot untuk sinyal rPPG dan respirasi.
        """
        # Label untuk video webcam
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)  # Ukuran tampilan video
        self.video_label.setAlignment(Qt.AlignCenter)  # Posisikan di tengah

        # Plot sinyal rPPG (dahi)
        self.rppg_plot = pg.PlotWidget(title="Sinyal rPPG (Dahi)")
        self.rppg_plot.setBackground('#0e1242')  # Warna latar belakang plot
        self.rppg_curve = self.rppg_plot.plot(pen=pg.mkPen('#acec00', width=2.5))  # Warna dan ketebalan garis plot

        # Plot sinyal respirasi (bahu & dada)
        self.resp_plot = pg.PlotWidget(title="Sinyal Respirasi (Bahu & Dada)")
        self.resp_plot.setBackground('#0e1242')
        self.resp_curve = self.resp_plot.plot(pen=pg.mkPen('#ffc01d', width=2.5))

        # Layout atas hanya untuk video, diposisikan tengah
        video_layout = QHBoxLayout()
        video_layout.addStretch()
        video_layout.addWidget(self.video_label)
        video_layout.addStretch()

        # Layout bawah untuk grafik sinyal
        bottom_layout = QVBoxLayout()
        bottom_layout.addWidget(self.rppg_plot)
        bottom_layout.addWidget(self.resp_plot)

        # Gabungkan semua layout ke dalam main layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(video_layout)
        main_layout.addLayout(bottom_layout)

        self.setStyleSheet("background-color: #3a457c; color: white;")  # Warna latar belakang aplikasi
        self.setLayout(main_layout)  # Set layout utama

    def init_mediapipe(self):
        """
        Inisialisasi model MediaPipe untuk face mesh (wajah) dan pose (tubuh).
        """
        # Model face mesh untuk deteksi landmark wajah
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        # Model pose untuk deteksi landmark tubuh
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=False, min_detection_confidence=0.5)

    def update_frame(self):
        """
        Membaca frame dari webcam, mendeteksi ROI pada wajah dan tubuh, mengambil sinyal,
        memproses dan memplot sinyal, menghitung BPM/BRPM, menampilkan ke GUI, dan mencatat ke file log.
        """
        ret, frame = self.cap.read()  # Ambil frame dari webcam
        if not ret:
            return  # Jika gagal, keluar dari fungsi

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Konversi ke RGB untuk MediaPipe
        h, w, _ = frame.shape  # Ambil dimensi frame

        # Deteksi landmark wajah dengan MediaPipe Face Mesh
        face_results = self.mp_face_mesh.process(frame_rgb)
        forehead_roi = None  # ROI dahi

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark  # Landmark wajah pertama
            left_eye = landmarks[33]  # Titik sudut mata kiri
            right_eye = landmarks[263]  # Titik sudut mata kanan
            eye_top = landmarks[159]  # Titik atas mata
            eye_bottom = landmarks[145]  # Titik bawah mata

            x1 = int(min(left_eye.x, right_eye.x) * w)  # Koordinat x kiri
            x2 = int(max(left_eye.x, right_eye.x) * w)  # Koordinat x kanan
            eye_center_y = int((eye_top.y + eye_bottom.y) / 2 * h)  # Titik tengah y mata

            roi_width = x2 - x1  # Lebar ROI dahi
            roi_height = int(roi_width * 0.5)  # Tinggi ROI dahi

            fx = x1
            fy = eye_center_y - roi_height - 10  # Offset ke atas dari mata
            fx, fy = max(0, fx), max(0, fy)  # Pastikan tidak negatif
            fx2, fy2 = min(w, fx + roi_width), min(h, fy + roi_height)  # Batas kanan bawah ROI

            cv2.rectangle(frame, (fx, fy), (fx2, fy2), (0, 236, 172), 2)  # Gambar kotak ROI dahi
            if fx2 > fx and fy2 > fy:
                forehead_roi = frame[fy:fy2, fx:fx2]  # Ambil ROI dahi

        # Ekstraksi rata-rata RGB dari ROI dahi untuk buffer POS
        if forehead_roi is not None and forehead_roi.size > 0:
            red_mean   = np.mean(forehead_roi[:, :, 2])
            green_mean = np.mean(forehead_roi[:, :, 1])
            blue_mean  = np.mean(forehead_roi[:, :, 0])
            rgb = [red_mean, green_mean, blue_mean]
            self.last_rgb = rgb
            self.rgb_buffer.append(rgb)
        elif self.last_rgb is not None:
            self.rgb_buffer.append(self.last_rgb)

        # Batasi panjang buffer RGB
        if len(self.rgb_buffer) > self.max_buffer:
            self.rgb_buffer = self.rgb_buffer[-self.max_buffer:]

        # Proses POS jika buffer cukup
        if len(self.rgb_buffer) > 60:
            rgb_arr = np.array(self.rgb_buffer[-self.max_buffer:])  # shape (N, 3)
            pos_signal = extract_pos_signal(rgb_arr)
            # Batasi panjang buffer rPPG agar sinkron dengan buffer lain
            self.rppg_buffer = list(pos_signal[-self.max_buffer:])
        else:
            self.rppg_buffer = []

        # Deteksi landmark pose tubuh dengan MediaPipe Pose
        pose_results = self.mp_pose.process(frame_rgb)
        if pose_results.pose_landmarks:
            lm = pose_results.pose_landmarks.landmark
            l_shoulder = lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]  # Titik bahu kiri
            r_shoulder = lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]  # Titik bahu kanan
            lx, ly = int(l_shoulder.x * w), int(l_shoulder.y * h)
            rx, ry = int(r_shoulder.x * w), int(r_shoulder.y * h)
            roi_shoulder_y1 = min(ly, ry) - int(0.05 * h)
            roi_shoulder_y2 = min(ly, ry) + int(0.10 * h)
            roi_shoulder_x1 = max(0, min(lx, rx) - int(0.15 * w))
            roi_shoulder_x2 = min(w, max(lx, rx) + int(0.15 * w))
            cv2.rectangle(frame, (roi_shoulder_x1, roi_shoulder_y1), (roi_shoulder_x2, roi_shoulder_y2), (29, 192, 255), 2)  # ROI bahu
            roi_shoulder = frame[roi_shoulder_y1:roi_shoulder_y2, roi_shoulder_x1:roi_shoulder_x2]
            shoulder_mean = np.mean(roi_shoulder[:, :, 0]) if roi_shoulder.size > 0 else 0  # Kanal biru

            chest_x1 = roi_shoulder_x1 + int(0.25 * (roi_shoulder_x2 - roi_shoulder_x1))
            chest_x2 = roi_shoulder_x2 - int(0.25 * (roi_shoulder_x2 - roi_shoulder_x1))
            chest_y1 = roi_shoulder_y2
            chest_y2 = chest_y1 + int(0.18 * h)
            cv2.rectangle(frame, (chest_x1, chest_y1), (chest_x2, chest_y2), (29, 192, 255), 2)  # ROI dada
            roi_chest = frame[chest_y1:chest_y2, chest_x1:chest_x2]
            chest_mean = np.mean(roi_chest[:, :, 0]) if roi_chest.size > 0 else 0  # Kanal biru

            # Gabungkan sinyal bahu & dada untuk respirasi
            if shoulder_mean > 0 and chest_mean > 0:
                self.resp_buffer.append((shoulder_mean + chest_mean) / 2)
            elif shoulder_mean > 0:
                self.resp_buffer.append(shoulder_mean)
            elif chest_mean > 0:
                self.resp_buffer.append(chest_mean)

        # Batasi panjang buffer sinyal
        if len(self.resp_buffer) > self.max_buffer:
            self.resp_buffer = self.resp_buffer[-self.max_buffer:]

        hr, br = 0.0, 0.0  # Inisialisasi nilai BPM dan BRPM
        hr_text, br_text = "-- BPM", "-- BRPM"  # Default teks

        # Proses sinyal rPPG jika buffer cukup
        if len(self.rppg_buffer) > 60:
            rppg_arr = np.array(self.rppg_buffer)
            rppg_arr = clean_signal(rppg_arr)  # Filter sinyal
            self.rppg_curve.setData(rppg_arr)  # Update plot
            hr = estimate_bpm_peaks(rppg_arr, 30, min_bpm=40, max_bpm=180)  # Estimasi BPM
            hr_text = f"{hr:.1f} BPM" if hr > 0 else "-- BPM"
        else:
            self.rppg_curve.setData([])

        # Proses sinyal respirasi jika buffer cukup
        if len(self.resp_buffer) > 60:
            resp_arr = np.array(self.resp_buffer)
            resp_arr = clean_signal(resp_arr)
            self.resp_curve.setData(resp_arr)
            br = estimate_bpm_peaks(resp_arr, 30, min_bpm=10, max_bpm=30)  # Estimasi BRPM
            br_text = f"{br:.1f} BRPM" if br > 0 else "-- BRPM"
        else:
            self.resp_curve.setData([])

        # Logging ke file CSV setiap interval waktu
        current_time = time.time()
        if current_time - self.start_time >= self.log_interval:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
            self.csv_writer.writerow([timestamp, f"{hr:.1f}", f"{br:.1f}"])
            self.csv_file.flush()
            self.start_time = current_time

        # Gambar teks BPM dan BRPM ke dalam frame video
        cv2.putText(frame, f"BPM: {hr_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (66, 18, 14), 2)
        cv2.putText(frame, f"BRPM: {br_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (66, 18, 14), 2)

        # Konversi frame ke QImage dan tampilkan di label video
        img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_BGR888)
        self.video_label.setPixmap(QPixmap.fromImage(img))

    def closeEvent(self, event):
        """
        Menangani event saat aplikasi ditutup: menutup webcam dan file log.

        Parameters:
            event (QCloseEvent): Event penutupan aplikasi.
        """
        self.cap.release()  # Lepaskan webcam
        self.csv_file.close()  # Tutup file log
        event.accept()  # Terima event penutupan