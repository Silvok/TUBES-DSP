import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, filtfilt

# --- Konfigurasi ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Parameter rPPG
ROI_FACE_POINTS = [10, 151, 338, 46, 276, 280, 281, 282, 300, 301, 302, 335, 336] # Contoh keypoints wajah dari MediaPipe Face Mesh
BUFFER_SIZE = 150 # Jumlah frame untuk analisis rPPG
FPS = 30 # Asumsi FPS kamera, sesuaikan jika perlu

# Filter sinyal
# Sinyal detak jantung biasanya antara 0.8 Hz (48 BPM) dan 3 Hz (180 BPM)
# Sinyal respirasi biasanya antara 0.1 Hz (6 napas/menit) dan 0.5 Hz (30 napas/menit)
LOWCUT_HR = 0.8
HIGHCUT_HR = 3.0
LOWCUT_RESP = 0.1
HIGHCUT_RESP = 0.5
ORDER = 2 # Orde filter Butterworth

# Fungsi untuk filter bandpass
def bandpass_filter(data, lowcut, highcut, fs, order=ORDER):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# --- Inisialisasi Kamera dan MediaPipe ---
cap = cv2.VideoCapture(0) # 0 untuk webcam default, sesuaikan jika ada kamera lain
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()

# Inisialisasi MediaPipe Pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Inisialisasi untuk Lucas-Kanade Optical Flow (jika diperlukan untuk rPPG tracking)
# params_lk = dict(winSize=(15, 15),
#                  maxLevel=2,
#                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# old_gray = None
# p0 = None # Initial points for tracking

# Buffers untuk sinyal
r_buffer = []
g_buffer = []
b_buffer = []
respiration_buffer = []

print("Memulai program. Tekan 'q' untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak bisa menerima frame (stream end?). Keluar ...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # --- MediaPipe Pose untuk Respirasi ---
    results_pose = pose.process(frame_rgb)

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results_pose.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Contoh: Deteksi pergerakan dada/perut untuk respirasi
        # Anda bisa memilih landmark yang relevan, misalnya:
        # RIGHT_SHOULDER (11), LEFT_SHOULDER (12), RIGHT_HIP (23), LEFT_HIP (24)
        # Atau landmark di tengah badan untuk pergerakan vertikal
        
        # Contoh sederhana: Pergerakan vertikal rata-rata dari bahu dan pinggul
        # Ini perlu disempurnakan!
        try:
            r_shoulder_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            l_shoulder_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            r_hip_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
            l_hip_y = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
            
            avg_body_y = (r_shoulder_y + l_shoulder_y + r_hip_y + l_hip_y) / 4
            respiration_buffer.append(avg_body_y)
            
            # Batasi buffer respirasi
            if len(respiration_buffer) > BUFFER_SIZE * 2: # Lebih panjang untuk menangkap siklus pernapasan
                respiration_buffer.pop(0)

            # Analisis respirasi (setelah cukup data)
            if len(respiration_buffer) > FPS * 10: # Minimal 10 detik data
                filtered_respiration = bandpass_filter(np.array(respiration_buffer), LOWCUT_RESP, HIGHCUT_RESP, FPS)
                # Anda bisa melakukan FFT atau Peak Detection di sini
                # Untuk penyederhanaan, kita hanya menunjukkan sinyalnya.
                # Nanti perlu implementasi puncak-ke-puncak untuk RPM.
                
                # Contoh: Menampilkan sinyal respirasi di konsol
                # print(f"Sinyal Respirasi: {filtered_respiration[-1]:.4f}")

        except Exception as e:
            # print(f"Error pada deteksi respirasi: {e}")
            pass # Lanjutkan jika tidak ada landmark

    # --- rPPG (Ekstraksi Sinyal Warna dari Wajah) ---
    # Ini memerlukan MediaPipe Face Mesh atau deteksi wajah lain untuk ROI yang akurat
    # Untuk contoh ini, kita asumsikan MediaPipe Face Mesh juga digunakan,
    # atau kita bisa mengestimasikan area wajah dari pose landmarks.
    
    # Pendekatan sederhana: Ambil area dahi/pipi jika pose terdeteksi
    if results_pose.pose_landmarks:
        # Asumsi: Dahi berada di atas hidung dan di antara mata
        # Landmark hidung (0), mata kanan (2), mata kiri (5) dari MediaPipe Face Mesh
        # Karena kita hanya pakai Pose, kita estimasi dari posisi kepala
        
        # Ini perlu diganti dengan deteksi wajah yang lebih robust (misal MediaPipe Face Mesh)
        # atau ROI statis jika kepala tidak bergerak banyak.
        
        # Jika menggunakan MediaPipe Face Mesh (rekomendasi untuk rPPG akurat)
        # mp_face_mesh = mp.solutions.face_mesh
        # face_mesh = mp_face_mesh.FaceMesh()
        # results_face_mesh = face_mesh.process(frame_rgb)
        
        # if results_face_mesh.multi_face_landmarks:
        #     for face_landmarks in results_face_mesh.multi_face_landmarks:
        #         # Ambil rata-rata piksel dari area wajah (misal dahi)
        #         # Ini perlu implementasi yang lebih detail untuk memilih ROI
        #         # Contoh: ambil koordinat dari ROI_FACE_POINTS dan rata-ratakan warna
        #         # Anda bisa menggunakan bounding box dari wajah dan mengambil area tengah atas.
        
        #         # Untuk demo, mari kita ambil kotak kecil di tengah dahi berdasarkan pose
        #         nose_landmark = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        #         left_eye_inner = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER]
        #         right_eye_inner = results_pose.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER]

        #         # Estimasi posisi dahi
        #         center_x = int(nose_landmark.x * frame.shape[1])
        #         center_y = int(nose_landmark.y * frame.shape[0])
                
        #         # Asumsi dahi ada di atas hidung, kira-kira 1/3 jarak antara hidung dan bagian atas kepala
        #         # Ini sangat kasar, lebih baik gunakan Face Mesh
        #         roi_y_start = max(0, center_y - int(frame.shape[0] * 0.1))
        #         roi_y_end = max(0, center_y - int(frame.shape[0] * 0.05)) # area di atas hidung
        #         roi_x_start = max(0, center_x - int(frame.shape[1] * 0.03))
        #         roi_x_end = min(frame.shape[1], center_x + int(frame.shape[1] * 0.03))

        #         # Ambil rata-rata warna di ROI
        #         if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
        #             roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        #             if roi.size > 0:
        #                 avg_b, avg_g, avg_r = np.mean(roi, axis=(0, 1))
        #                 r_buffer.append(avg_r)
        #                 g_buffer.append(avg_g)
        #                 b_buffer.append(avg_b)

        #                 # Gambar ROI
        #                 cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        #         else:
        #             # print("ROI tidak valid.")
        #             pass # Tidak ada ROI yang valid

        # Pendekatan rPPG yang lebih baik dengan Lucas-Kanade (jika tidak ada Face Mesh)
        # Ini membutuhkan titik awal yang stabil
        # Untuk kasus ini, karena kita pakai Pose, kita bisa pilih titik di dahi sebagai titik awal
        
        # Ambil rata-rata intensitas piksel dari area yang luas (misalnya bagian atas wajah)
        # Ini lebih simpel, tapi rentan terhadap gerakan
        h, w, _ = frame.shape
        # Pilih area sekitar dahi, contoh: 1/4 lebar di tengah, 1/8 tinggi di bagian atas
        roi_x_start = int(w * 0.4)
        roi_x_end = int(w * 0.6)
        roi_y_start = int(h * 0.05)
        roi_y_end = int(h * 0.15)
        
        if roi_y_end > roi_y_start and roi_x_end > roi_x_start:
            roi_face = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            if roi_face.size > 0:
                avg_b, avg_g, avg_r = np.mean(roi_face, axis=(0, 1))
                r_buffer.append(avg_r)
                g_buffer.append(avg_g)
                b_buffer.append(avg_b)
                cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 0), 2)
        
    # Batasi buffer rPPG
    if len(r_buffer) > BUFFER_SIZE:
        r_buffer.pop(0)
        g_buffer.pop(0)
        b_buffer.pop(0)

    # Analisis rPPG (setelah cukup data)
    if len(g_buffer) == BUFFER_SIZE:
        # Menggunakan saluran G (hijau) karena paling sensitif terhadap perubahan volume darah
        g_signal = np.array(g_buffer)
        
        # Normalisasi (penting untuk mengurangi efek pencahayaan)
        normalized_g = (g_signal - np.mean(g_signal)) / np.std(g_signal)
        
        # Filter sinyal
        filtered_g_signal = bandpass_filter(normalized_g, LOWCUT_HR, HIGHCUT_HR, FPS)
        
        # Menghitung BPM (Beat Per Minute)
        # Menggunakan Fast Fourier Transform (FFT) untuk menemukan frekuensi dominan
        N = len(filtered_g_signal)
        yf = np.fft.fft(filtered_g_signal)
        xf = np.fft.fftfreq(N, 1 / FPS)
        
        # Ambil frekuensi positif
        idx = np.where((xf > LOWCUT_HR) & (xf < HIGHCUT_HR))
        if len(idx[0]) > 0:
            dominant_freq_idx = idx[0][np.argmax(np.abs(yf[idx]))]
            dominant_freq = xf[dominant_freq_idx]
            bpm = dominant_freq * 60
            cv2.putText(frame, f"BPM: {int(bpm)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "BPM: Deteksi Gagal", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # --- Tampilan ---
    cv2.imshow('Respirasi & rPPG Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Pembersihan ---
cap.release()
cv2.destroyAllWindows()
pose.close() # Penting untuk menutup objek MediaPipe
# if 'face_mesh' in locals(): face_mesh.close()