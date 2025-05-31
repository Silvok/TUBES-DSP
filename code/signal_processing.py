import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt

def clean_signal(sig):
    """
    Membersihkan sinyal input dengan normalisasi, median filter, dan exponential smoothing.

    Parameters:
        sig (np.ndarray): Sinyal input 1D.

    Returns:
        np.ndarray: Sinyal yang telah dibersihkan.
    """
    # Normalisasi sinyal: mengurangi mean dan membagi dengan standar deviasi (agar rata-rata 0 dan deviasi 1)
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
    # Mengurangi noise impulsif dengan median filter (kernel size 5)
    sig = medfilt(sig, kernel_size=5)
    # Menghaluskan sinyal dengan exponential smoothing
    sig = exponential_smoothing(sig)
    return sig

def exponential_smoothing(signal, alpha=0.3):
    """
    Menghaluskan sinyal menggunakan metode exponential smoothing.

    Parameters:
        signal (np.ndarray): Sinyal input 1D.
        alpha (float): Koefisien smoothing (0 < alpha < 1).

    Returns:
        np.ndarray: Sinyal yang telah dihaluskan.
    """
    # Membuat array hasil dengan ukuran sama seperti sinyal input
    result = np.zeros_like(signal)
    # Inisialisasi nilai pertama hasil dengan nilai pertama sinyal
    result[0] = signal[0]
    # Melakukan smoothing untuk setiap titik data
    for t in range(1, len(signal)):
        # Rumus exponential smoothing
        result[t] = alpha * signal[t] + (1 - alpha) * result[t - 1]
    return result

def bandpass(signal, lowcut, highcut, fs, order=4):
    """
    Menerapkan filter bandpass Butterworth pada sinyal.

    Parameters:
        signal (np.ndarray): Sinyal input 1D.
        lowcut (float): Frekuensi cut-off bawah (Hz).
        highcut (float): Frekuensi cut-off atas (Hz).
        fs (float): Frekuensi sampling (Hz).
        order (int): Orde filter.

    Returns:
        np.ndarray: Sinyal hasil filter bandpass.
    """
    # Menghitung frekuensi Nyquist
    nyq = 0.5 * fs
    # Normalisasi frekuensi cut-off
    low = lowcut / nyq
    high = highcut / nyq
    # Membuat koefisien filter Butterworth
    b, a = butter(order, [low, high], btype='band')
    # Menerapkan filter secara forward-backward (zero-phase)
    return filtfilt(b, a, signal)

def estimate_bpm_peaks(signal, fs, min_bpm=8, max_bpm=30):
    """
    Mengestimasi BPM (beats per minute) dari sinyal dengan mendeteksi puncak (peaks).

    Parameters:
        signal (np.ndarray): Sinyal input 1D.
        fs (float): Frekuensi sampling (Hz).
        min_bpm (float): BPM minimum yang diharapkan.
        max_bpm (float): BPM maksimum yang diharapkan.

    Returns:
        float: Nilai BPM hasil estimasi, atau 0.0 jika tidak valid.
    """
    # Filter sinyal pada rentang frekuensi yang sesuai dengan rentang BPM
    signal = bandpass(signal, min_bpm / 60, max_bpm / 60, fs)
    # Hitung jarak minimum antar puncak berdasarkan BPM maksimum
    min_dist = int(fs * 60 / max_bpm)
    # Deteksi puncak pada sinyal
    peaks, _ = find_peaks(signal, distance=min_dist)
    # Jika jumlah puncak cukup untuk estimasi BPM
    if len(peaks) > 3:
        # Hitung interval antar puncak (dalam detik)
        intervals = np.diff(peaks) / fs
        # Hitung rata-rata periode antar puncak
        mean_period = np.mean(intervals)
        # Konversi periode rata-rata ke BPM
        bpm = 60.0 / mean_period
        # Pastikan BPM dalam rentang yang diharapkan
        if min_bpm <= bpm <= max_bpm:
            return bpm
    # Jika tidak valid, kembalikan 0.0
    return 0.0