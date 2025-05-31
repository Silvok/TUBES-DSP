from PyQt5.QtWidgets import QApplication
import sys
from code.main_gui import SignalMonitor

def main():
    """
    Fungsi utama untuk menjalankan aplikasi GUI pemantauan sinyal.

    Membuat instance QApplication, menampilkan jendela utama (SignalMonitor),
    dan memulai event loop aplikasi.

    Parameters:
        Tidak ada.

    Returns:
        Tidak ada. Fungsi ini akan keluar ketika aplikasi ditutup.
    """
    # Membuat objek aplikasi Qt
    app = QApplication(sys.argv)
    # Membuat dan menampilkan jendela utama aplikasi
    window = SignalMonitor()
    window.show()
    # Memulai event loop aplikasi dan keluar saat aplikasi ditutup
    sys.exit(app.exec_())

if __name__ == "__main__":
    # Mengeksekusi fungsi main() jika file dijalankan secara langsung
    main()