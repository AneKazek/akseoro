
import os
from PIL import Image

# --- Konfigurasi ---
dataset_path = 'E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/data/aksarajawa-hanacaraka'
corrupted_files = []

print(f"Memulai validasi gambar di: {dataset_path}\n")

# Iterasi melalui setiap subdirektori (setiap kelas aksara)
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        # Iterasi melalui setiap file di dalam subdirektori
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Coba buka file gambar
                with Image.open(file_path) as img:
                    # Verifikasi bahwa file bisa dimuat
                    img.verify()
            except Exception as e:
                # Jika terjadi error, file tersebut kemungkinan rusak
                print(f"Ditemukan file rusak: {file_path} -> Error: {e}")
                corrupted_files.append(file_path)

print("\nValidasi selesai.")

if not corrupted_files:
    print("Tidak ada file gambar yang rusak ditemukan.")
else:
    print(f"\nTotal file rusak ditemukan: {len(corrupted_files)}")
    print("Silakan hapus file-file di atas sebelum melanjutkan pelatihan.")

