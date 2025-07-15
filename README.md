# Aksara Jawa Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi aksara Jawa menggunakan teknik pembelajaran mesin. Sistem ini dirancang untuk mengidentifikasi dan mengklasifikasikan berbagai karakter aksara Jawa dari gambar, yang dapat digunakan untuk digitalisasi naskah kuno atau aplikasi pendidikan.

## Fitur

- **Klasifikasi Aksara Jawa**: Mengidentifikasi karakter aksara Jawa dengan akurasi tinggi.
- **Augmentasi Data**: Meningkatkan ukuran dataset untuk melatih model yang lebih robust.
- **Model yang Dapat Diekspor**: Model yang terlatih dapat diekspor untuk inferensi pada perangkat yang berbeda.
- **Validasi Gambar**: Alat untuk memvalidasi dataset gambar.

## Instalasi

Untuk menjalankan proyek ini secara lokal, ikuti langkah-langkah berikut:

1.  **Kloning repositori:**

    ```bash
    git clone https://github.com/your-username/aksara-jawa-classifier.git
    cd aksara-jawa-classifier
    ```

2.  **Buat dan aktifkan virtual environment (opsional, tapi direkomendasikan):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Instal dependensi:**

    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan

### 1. Menyiapkan Dataset

Pastikan Anda memiliki dataset aksara Jawa yang terorganisir. Struktur direktori yang diharapkan adalah:

```
data/
└── aksarajawa-hanacaraka/
    ├── ba/
    │   └── image_001.png
    ├── ca/
    │   └── image_002.png
    └── ...
```

### 2. Augmentasi Data (Opsional)

Untuk memperbanyak dataset dan meningkatkan performa model:

```bash
python code/augment_dataset.py
```

Hasil augmentasi akan disimpan di `data/aksarajawa-hanacaraka_augmented/`.

### 3. Melatih Model Klasifikasi

Untuk melatih model baru:

```bash
python code/train_classifier.py
```

Model yang terlatih akan disimpan di direktori `exported_model/`.

### 4. Melakukan Prediksi

Untuk melakukan prediksi pada gambar baru:

```bash
python code/predict.py --image_path "path/to/your/image.png"
```

### 5. Memvalidasi Gambar

Untuk memvalidasi gambar dalam dataset:

```bash
python code/validate_images.py
```

## Struktur Proyek

```
.gitignore
LICENCE.txt
README.md
requirements.txt
code/
├── augment_dataset.py
├── predict.py
├── train_classifier.py
└── validate_images.py
data/
├── aksarajawa-hanacaraka/          # Dataset asli
└── aksarajawa-hanacaraka_augmented/  # Dataset yang diaugmentasi
exported_model/
├── labels.txt
├── model.tflite
└── saved_model/                    # Model TensorFlow yang diekspor
models/                             # Direktori untuk model yang disimpan selama pelatihan
reports/
└── figures/                        # Direktori untuk laporan dan gambar
src/
├── data/
│   └── make_dataset.py
├── features/
│   └── build_features.py
├── models/
│   ├── predict_model.py
│   └── train_model.py
└── visualization/
    └── visualize.py
```

## Kontribusi

Kontribusi sangat dihargai! Jika Anda ingin berkontribusi, silakan ikuti langkah-langkah berikut:

1.  Fork repositori ini.
2.  Buat branch baru (`git checkout -b feature/nama-fitur-baru`).
3.  Lakukan perubahan Anda.
4.  Commit perubahan Anda (`git commit -m 'Tambahkan fitur baru'`).
5.  Push ke branch Anda (`git push origin feature/nama-fitur-baru`).
6.  Buka Pull Request.

## Lisensi

Proyek ini dilisensikan di bawah [LICENCE.txt](LICENCE.txt).

## Kontak

Jika Anda memiliki pertanyaan atau saran, jangan ragu untuk menghubungi saya.