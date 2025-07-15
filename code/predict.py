
import numpy as np
from PIL import Image
import tensorflow.lite as tflite
import os

# --- Konfigurasi ---
# Path ke model dan label yang telah diekspor
model_path = "E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/exported_model/model.tflite"
labels_path = "E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/exported_model/labels.txt"

# Path ke gambar sampel untuk diuji
# Ganti dengan path gambar yang ingin Anda klasifikasikan
sample_image_path = "E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/data/aksarajawa-hanacaraka/ha/4693335674404368252_base64_5.png"

# --- Fungsi Utama ---
def classify_image(model_path, labels_path, image_path):
    """
    Memuat model TFLite, melakukan pra-pemrosesan pada gambar,
    dan mengembalikan prediksi teratas beserta skor keyakinannya.
    """
    # 1. Muat label dari file
    try:
        with open(labels_path, 'r') as f:
            labels = f.read().splitlines()
    except FileNotFoundError:
        print(f"Error: File label tidak ditemukan di {labels_path}")
        return

    # 2. Muat model TFLite dan alokasikan tensor
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
    except ValueError:
        print(f"Error: File model tidak ditemukan atau rusak di {model_path}")
        return

    # 3. Dapatkan detail input dan output dari model
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # 4. Dapatkan tinggi dan lebar input yang diharapkan oleh model
    _, height, width, _ = input_details['shape']

    # 5. Muat dan pra-proses gambar sampel
    try:
        image = Image.open(image_path).convert('RGB').resize((width, height))
    except FileNotFoundError:
        print(f"Error: File gambar tidak ditemukan di {image_path}")
        return

    # 6. Konversi gambar ke numpy array, normalisasi, dan siapkan untuk model
    # Model MobileNetV2 mengharapkan nilai piksel dalam rentang [-1, 1].
    input_data = np.array(image, dtype=np.float32)
    input_data = (input_data / 127.5) - 1.0
    input_data = np.expand_dims(input_data, axis=0)

    # 7. Atur tensor input dan jalankan inferensi
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # 8. Dapatkan hasil output
    output_data = interpreter.get_tensor(output_details['index'])
    scores = np.squeeze(output_data)

    # 9. Lakukan de-kuantisasi pada skor jika outputnya adalah int8
    if output_details['dtype'] == np.uint8:
        output_scale, output_zero_point = output_details['quantization']
        scores = (scores.astype(np.float32) - output_zero_point) * output_scale

    # 10. Dapatkan prediksi teratas
    top_prediction_index = np.argmax(scores)
    predicted_label = labels[top_prediction_index]
    confidence = scores[top_prediction_index]

    print(f"Gambar dianalisis: {os.path.basename(image_path)}")
    print(f"Prediksi Aksara: '{predicted_label}'")
    print(f"Skor Keyakinan: {confidence:.2%}")

# --- Jalankan Skrip ---
if __name__ == "__main__":
    if not os.path.exists(model_path):
        print("Model (model.tflite) tidak ditemukan.")
        print("Harap jalankan skrip 'train_classifier.py' terlebih dahulu untuk melatih dan mengekspor model.")
    else:
        classify_image(model_path, labels_path, sample_image_path)
