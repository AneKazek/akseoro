import tensorflow as tf
import numpy as np
import os

# --- Konfigurasi ---
dataset_path = 'E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/data/aksarajawa-hanacaraka'
img_height = 224
img_width = 224
batch_size = 32
epochs = 15
export_dir = 'E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/exported_model'

# --- 1. Memuat dan Mempersiapkan Dataset ---
print(f"Memuat dataset dari: {dataset_path}")

# Buat dataset pelatihan (80% dari data)
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Buat dataset validasi (20% dari data)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Dapatkan nama kelas (label) dari dataset
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Ditemukan {num_classes} kelas: {class_names}")

# --- 2. Augmentasi Data ---
# Membuat lapisan augmentasi untuk meningkatkan variasi data latih
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Terapkan augmentasi hanya pada dataset pelatihan
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# Konfigurasi dataset untuk performa
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Membangun Model (Transfer Learning) ---
print("\nMembangun model menggunakan MobileNetV2...")

# Muat model dasar MobileNetV2 yang sudah dilatih di ImageNet, tanpa lapisan atas
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Bekukan bobot model dasar agar tidak ikut terlatih
base_model.trainable = False

# Buat model baru di atas model dasar
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
# Normalisasi input piksel ke rentang [-1, 1] yang diharapkan oleh MobileNetV2
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# --- 4. Kompilasi dan Pelatihan Model ---
print("Memulai kompilasi dan pelatihan model...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

print("\nPelatihan selesai.")

# --- 5. Konversi ke TensorFlow Lite dan Kuantisasi ---
print(f"\nMengekspor model ke format TFLite di direktori: {export_dir}")

# Buat direktori jika belum ada
if not os.path.exists(export_dir):
    os.makedirs(export_dir)

# Ekspor model ke format SavedModel
keras_model_path = os.path.join(export_dir, "saved_model")
model.export(keras_model_path)

# Konversi model Keras ke TFLite dengan kuantisasi
converter = tf.lite.TFLiteConverter.from_saved_model(keras_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()

# Simpan model TFLite yang sudah dikuantisasi
tflite_model_path = os.path.join(export_dir, 'model.tflite')
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

# --- 6. Simpan Label ---
labels_path = os.path.join(export_dir, 'labels.txt')
with open(labels_path, 'w') as f:
    f.write('\n'.join(class_names))

print("\nProses selesai. Model `model.tflite` dan `labels.txt` telah berhasil dibuat.")