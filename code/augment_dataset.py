
import os
import random
from PIL import Image, ImageEnhance

# --- Konfigurasi ---
input_dataset_path = 'E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/data/aksarajawa-hanacaraka'
output_dataset_path = 'E:/BACKUP DRIVE D/Kerjaan/AI Engineering/akseoro/data/aksarajawa-hanacaraka_augmented'
num_augmentations_per_image = 5 # Jumlah versi augmentasi yang akan dibuat untuk setiap gambar asli

# --- Fungsi Augmentasi ---
def apply_augmentations(image):
    # Rotasi acak (sudut kecil)
    angle = random.uniform(-10, 10)
    image = image.rotate(angle, expand=False, fillcolor=(255, 255, 255)) # fill with white for background

    # Zoom acak
    zoom_factor = random.uniform(0.9, 1.1)
    width, height = image.size
    new_width, new_height = int(width * zoom_factor), int(height * zoom_factor)
    image = image.resize((new_width, new_height), Image.LANCZOS)
    # Crop atau pad kembali ke ukuran asli
    if zoom_factor > 1.0:
        left = (new_width - width) / 2
        top = (new_height - height) / 2
        right = (new_width + width) / 2
        bottom = (new_height + height) / 2
        image = image.crop((left, top, right, bottom))
    else:
        # Pad with white if zoomed out
        new_image = Image.new("RGB", (width, height), (255, 255, 255))
        new_image.paste(image, ((width - new_width) // 2, (height - new_height) // 2))
        image = new_image

    # Kecerahan acak
    brightness_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)

    # Kontras acak
    contrast_factor = random.uniform(0.8, 1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)

    # Pergeseran acak (translation)
    max_shift = 10 # Maksimal 10 piksel pergeseran
    x_shift = random.randint(-max_shift, max_shift)
    y_shift = random.randint(-max_shift, max_shift)
    image = image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift), fillcolor=(255, 255, 255))

    return image

# --- Proses Augmentasi Dataset ---
print(f"Memulai augmentasi dataset dari: {input_dataset_path}")
print(f"Hasil augmentasi akan disimpan di: {output_dataset_path}")

# Buat direktori output jika belum ada
os.makedirs(output_dataset_path, exist_ok=True)

# Iterasi melalui setiap subdirektori (kelas) di dataset input
for class_name in os.listdir(input_dataset_path):
    class_input_path = os.path.join(input_dataset_path, class_name)
    class_output_path = os.path.join(output_dataset_path, class_name)

    # Pastikan itu adalah direktori kelas
    if os.path.isdir(class_input_path):
        os.makedirs(class_output_path, exist_ok=True)
        print(f"Memproses kelas: {class_name}")

        # Iterasi melalui setiap gambar di direktori kelas
        for image_name in os.listdir(class_input_path):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                original_image_path = os.path.join(class_input_path, image_name)
                try:
                    original_image = Image.open(original_image_path).convert("RGB")

                    # Simpan gambar asli ke direktori output augmented
                    original_image.save(os.path.join(class_output_path, f"original_{image_name}"))

                    # Buat dan simpan versi augmentasi
                    for i in range(num_augmentations_per_image):
                        augmented_image = apply_augmentations(original_image)
                        augmented_image_name = f"aug_{i}_{image_name}"
                        augmented_image.save(os.path.join(class_output_path, augmented_image_name))
                except Exception as e:
                    print(f"Gagal memproses gambar {original_image_path}: {e}")

print("\nProses augmentasi dataset selesai.")
