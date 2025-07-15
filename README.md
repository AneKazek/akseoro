# Akseoro : Aksara Jawir Classifier

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-0.0.0-yellow?style=for-the-badge&logo=tensorflow&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-9.x-green?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.x-blueviolet?style=for-the-badge&logo=numpy&logoColor=white)

## Project Description

This project aims to develop a Javanese script classification system using machine learning techniques. The system is designed to identify and classify various Javanese script characters from images, achieving an impressive classification accuracy of 99.2%. This can be used for digitizing ancient manuscripts or educational applications.

## Features

-   **Javanese Script Classification**: Accurately identifies Javanese script characters.
-   **Data Augmentation**: Increases dataset size to train more robust models.
-   **Exportable Models**: Trained models can be exported for inference on various devices.
-   **Image Validation**: Tools for validating image datasets.

## Technologies Used

This project leverages the following key technologies:

*   **Python**: The primary programming language for all scripts.
*   **TensorFlow** (with `tf.keras`): The main framework for building, training, and evaluating deep learning models.
*   **PIL (Pillow)**: A Python Imaging Library fork used for image data augmentation (rotation, zoom, brightness, contrast, shifting) in the `augment_dataset.py` script.
*   **NumPy**: The fundamental library for numerical computing, especially for manipulating image data as arrays.
*   **TensorFlow Lite**: Used to convert trained models into a lightweight format (`.tflite`) for inference on resource-constrained devices.

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/AneKazek/akseoro.git
    cd aksara-jawa-classifier
    ```

2.  **Create and activate a virtual environment (optional, but recommended):**

    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Dataset

Ensure you have an organized Javanese script dataset. The expected directory structure is:

```
data/
└── aksarajawa-hanacaraka/
    ├── ba/
    │   └── image_001.png
    ├── ca/
    │   └── image_002.png
    └── ...
```

### 2. Data Augmentation (Optional)

To expand the dataset and improve model performance:

```bash
python code/augment_dataset.py
```

Augmented data will be saved in `data/aksarajawa-hanacaraka_augmented/`.

### 3. Train Classification Model

To train a new model:

```bash
python code/train_classifier.py
```

The trained model will be saved in the `exported_model/` directory.

### 4. Make Predictions

To make predictions on new images:

```bash
python code/predict.py --image_path "path/to/your/image.png"
```

### 5. Validate Images

To validate images in the dataset:

```bash
python code/validate_images.py
```

## Project Structure

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
├── aksarajawa-hanacaraka/          # Original dataset
└── aksarajawa-hanacaraka_augmented/  # Augmented dataset
exported_model/
├── labels.txt
├── model.tflite
└── saved_model/                    # Exported TensorFlow model
models/                             # Directory for models saved during training
reports/
└── figures/                        # Directory for reports and figures
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

## Contributing

Contributions are highly appreciated! If you'd like to contribute, please follow these steps:

1.  Fork this repository.
2.  Create a new branch (`git checkout -b feature/new-feature-name`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to your branch (`git push origin feature/new-feature-name`).
6.  Open a Pull Request.

## License

This project is licensed under the [LICENCE.txt](LICENCE.txt).

## Contact

If you have any questions or suggestions, feel free to contact me.
