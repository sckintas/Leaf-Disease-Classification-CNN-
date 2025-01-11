# Leaf Disease Detection

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## Introduction
Leaf Disease Detection is a machine learning project aimed at identifying diseases in plant leaves using computer vision and deep learning. This tool can assist farmers and agronomists in monitoring crop health and taking early action to mitigate potential damage.

---

## Features
- Automatically detects and classifies leaf diseases.
- Supports multiple plant species and disease types.
- High accuracy achieved using Convolutional Neural Networks (CNN).
- Web-based interface for easy interaction.
- Real-time predictions with user-uploaded images.

---

## Technologies Used
- **Programming Language:** Python
- **Web Framework:** Flask
- **Deep Learning Framework:** TensorFlow/Keras
- **Image Processing:** Pillow 
- **Frontend:** HTML
- **Visualization:** Matplotlib
- **Version Control:** Git

---

## Dataset
The dataset consists of high-resolution images of healthy and diseased leaves. Images are preprocessed, augmented, and divided into training, validation, and testing datasets.

- **Source:** Kaggle or other agriculture-specific datasets.
- **Categories:** Includes various diseases such as:
  - Powdery Mildew
  - Rust
  - Leaf Spot
  - Healthy Leaves

---

## Model Architecture
The model is built using a Convolutional Neural Network (CNN) with the following layers:
- **Input Layer:** Processes input images (resized to uniform dimensions of 128x128).
- **Convolutional Layers:** Extract features using multiple kernels.
- **Pooling Layers:** Reduce dimensionality.
- **Dense Layers:** Classify the extracted features.
- **Output Layer:** Produces probabilities for each class.

Additional techniques:
- **Data Augmentation:** Enhances dataset diversity.
- **Dropout:** Prevents overfitting.
- **Adam Optimizer:** Used for training.

---

## Installation
Follow these steps to set up the project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/sckintas/Leaf-Disease-Classification-CNN-.git
   cd Leaf-Disease-Classification-CNN-
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset and place it in the appropriate directory (e.g., `data/`).

5. Ensure the trained model file `leaf_model.h5` is located in the `app` directory.

---

## Usage
1. Start the Flask web application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload an image of a leaf using the web interface.

4. View the predicted disease and the uploaded image on the results page.

---

## Results
The model achieved the following performance metrics on the test dataset:

- **Training Accuracy:** 98.5%
- **Validation Accuracy:** 98.4%
- **Test Accuracy:** 92.04%
- **Test Loss:** 0.3245

### Detailed Classification Report:

| Class                                | Precision | Recall | F1-Score | Support |
|--------------------------------------|-----------|--------|----------|---------|
| Apple___Apple_scab                   | 0.99      | 0.95   | 0.97     | 504     |
| Apple___Black_rot                    | 0.89      | 0.99   | 0.94     | 497     |
| Apple___Cedar_apple_rust             | 1.00      | 0.96   | 0.98     | 440     |
| Apple___healthy                      | 0.93      | 0.93   | 0.93     | 502     |
| Blueberry___healthy                  | 0.99      | 0.93   | 0.96     | 454     |
| Cherry_(including_sour)__Powdery_mildew | 1.00   | 0.96   | 0.98     | 421     |
| Cherry_(including_sour)__healthy     | 0.84      | 0.99   | 0.91     | 456     |
| Corn_(maize)__Cercospora_leaf_spot   | 0.92      | 0.93   | 0.93     | 410     |
| Corn_(maize)__Common_rust            | 1.00      | 1.00   | 1.00     | 477     |
| Corn_(maize)__Northern_Leaf_Blight   | 0.96      | 0.95   | 0.95     | 477     |
| ... (truncated for brevity)          |           |        |          |         |

- **Overall Accuracy:** 92.04%
- **Macro Average:** Precision: 0.94, Recall: 0.92, F1-Score: 0.92
- **Weighted Average:** Precision: 0.94, Recall: 0.92, F1-Score: 0.92

---

## Contributing
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add a new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Feel free to reach out with questions or suggestions!

