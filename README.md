# Flowsheet-classifier
This repo contains code to develop a classifier which detects whether a given image is Flowsheet or not a Flowsheet

# 📄 Flowsheet Classifier

A web application that classifies uploaded images as **Flowsheet** or **Not a Flowsheet** using a deep learning model built on top of the VGG16 architecture. This tool is useful for automating document organization or validation in process engineering contexts.

## 🚀 Features

- 🖼️ Upload an image (`.png`, `.jpg`, `.jpeg`)
- 🔍 Classifies whether the image is a **flowsheet** or **not a flowsheet**
- 📊 Displays prediction confidence and probability scores
- ⚙️ Powered by a pre-trained VGG16 model and TensorFlow
- 🖥️ Streamlit interface for easy deployment and use

## 🧠 Model Details

The classifier uses a transfer learning approach based on the VGG16 convolutional neural network architecture. The model was fine-tuned to detect engineering flowsheets.

Model input:
- Image resized to **224x224**
- Preprocessing aligned with `tf.keras.applications.vgg16.preprocess_input`

## 🛠️ Installation

### Clone the Repository
```bash
git clone https://github.com/PRP2003/Flowsheet-classifier.git
cd Flowsheet-classifier
```

### Run the app
```bash
streamlit run main.py
```

## 📁 Project Structure
```bash
├── main.py                # Streamlit frontend app
├── vgg16_model.ipynb      # Model training and evaluation notebook
├── trained_model             # Trained model file (placeholder path)
├── classtext_file     # Text file with class labels (placeholder path)
```

## 🧪 Example

Upload sample images:

    A typical chemical engineering flowsheet diagram

    A random photo or scanned document

The model will return:

    Predicted class (Flowsheet or Not a Flowsheet)

    Confidence score

    Class probability breakdown (expandable section)
