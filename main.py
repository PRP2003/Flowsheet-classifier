# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuration
st.set_page_config(page_title="Flowsheet Classifier", layout="centered")

MODEL_PATH = "model_path"
CLASS_NAMES_PATH = "classtext_filepath"
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Model and Class Names Loading 
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model_and_classes():
    #Load the trained Keras model and the class names
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model or class names: {e}")
        return None, None

model, class_names = load_model_and_classes()

# Image Preprocessing
def preprocess_image(image_pil):
    
    img = image_pil.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = tf.keras.utils.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.vgg16.preprocess_input(img_array_expanded)

# Streamlit App UI 
st.title("📄 Flowsheet Classifier ")

st.markdown("""
Upload an image and the model will predict whether it is a **flowsheet** or **not a flowsheet**.
This model is based on a pre-trained VGG16 network.
""")

uploaded_file = st.file_uploader(
    "Choose an image...", type=["png", "jpg", "jpeg"]
)

if model is None:
    st.warning("Model could not be loaded. Please check the file paths and try again.")
elif uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Add a button to trigger prediction
    if st.button("Classify Image"):
        with st.spinner("Analyzing the image..."):
            # Preprocess the image and make a prediction
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
            
            # Interpret the results
            probabilities = tf.nn.softmax(predictions[0])
            score = tf.reduce_max(probabilities)
            predicted_class_index = tf.argmax(probabilities).numpy()
            predicted_class_name = class_names[predicted_class_index]

        # Display the result
        st.subheader("Prediction Result")
        if predicted_class_name.lower() == "flowsheet":
            st.success(f"This looks like a **Flowsheet**!")
        else:
            st.info(f"This looks like it is **Not a Flowsheet**.")
        
        st.write(f"**Confidence:** {score:.2%}")

        # Optional: Display detailed probabilities
        with st.expander("View Detailed Probabilities"):
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name.capitalize()}: {probabilities[i]:.2%}")