

import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
import os
import pandas as pd # Import pandas for displaying probabilities

# --- Feature Extraction Function (must be identical to training) ---
def extract_features(img_array):
    if img_array is None:
        return None

    # Resize the image
    img_resized = cv2.resize(img_array, (100, 100))

    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values
    gray_normalized = gray / 255.0

    # Flatten the image into a 1D feature vector
    return gray_normalized.flatten()

# --- Load Model and LabelEncoder ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('cancer_detection_model.joblib')
        label_encoder = joblib.load('label_encoder.joblib')
        return model, label_encoder
    except FileNotFoundError:
        st.error("Error: Model or LabelEncoder files not found. Please ensure 'cancer_detection_model.joblib' and 'label_encoder.joblib' are in the same directory as this app.py file.")
        st.stop()

model, label_encoder = load_resources()

# --- Streamlit App Interface ---
st.title("Cancer Detection System")
st.write("Upload an image to predict if it's related to Blood, Skin, or Lung Cancer.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Convert PIL Image to OpenCV format
    img_array = np.array(image)
    img_array_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Extract features
    features = extract_features(img_array_cv)

    if features is not None:
        # Reshape for prediction (model expects 2D array)
        features = features.reshape(1, -1)

        # Make prediction
        prediction_numeric = model.predict(features)

        # Inverse transform to get the original label string
        predicted_label = label_encoder.inverse_transform(prediction_numeric)[0]

        display_label = "Unknown Cancer Type Detected"
        if predicted_label.startswith('blood'):
            display_label = "Blood Cancer Detected"
        elif predicted_label.startswith('skin'):
            display_label = "Skin Cancer Detected"
        elif predicted_label.startswith('lung'):
            display_label = "Lung Cancer Detected"
        
        st.success(f"Prediction: {display_label}")

        # Optional: Display probability scores if your model supports it (e.g., RandomForestClassifier.predict_proba)
        try:
            probabilities = model.predict_proba(features)[0]
            probability_df = pd.DataFrame({
                'Class': label_encoder.classes_,
                'Probability': probabilities
            }).sort_values(by='Probability', ascending=False)
            st.write("### Prediction Probabilities:")
            st.dataframe(probability_df)
        except AttributeError:
            st.info("Model does not support `predict_proba` for probability scores.")

    else:
        st.error("Could not process the uploaded image. Please try another one.")
