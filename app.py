# app.py
import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops

# ---------------------------
# Load saved models
# ---------------------------
rf_crop = joblib.load("/Users/pragadeeswarrs/Desktop/AgroAid/rf_crop.pkl")
rf_disease_dict = joblib.load("/Users/pragadeeswarrs/Desktop/AgroAid/rf_disease_dict.pkl")

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AgroAid Crop Disease Detector", layout="centered")
st.title("🌱 AgroAid Crop Disease Detector")
st.write("Upload a crop image, and the AI will detect the crop type and its disease.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is not None:
        # Display uploaded image
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)
        
        # ---------------------------
        # Preprocess image
        # ---------------------------
        img_resized = cv2.resize(img, (128, 128)) / 255.0

        # Color features
        mean_r, mean_g, mean_b = np.mean(img_resized[:,:,0]), np.mean(img_resized[:,:,1]), np.mean(img_resized[:,:,2])
        std_r, std_g, std_b = np.std(img_resized[:,:,0]), np.std(img_resized[:,:,1]), np.std(img_resized[:,:,2])
        color_features = [mean_r, mean_g, mean_b, std_r, std_g, std_b]

        # Texture features (GLCM)
        gray = (img_resized*255).astype(np.uint8)
        gray = np.mean(gray, axis=2).astype(np.uint8)
        glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm,'contrast')[0,0]
        homogeneity = graycoprops(glcm,'homogeneity')[0,0]
        texture_features = [contrast, homogeneity]

        features = np.array(color_features + texture_features).reshape(1, -1)

        # ---------------------------
        # Predict crop
        # ---------------------------
        crop_pred = rf_crop.predict(features)[0]

        # ---------------------------
        # Predict disease
        # ---------------------------
        if crop_pred in rf_disease_dict:
            disease_pred = rf_disease_dict[crop_pred].predict(features)[0]
        else:
            disease_pred = "Unknown"

        # ---------------------------
        # Display results neatly
        # ---------------------------
        st.markdown("### Prediction Results")
        st.success(f"**Crop:** {crop_pred}")
        st.warning(f"**Disease:** {disease_pred}")

    else:
        st.error("⚠️ Could not read the uploaded image. Please upload a valid image file.")
