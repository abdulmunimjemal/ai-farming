import streamlit as st
from utils import load_model, predict_fertilizer, predict_crop, predict_crop_disease
import torch

# Streamlit app configuration
st.set_page_config(page_title="Agricultural AI Assistant", layout="wide", initial_sidebar_state="expanded")

# Title and description
st.title("🌾 Agricultural AI Assistant 🌿")
st.markdown("""
Welcome to the Agricultural AI Assistant. This application allows you to:
- 🌱 **Predict the type of fertilizer** required based on soil composition.
- 🚜 **Recommend the best crop** to plant based on soil and weather conditions.
- 🍃 **Diagnose crop diseases** from images.

This is the final project for the Fundamentals of AI Course at Addis Ababa Institute of Technology (AAiT).
""")

@st.cache_resource
def cached_load_model(model_path):
    return load_model(model_path)

# Sidebar for model loading
st.sidebar.header("📂 Load Models")
fertilizer_model_path = "models/fertilizer-recommendation-model.joblib"
crop_model_path = "models/crop-recommendation-model.joblib"
disease_model_path = "models/crop_disease_model.pth"

# Load models
st.sidebar.write("Loading models...")
fertilizer_model = cached_load_model(fertilizer_model_path)
crop_model = cached_load_model(crop_model_path)
disease_model = cached_load_model(disease_model_path)
st.sidebar.success("Models loaded successfully!")

# clear the sidebar
st.sidebar.empty()
# Sidebar content after models are loaded
st.sidebar.header("📊 Model Information")
st.sidebar.markdown("""
- Fertilizer model: Loaded
- Crop recommendation model: Loaded
- Disease classification model: Loaded
""")


st.sidebar.header("👥 Group Members")
st.sidebar.markdown("""
- Abdulmunim Jundurahman
- Fethiya Safi
- Fuad Mohammed
- Salman Ali
- Obsu Kebede
""")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["🧪 Fertilizer Prediction", "🌾 Crop Recommendation", "🔍 Crop Disease Diagnosis"])

with tab1:
    st.header("🧪 Fertilizer Prediction")
    st.markdown("Input the soil composition to predict the best fertilizer.")

    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.number_input("Nitrogen content", min_value=0, max_value=100, value=0)
    with col2:
        phosphorous = st.number_input("Phosphorous content", min_value=0, max_value=100, value=0)
    with col3:
        potassium = st.number_input("Potassium content", min_value=0, max_value=100, value=0)

    if st.button("Predict Fertilizer"):
        fertilizer, confidence = predict_fertilizer(nitrogen, phosphorous, potassium, fertilizer_model)
        st.success(f"Recommended Fertilizer: **{fertilizer}** (Confidence: **{confidence:.2f}%**)")

with tab2:
    st.header("🌾 Crop Recommendation")
    st.markdown("Input soil and weather conditions to get crop recommendations.")

    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen Ratio", min_value=0, max_value=100, value=0)
        temperature = st.number_input("Average Temperature (°C)", min_value=-30.0, max_value=50.0, value=0.0)
        ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0)
    with col2:
        P = st.number_input("Phosphorus Ratio", min_value=0, max_value=100, value=0)
        humidity = st.number_input("Relative Humidity (%)", min_value=0.0, max_value=100.0, value=0.0)
        rainfall = st.number_input("Yearly Rainfall (mm)", min_value=0.0, max_value=10000.0, value=0.0)
    with col3:
        K = st.number_input("Potassium Ratio", min_value=0, max_value=100, value=0)

    if st.button("Recommend Crop"):
        crop, confidence = predict_crop(N, P, K, temperature, humidity, ph, rainfall, crop_model)
        st.success(f"Recommended Crop: **{crop}** (Confidence: **{confidence:.2f}%**)")

with tab3:
    st.header("🔍 Crop Disease Diagnosis")
    st.markdown("Upload an image of the crop leaf to diagnose the disease.")

    uploaded_file = st.file_uploader("Upload an image of the crop leaf", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        if st.button("Diagnose Disease"):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            disease, confidence = predict_crop_disease(uploaded_file, disease_model, device)
            st.success(f"Predicted Disease: **{disease}** (Confidence: **{confidence:.2f}%**)")

# Footer
st.markdown("""
---
Created with ❤️ at [AAiT](https://www.aait.edu.et/)

Special Thanks to the Knowledge and Support of our Instructor [Amanuel Mersha](https://www.linkedin.com/in/leobitz/) and the Kaggle OpenSource Community.
""")