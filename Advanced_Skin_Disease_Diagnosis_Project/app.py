import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
import numpy as np
import pickle
import os

# Paths for loading saved components
MODEL_PATH = "C:/Users/Abhijeet/Downloads/ASDP/mobilenet_model.h5"
SVM_PATH = "C:/Users/Abhijeet/Downloads/ASDP/svm_classifier.pkl"
SCALER_PATH = "C:/Users/Abhijeet/Downloads/ASDP/scaler.pkl"


class_names = [
    "AthleteFoot",
    "Cellulitis",
    "Chickenpox",
    "Cutaneous Larva Migrans",
    "Impetigo",
    "Nail Fungus",
    "Ringworm",
    "Shingles"
]


@st.cache_resource
def load_components():
    # Load the model and components
    base_model = tf.keras.models.load_model(MODEL_PATH)
    with open(SVM_PATH, "rb") as f:
        svm_classifier = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return base_model, svm_classifier, scaler

base_model, svm_classifier, scaler = load_components()

# Helper function to preprocess image
def preprocess_image(img_path, image_size=(224, 224)):
    img = image.load_img(img_path, target_size=image_size)
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function
def predict_image(img_path):
    img_array = preprocess_image(img_path)

    # Extract features using MobileNet
    features = base_model.predict(img_array)
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict with SVM
    pred = svm_classifier.predict(features_scaled)  # Get the predicted class index
    pred_prob = svm_classifier.predict_proba(features_scaled)  # Get prediction probabilities

    # Convert the predicted class index to the corresponding class name (disease name)
    pred_class_index = pred[0]  # Get the predicted class index
    pred_class = class_names[pred_class_index]  # Map index to class name (disease name)
    
    # Create a dictionary of prediction probabilities for each class
    pred_prob_dict = {class_names[i]: prob for i, prob in enumerate(pred_prob[0])}

    return pred_class, pred_prob_dict

# Streamlit App
st.title("Skin Disease Diagnosis")
st.write("Upload an image to classify the skin disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)


    # Save the uploaded file temporarily
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the class and probabilities
    predicted_class, predicted_probabilities = predict_image(temp_file_path)

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class:** {predicted_class}")  # Display the predicted class name
    st.write("**Prediction Probabilities:**")
    
    # Display prediction probabilities as a bar chart
    st.bar_chart(predicted_probabilities)

    # Clean up temporary file
    os.remove(temp_file_path)
