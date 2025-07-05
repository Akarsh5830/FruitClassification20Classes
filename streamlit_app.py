import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Set the title of the app
st.title("🍎 Fruit Classifier using MobileNetV2")
st.write("Upload an image of a fruit and the model will predict its type.")

# Load the pre-trained model
model = load_model("fruit_classifier_mobilenetv2.h5")  # Make sure the file name matches

# Define the class labels (in exact order of training, cleaned)
class_labels = [
    "Apple Golden", "Apple Red", "Banana", "Cherry", "Grape Blue",
    "Guava", "Kiwi", "Lemon", "Mango", "Orange",
    "Papaya", "Peach", "Pear", "Pineapple", "Plum",
    "Pomegranate", "Raspberry", "Strawberry", "Tomato", "Watermelon"
]

# File uploader for image input
uploaded_file = st.file_uploader("📷 Upload a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize (if used in training)

    # Predict the class
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display the prediction
    st.markdown(f"### 🧠 Predicted Fruit: **{predicted_class}**")
    st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")
