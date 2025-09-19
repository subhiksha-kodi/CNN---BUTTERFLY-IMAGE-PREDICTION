import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ------------------------
# Custom Page Styling
# ------------------------
st.set_page_config(page_title="Butterfly Classifier", page_icon="🦋", layout="centered")

# Inject CSS for modern glass style
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #1f1c2c, #928DAB);
    background-size: cover;
    color: white;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

.pred-card, .upload-card {
    padding: 25px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.4);
    margin-top: 20px;
    text-align: center;
}

.pred-card h3 {
    color: #8be9fd;
    font-size: 24px;
}

.pred-card p {
    font-size: 18px;
    color: #f1f1f1;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ------------------------
# Load model and labels
# ------------------------
MODEL_PATH = "butterfly_classifier.h5"
LABELS_PATH = "class_labels.json"
IMG_SIZE = (128, 128)  # same size used in training

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

class_labels = {int(k): v for k, v in class_labels.items()}

# ------------------------
# Streamlit App
# ------------------------
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.title("🦋 Butterfly Species Classifier")
st.write("Upload a butterfly image to predict its category.")
uploaded_file = st.file_uploader("Choose a butterfly image...", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Show uploaded image inside styled card
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Preprocess
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction)
    pred_label = class_labels[pred_index]
    confidence = np.max(prediction) * 100

    # Show result in glassmorphic styled card
    st.markdown(
        f"""
        <div class="pred-card">
            <h3>Prediction: <b>{pred_label}</b></h3>
            <p>Confidence: {confidence:.2f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
