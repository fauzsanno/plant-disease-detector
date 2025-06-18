import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Konstanta
IMAGE_SIZE = 256
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]  # Sesuaikan jika nama kelas berbeda

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/1.h5")
    return model

model = load_model()

# UI Streamlit
st.title("🌿 Deteksi Penyakit Daun Kentang")
st.write("Upload gambar daun kentang, dan model akan memprediksi penyakitnya.")

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload gambar daun...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    # Output
    st.markdown(f"### 🧪 Prediksi: `{predicted_class}`")
    st.markdown(f"### 🔍 Keyakinan Model: `{confidence}%`")
