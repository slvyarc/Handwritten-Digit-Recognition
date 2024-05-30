import sys
import os

if not os.environ.get("PYTHONIOENCODING"):
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf8', buffering=1)

import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# Buat direktori 'prediction'
if not os.path.exists('prediction'):
    os.makedirs('prediction')

# Fungsi prediksi
def predictDigit(image):
    model = tf.keras.models.load_model("model/mnist_model.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32')
    img = img / 255.0
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result, pred[0], img.reshape((28, 28))

# Konfigurasi halaman Streamlit
st.set_page_config(page_title='Handwritten Digit Recognition', layout='wide')
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #6F26DC; 
        text-align: center;
    }
    .sub-title {
        font-size: 1.2em;
        color: #6F26DC;
        text-align: center;
        margin-bottom: 1em;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
    }
    .stButton button {
        background-color: #6F26DC;
        color: white;
        border-radius: 5px;
        font-size: 1em;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
    }
    .canvas-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        margin-top: 20px;
    }
    .sidebar .element-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 10px;
    }
    .stSlider > div > div > div > div {
        background: #6F26DC; 
    }
    .stSlider label {
    color: #000000; 
    }
    .stSlider > div > div > div {
    color: #000000; /* Change slider number color to white */
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown('<div class="main-title">Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Draw the digit on canvas and click on "Predict"</div>', unsafe_allow_html=True)

# Komponen canvas
drawing_mode = "freedraw"
stroke_width = st.slider('Stroke Width', 1, 20, 7)
stroke_color = '#FFFFFF'  # Set stroke color to white
bg_color = '#000000'  # Set background color to black

# Membuat komponen canvas dalam container 
st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=250,
    width=250,
    key="canvas",
)
st.markdown('</div>', unsafe_allow_html=True)

# Tombol prediksi dalam container yang sama
if st.button('Predict'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image = input_image.convert('RGB')  # Convert to RGB before saving
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res, probabilities, processed_img = predictDigit(img)
        
        st.markdown(f'<h1 style="color:black;">Prediction: {res}</h1>', unsafe_allow_html=True)
        
        # Visualisasi Gambar Input
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        
        # Tampilkan gambar asli
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title("Input Image")
        ax[0].axis('off')
        
        # Tampilkan heatmap
        sns.heatmap(processed_img, ax=ax[1], cmap="viridis", cbar=False)
        ax[1].set_title("Processed Image")
        ax[1].axis('off')
        
        # Tampilkan probabilitas prediksi
        colors = ['#FF1493', '#7FFF00', '#00FFFF', '#FF69B4', '#8A2BE2', '#DC143C', '#FFD700', '#4B0082', '#00FF7F', '#1E90FF']
        ax[2].bar(range(10), probabilities, color=colors)
        ax[2].set_xticks(range(10))
        ax[2].set_title("Prediction Probabilities")
        ax[2].set_xlabel("Digits")
        ax[2].set_ylabel("Probability")
        
        # Tambahkan legenda untuk menjelaskan warna
        legend_labels = ['Digit ' + str(i) for i in range(10)]
        handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(10)]
        ax[2].legend(handles, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left')

        st.pyplot(fig)
    else:
        st.header('Please draw a digit on the canvas.')

# Sidebar
st.sidebar.markdown('<div class="element-container">', unsafe_allow_html=True)
image_path = "gambar.png"
if os.path.exists(image_path):
    st.sidebar.image(image_path, caption='Digit Recognition', use_column_width=True)
else:
    st.sidebar.write("Image not found.")
st.sidebar.write("### Connect with me:")
st.sidebar.write("Silvia Dharma")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/silviadharma)")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/slvyarc/Handwritten-Digit-Recognition.git)")
st.sidebar.write("For inquiries and collaborations, feel free to contact me!")
st.sidebar.write("Keep riding and stay healthy!")
st.sidebar.markdown('</div>', unsafe_allow_html=True)
