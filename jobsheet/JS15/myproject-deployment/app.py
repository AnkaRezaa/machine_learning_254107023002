import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Path Model & Scaler Proyek Anda
MODEL_PATH = 'model.h5'
SCALER_PATH = 'scaler.pkl'

model = None
scaler = None

try:
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")

@app.route('/', methods=['GET'])
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Machine Learning Project Classifier</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; text-align: center; padding: 50px; }
            .container { background: white; max-width: 500px; margin: auto; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            h2 { color: #333; margin-bottom: 20px; }
            input[type=file] { margin: 20px 0; }
            button { background-color: #007bff; color: white; border: none; padding: 10px 24px; font-size: 16px; border-radius: 6px; cursor: pointer; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>ML Project Classifier</h2>
            <p>Silakan unggah berkas citra untuk melakukan inferensi prediksi.</p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br>
                <button type="submit">Prediksi Citra</button>
            </form>
        </div>
    </body>
    </html>
    """)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        nparr = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Sesuaikan dengan target ukuran citra model Anda
        img_resized = cv2.resize(img, (64, 64)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)

        prediction = model.predict(img_input)
        prob = float(prediction[0][0]) if prediction.shape[-1] == 1 else float(np.max(prediction[0]))
        result_label = f"Hasil Prediksi (Probabilitas: {prob:.4f})"

        return f"<div style='text-align:center; padding:50px;'><h2>{result_label}</h2><br><a href='/'>Kembali</a></div>"
    except Exception as e:
        return f"<div style='text-align:center; padding:50px;'><h3>Terjadi Kesalahan: {e}</h3><br><a href='/'>Kembali</a></div>"

if __name__ == '__main__':
    # Port 7860 wajib untuk Hugging Face Spaces
    app.run(host='0.0.0.0', port=7860)
