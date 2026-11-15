# AntiBurnOut AI Screening Service - Hugging Face Deployment

Aplikasi skrining dan prediksi risiko burnout berbasis Machine Learning & Deep Neural Network (ANN) menggunakan dataset `mental_health_burnout_prediction_dataset.csv`.

**Identitas Mahasiswa:**
- Nama : MUHAMMAD ANKA REZA
- NIM  : 254107023002
- Kelas: TI/3H

## Struktur Berkas
- `app.py`: Server web Flask yang menyediakan UI formulir dan endpoint REST API `/predict`.
- `burnout_model.h5`: Model Neural Network Keras yang telah dilatih.
- `preprocessor.pkl`: Objek scaler standardisasi data.
- `Dockerfile`: File konfigurasi container untuk cloud platform.
- `templates/index.html`: Antarmuka formulir skrining berbasis Tailwind CSS.

## Menjalankan Secara Lokal
```bash
pip install -r requirements.txt
python app.py
```
Akses di peramban: `http://localhost:7860`

## Menjalankan dengan Docker
```bash
docker build -t antiburnout-app .
docker run -p 7860:7860 antiburnout-app
```
