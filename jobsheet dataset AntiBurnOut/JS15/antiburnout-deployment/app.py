import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'burnout_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')

# Load trained model & preprocessor
model = None
scaler_data = None

def get_model():
    global model, scaler_data
    if model is None and os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
    if scaler_data is None and os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, 'rb') as f:
            scaler_data = pickle.load(f)
    return model, scaler_data

FEATURE_NAMES = [
    'Age', 'Work_Hours_Per_Week', 'Sleep_Hours', 'Work_Life_Balance',
    'Job_Satisfaction', 'Anxiety_Score', 'Depression_Score',
    'Mood_Score', 'Physical_Activity_Hours', 'Screen_Time_Hours'
]

RECOMMENDATIONS = {
    'Low': {
        'status': 'Risiko Rendah (Sehat)',
        'color': 'success',
        'badge': 'bg-green-100 text-green-800 border-green-300',
        'advice': 'Kondisi kesehatan mental dan keseimbangan kerja Anda berada dalam rentang sangat baik. Pertahankan pola tidur teratur, aktivitas fisik rutin, dan hubungan sosial yang positif.'
    },
    'Moderate': {
        'status': 'Risiko Sedang (Waspada)',
        'color': 'warning',
        'badge': 'bg-yellow-100 text-yellow-800 border-yellow-300',
        'advice': 'Terdapat indikasi kelelahan kerja awal. Disarankan untuk mengambil jeda istirahat terencana (micro-breaks), mengurangi jam lembur, dan mendiskusikan beban kerja dengan tim Anda.'
    },
    'High': {
        'status': 'Risiko Tinggi (Kritis)',
        'color': 'danger',
        'badge': 'bg-red-100 text-red-800 border-red-300',
        'advice': 'Tingkat stres dan indikator kelelahan berada pada level kritis. Sangat direkomendasikan untuk segera berkonsultasi dengan profesional konseling/psikolog serta mengambil cuti pemulihan.'
    }
}

@app.route('/')
def home():
    return render_template('index.html', features=FEATURE_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        mdl, pre = get_model()
        if mdl is None or pre is None:
            return jsonify({'error': 'Model belum siap atau artefak tidak ditemukan'}), 500
        
        # Read parameters from form or json
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form
        
        input_values = []
        for feat in FEATURE_NAMES:
            val = float(data.get(feat, 0.0))
            input_values.append(val)
        
        input_array = np.array([input_values])
        input_scaled = pre['scaler'].transform(input_array)
        
        preds = mdl.predict(input_scaled)[0]
        class_idx = int(np.argmax(preds))
        classes = ['Low', 'Moderate', 'High']
        risk_label = classes[class_idx]
        confidence = float(preds[class_idx]) * 100
        
        result = {
            'risk_level': risk_label,
            'confidence': round(confidence, 2),
            'probabilities': {
                'Low': round(float(preds[0]) * 100, 2),
                'Moderate': round(float(preds[1]) * 100, 2),
                'High': round(float(preds[2]) * 100, 2)
            },
            'meta': RECOMMENDATIONS[risk_label]
        }
        
        if request.is_json:
            return jsonify(result)
        return render_template('index.html', features=FEATURE_NAMES, result=result, last_inputs=data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
