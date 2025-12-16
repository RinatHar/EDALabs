from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import joblib
import os
from eda import create_eda_report
from model import train_model, predict

app = Flask(__name__)
CORS(app)  # Разрешить все источники

DATA_PATH = 'data/online_shoppers.csv'
MODEL_PATH = 'model/shopper_model.pkl'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return jsonify({
            'status': 'success',
            'data': df.head(100).to_dict(orient='records'),
            'shape': df.shape,
            'columns': list(df.columns)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/eda', methods=['GET'])
def generate_eda():
    try:
        report_path = create_eda_report()
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train():
    try:
        accuracy = train_model()
        return jsonify({
            'status': 'success',
            'message': f'Model trained with accuracy: {accuracy:.4f}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def make_prediction():
    try:
        data = request.json
        prediction = predict(data)
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'result': 'Purchase' if prediction == 1 else 'No Purchase'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'online-shoppers-api'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)