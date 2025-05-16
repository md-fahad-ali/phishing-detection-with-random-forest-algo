from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import warnings
from flask_cors import CORS  


warnings.filterwarnings('ignore')


from phishing_detection import PhishingDetector

app = Flask(__name__)
CORS(app)  


@app.before_request
def load_model():
    global detector
    
    
    if not os.path.exists('phishing_detector.pkl'):
        print("Model not found. Please run phishing_detection.py first to train the model.")
        return jsonify({'error': 'Model not found. Please run phishing_detection.py first.'}), 500
    
    
    if not globals().get('detector'):
        detector = PhishingDetector.load('phishing_detector.pkl')


def normalize_url(url):
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    
    if ':' in domain:
        domain = domain.split(':')[0]
    
    
    if domain.startswith('www.'):
        domain = domain[4:]
    
    
    normalized_url = f"{parsed_url.scheme}://{domain}{parsed_url.path}"
    if parsed_url.params:
        normalized_url += f";{parsed_url.params}"
    if parsed_url.query:
        normalized_url += f"?{parsed_url.query}"
    if parsed_url.fragment:
        normalized_url += f"
    
    return normalized_url


def predict_phishing(urls):
    
    normalized_urls = [normalize_url(url) for url in urls]
    
    
    predictions, probabilities = detector.predict(normalized_urls)
    
    return predictions, probabilities, normalized_urls

@app.route('/api/check', methods=['POST'])
def check_url():
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400
    
    url = data['url']
    
    
    try:
        predictions, probabilities, normalized_urls = predict_phishing([url])
        is_phishing = bool(predictions[0])
        probability = float(probabilities[0])
        
        return jsonify({
            'url': url,
            'normalized_url': normalized_urls[0],
            'is_phishing': is_phishing,
            'phishing_probability': probability,
            'prediction': 'Phishing' if is_phishing else 'Not Phishing'
        })
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/check-batch', methods=['POST'])
def check_urls_batch():
    data = request.get_json()
    
    if not data or 'urls' not in data:
        return jsonify({'error': 'No URLs provided'}), 400
    
    urls = data['urls']
    
    if not isinstance(urls, list):
        return jsonify({'error': 'URLs must be provided as a list'}), 400
    
    try:
        predictions, probabilities, normalized_urls = predict_phishing(urls)
        
        results = []
        for url, norm_url, pred, prob in zip(urls, normalized_urls, predictions, probabilities):
            results.append({
                'url': url,
                'normalized_url': norm_url,
                'is_phishing': bool(pred),
                'phishing_probability': float(prob),
                'prediction': 'Phishing' if pred else 'Not Phishing'
            })
        
        return jsonify({'results': results})
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def index():
    return """
    <html>
        <head>
            <title>Phishing URL Detection API</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                h1 { color: 
                pre { background-color: 
                .endpoint { margin-bottom: 20px; }
            </style>
        </head>
        <body>
            <h1>Phishing URL Detection API</h1>
            <p>This API provides endpoints to check URLs for phishing.</p>
            
            <div class="endpoint">
                <h2>Check a single URL</h2>
                <p><strong>Endpoint:</strong> POST /api/check</p>
                <p><strong>Request body:</strong></p>
                <pre>
{
    "url": "https://example.com"
}
                </pre>
                <p><strong>Response:</strong></p>
                <pre>
{
    "url": "https://example.com",
    "normalized_url": "https://example.com",
    "is_phishing": false,
    "phishing_probability": 0.02,
    "prediction": "Not Phishing"
}
                </pre>
            </div>
            
            <div class="endpoint">
                <h2>Check multiple URLs</h2>
                <p><strong>Endpoint:</strong> POST /api/check-batch</p>
                <p><strong>Request body:</strong></p>
                <pre>
{
    "urls": [
        "https://example.com",
        "http://suspicious-site.xyz"
    ]
}
                </pre>
                <p><strong>Response:</strong></p>
                <pre>
{
    "results": [
        {
            "url": "https://example.com",
            "normalized_url": "https://example.com",
            "is_phishing": false,
            "phishing_probability": 0.02,
            "prediction": "Not Phishing"
        },
        {
            "url": "http://suspicious-site.xyz",
            "normalized_url": "http://suspicious-site.xyz",
            "is_phishing": true,
            "phishing_probability": 0.98,
            "prediction": "Phishing"
        }
    ]
}
                </pre>
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug_mode)