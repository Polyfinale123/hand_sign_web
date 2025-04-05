from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import pickle
import os

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_url = request.json['image']
        encoded_data = data_url.split(',')[1]
        img_data = base64.b64decode(encoded_data)
        image = Image.open(io.BytesIO(img_data)).convert('RGB')
        image = image.resize((64, 64))
        img_array = np.array(image).reshape(1, -1)
        
        prediction = model.predict(img_array)[0]
        return jsonify({'prediction': str(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
