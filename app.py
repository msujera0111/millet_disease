import os
from flask import Flask, redirect, render_template, request, jsonify
from PIL import Image
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from io import BytesIO

# Load model and data
keras_model = load_model("CNN-sorghum_30ep-4-05_2025-05-05.keras")
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')


app = Flask(__name__)

def predict_image(image_file):
    try:
        img = Image.open(image_file).convert("RGB")
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_tensor = tf.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = keras_model.predict(img_tensor)
        predicted_class = np.argmax(prediction[0])

        # Define the class index to label mapping
        class_labels = {
            0: 'Anthracnose and Red Rot',
            1: 'Cereal Grain molds',
            2: 'Covered Kernel smut',
            3: 'Head Smut',
            4: 'Healthy',
            5: 'Rust',
            6: 'loose smut'
        }
        # Logging
        print(f"Predicted Class Index: {predicted_class}")
        print(f"Predicted Class Label: {class_labels.get(predicted_class, 'Unknown')}")

        return int(predicted_class)

    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg','.JPEG','.PNG','JPG'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']

        if not image or not allowed_file(image.filename):
            return jsonify({"error": "Invalid file format. Only .png, .jpg, .jpeg allowed."})

        try:
            image_bytes = BytesIO(image.read())  # In-memory image

            pred = predict_image(image_bytes)

            if isinstance(pred, dict) and 'error' in pred:
                return jsonify(pred)

            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

            return render_template(
                'submit.html',
                title=title,
                desc=description,
                prevent=prevent,
                image_url=image_url,
                pred=pred,
                sname=supplement_name,
                simage=supplement_image_url,
                buy_link=supplement_buy_link
            )

        except Exception as e:
            return jsonify({"error": str(e)})
        
@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  
    app.run(host='0.0.0.0', port=port, debug=False)
