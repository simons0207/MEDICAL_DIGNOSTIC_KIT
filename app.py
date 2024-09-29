from flask import Flask, render_template, request, flash, redirect,url_for
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


app = Flask(__name__)

malaria_model = load_model('models/malaria2.h5')
try:
    model = tf.keras.models.load_model('models/pneumonia1.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def prepare_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((200, 200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    print("Image shape:", img_array.shape)
    prediction = model.predict(img_array)
    predicted_class = (prediction[0][0] > 0.5).astype("int32")
    label = 'Uninfected' if predicted_class == 1 else 'Infected'
    
    return label, prediction[0][0]

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            print("Form Data Received:", to_predict_dict)
            print("List of Values:", to_predict_list)
            pred = predict(to_predict_list, to_predict_dict)
            print("Prediction Result:", pred)
    except Exception as e:
        print("Error:", e)
        message = "Please enter valid Data"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred = pred)

@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    if 'image' not in request.files:
        message = "Please upload an image"
        return render_template('malaria.html', message=message)

    file = request.files['image']

    if file.filename == '':
        message = "No selected file"
        return render_template('malaria.html', message=message)

    if file:
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        label, probability = model_predict(file_path, malaria_model)

        pred = 1 if label == 'Infected' else 0

        return render_template('malaria_predict.html', pred=pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if model is None:
        return render_template('pneumonia.html', message="Model not loaded. Please contact the administrator.")

    if 'image' not in request.files:
        return render_template('pneumonia.html', message="No file part")
    
    file = request.files['image']
    
    if file.filename == '':
        return render_template('pneumonia.html', message="No selected file")
    
    if file:
        img = Image.open(file.stream)
        prepared_image = prepare_image(img)
        prediction = model.predict(prepared_image)
        print(f"Raw prediction: {prediction}")
        pred = 1 if prediction[0][0] > 0.5 else 0
        return render_template('pneumonia_predict.html', pred=pred, prediction=prediction[0][0])

if __name__ == '__main__':
	app.run(debug = True)