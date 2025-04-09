from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from torchvision import transforms, models
from torch import nn
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__, static_folder='static')

# --- Configuration ---
BREED_MODEL_PATH = "models/cattle_breed_classifier_full_model.pth"
BREED_LABELS_PATH = "models/breed_labels.txt"
DISEASE_MODEL_PATH = "models/disease_model.pkl"
DISEASE_LABEL_ENCODER_PATH = "models/disease_label_encoder.pkl"
DEVICE = torch.device("cpu")  # Force CPU for Render
CURRENT_LOCATION = "Indore, Madhya Pradesh, India"

# --- Load breed labels ---
breed_labels = []
try:
    with open(BREED_LABELS_PATH, "r") as f:
        breed_labels = [line.strip() for line in f]
except Exception as e:
    print(f"Failed to load breed labels: {e}")

# --- Load breed model (CPU Only) ---
breed_model = None
if breed_labels:
    try:
        num_classes = len(breed_labels)
        breed_model = models.resnet18(weights=None)
        breed_model.fc = nn.Linear(breed_model.fc.in_features, num_classes)
        breed_model.load_state_dict(torch.load(BREED_MODEL_PATH, map_location=DEVICE))
        breed_model.to(DEVICE)
        breed_model.eval()
        print("Breed model loaded on CPU.")
    except Exception as e:
        print(f"Failed to load breed model: {e}")

# --- Load disease model ---
disease_model = None
try:
    disease_model = joblib.load(DISEASE_MODEL_PATH)
    print("Disease model loaded.")
except Exception as e:
    print(f"Failed to load disease model: {e}")

# --- Load disease label encoder ---
disease_label_encoder = None
try:
    disease_label_encoder = joblib.load(DISEASE_LABEL_ENCODER_PATH)
    print("Disease label encoder loaded.")
except Exception as e:
    print(f"Failed to load label encoder: {e}")

# --- Gemini Setup ---
GEMINI_API_KEY = "AIzaSyA7mhqa0nWST2zY0m-fwhoPt8EXwIk2bqE"  # Replace if needed
gemini_model = None
try:
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")
except Exception as e:
    print(f"Failed to configure Gemini: {e}")

# --- Image Transform ---
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Routes ---
@app.route("/")
def home():
    return render_template("index.html", location=CURRENT_LOCATION)

@app.route("/breed")
def breed_page():
    return render_template("breed.html", location=CURRENT_LOCATION)

@app.route("/predict_breed", methods=["POST"])
def predict_breed():
    if not breed_model or 'file' not in request.files:
        return render_template("breed.html", error="Model not available or no file uploaded.", location=CURRENT_LOCATION)

    file = request.files['file']
    if file.filename == '':
        return render_template("breed.html", error="No file selected.", location=CURRENT_LOCATION)

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join("static/uploads", filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(filepath)

        img = Image.open(filepath).convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = breed_model(img_tensor)
            predicted_idx = torch.argmax(output, 1).item()
            predicted_label = breed_labels[predicted_idx]

        insights = None
        if gemini_model:
            try:
                prompt = f"Give insights about cattle breed '{predicted_label}' and its usefulness in {CURRENT_LOCATION}."
                response = gemini_model.generate_content(prompt)
                insights = response.text
            except:
                insights = "Gemini insights could not be retrieved."

        return render_template("breed.html", prediction=predicted_label, image_path=filepath, insights=insights, location=CURRENT_LOCATION)
    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template("breed.html", error="Breed prediction failed.", location=CURRENT_LOCATION)

@app.route("/disease")
def disease_page():
    return render_template("disease.html", location=CURRENT_LOCATION)

@app.route("/disease_prediction")
def disease_prediction_page():
    return render_template("disease_prediction.html", location=CURRENT_LOCATION)

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    if not disease_model or not disease_label_encoder:
        return render_template("disease_prediction.html", prediction="Model not available.", location=CURRENT_LOCATION)

    try:
        symptoms = [
            'Fever', 'Weight_Loss', 'Reduced_Milk', 'Diarrhea', 'Lameness', 'Cough', 'Swollen_Udder',
            'Skin_Nodules', 'Blisters_Mouth', 'Nasal_Discharge', 'Infertility', 'Blood_Oozing',
            'Muscle_Swelling', 'Loss_of_Appetite', 'Abortion', 'Sudden_Death'
        ]

        features = []
        for symptom in symptoms:
            val = request.form.get(symptom)
            try:
                features.append(float(val))
            except:
                features.append(0.0)

        input_data = np.array(features).reshape(1, -1)
        print("Symptom values:", features)

        predicted_class_index = disease_model.predict(input_data)[0]
        predicted_disease = disease_label_encoder.inverse_transform([predicted_class_index])[0]

        if hasattr(disease_model, "predict_proba"):
            proba = disease_model.predict_proba(input_data)[0]
            confidence = f" (Confidence: {round(max(proba)*100, 2)}%)"
        else:
            confidence = ""

        advice = None
        if gemini_model:
            try:
                prompt = f"Give treatment and care advice for cattle disease '{predicted_disease}' in {CURRENT_LOCATION}."
                response = gemini_model.generate_content(prompt)
                advice = response.text
            except:
                advice = "Gemini advice could not be retrieved."

        return render_template(
            "disease_prediction.html",
            prediction=predicted_disease + confidence,
            advice=advice,
            location=CURRENT_LOCATION
        )
    except Exception as e:
        print(f"Disease prediction error: {e}")
        return render_template("disease_prediction.html", prediction="Prediction error.", location=CURRENT_LOCATION)

if __name__ == "__main__":
    app.run(debug=True)
