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
import glob

app = Flask(__name__, static_folder='static')

# --- Configuration ---
BREED_MODEL_PATH = "models/cattle_breed_classifier_full_model.pth"
BREED_LABELS_PATH = "models/breed_labels.txt"
DISEASE_MODEL_PATH = "models/disease_model.pkl"
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

# --- Gemini Setup ---
GEMINI_API_KEY = "AIzaSyA7mhqa0nWST2zY0m-fwhoPt8EXwIk2bqE"
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
        # --- Clean old uploads ---
        for f in glob.glob("static/uploads/*"):
            os.remove(f)

        # --- Save new upload ---
        filename = secure_filename(file.filename)
        filepath = os.path.join("static/uploads", filename)
        os.makedirs("static/uploads", exist_ok=True)
        file.save(filepath)

        # --- Process image ---
        img = Image.open(filepath).convert("RGB")
        img_tensor = img_transform(img).unsqueeze(0).to(DEVICE)

        # --- Predict ---
        with torch.no_grad():
            output = breed_model(img_tensor)
            predicted_idx = torch.argmax(output, 1).item()
            predicted_label = breed_labels[predicted_idx]

        # --- Cleanup tensors ---
        del img_tensor, output
        torch.cuda.empty_cache()  # Safe even on CPU

        # --- Gemini Insights ---
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
    if not disease_model:
        return render_template("disease_prediction.html", prediction="Model not available.", location=CURRENT_LOCATION)

    try:
        symptoms = [
            'Fever', 'Weight_Loss', 'Reduced_Milk', 'Diarrhea', 'Lameness', 'Cough', 'Swollen_Udder',
            'Skin_Nodules', 'Blisters_Mouth', 'Nasal_Discharge', 'Infertility', 'Blood_Oozing',
            'Muscle_Swelling', 'Loss_of_Appetite', 'Abortion', 'Sudden_Death'
        ]
        features = [float(request.form.get(symptom, 0)) for symptom in symptoms]
        input_data = np.array(features).reshape(1, -1)
        prediction = disease_model.predict(input_data)[0]

        advice = None
        if gemini_model:
            try:
                prompt = f"Give treatment and care advice for cattle disease '{prediction}' in {CURRENT_LOCATION}."
                response = gemini_model.generate_content(prompt)
                advice = response.text
            except:
                advice = "Gemini advice could not be retrieved."

        return render_template("disease_prediction.html", prediction=prediction, advice=advice, location=CURRENT_LOCATION)
    except Exception as e:
        print(f"Disease prediction error: {e}")
        return render_template("disease_prediction.html", prediction="Prediction error.", location=CURRENT_LOCATION)

