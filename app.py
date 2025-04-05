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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import google.generativeai as genai

app = Flask(__name__)

# ====== CONFIG ======
BREED_MODEL_PATH = "models/cattle_breed_classifier_full_model.pth"
BREED_LABELS_PATH = "models/breed_labels.txt"
DISEASE_DATA_PATH = "data/Cow_Disease_Train.xlsx"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_LOCATION = "Indore, Madhya Pradesh, India"  # Define location

# ====== LOAD BREED LABELS ======
try:
    with open(BREED_LABELS_PATH, "r") as f:
        breed_labels = f.read().splitlines()
    num_classes = len(breed_labels)
except Exception as e:
    print(f"Error loading breed labels: {e}")
    breed_labels = []
    num_classes = 0

# ====== LOAD BREED MODEL ======
breed_model = None
if num_classes > 0:
    try:
        breed_model = models.resnet18(pretrained=False)  # pretrained=False is CRITICAL
        breed_model.fc = nn.Linear(breed_model.fc.in_features, num_classes)
        breed_model.load_state_dict(torch.load(BREED_MODEL_PATH, map_location=DEVICE))
        breed_model.to(DEVICE)
        breed_model.eval()
        print("Breed model loaded successfully.")
    except Exception as e:
        print(f"Error loading breed model: {e}")
        breed_model = None
else:
    print("Breed labels not available, skipping breed model load.")

# ====== IMAGE TRANSFORM ======
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

# ====== CONFIGURE GEMINI ======
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ====== LOAD AND TRAIN DISEASE MODEL ======
disease_model = None
try:
    df = pd.read_excel(DISEASE_DATA_PATH)
    X = df.drop("Disease", axis=1)
    y = df["Disease"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    disease_model = RandomForestClassifier(n_estimators=100, random_state=42)
    disease_model.fit(X_train, y_train)

    y_pred = disease_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Disease Model Accuracy: {acc:.2f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(disease_model, "models/disease_model.pkl")
except Exception as e:
    print(f"Error training disease model: {e}")
    disease_model = None

# ====== ROUTES ======

@app.route("/")
def home():
    return render_template("index.html", location=CURRENT_LOCATION)

@app.route("/breed")
def breed_page():
    return render_template("breed.html", location=CURRENT_LOCATION)

@app.route("/predict_breed", methods=["POST"])
def predict_breed():
    if not breed_model:
        return "Breed prediction model is not available."

    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    img_tensor = img_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = breed_model(img_tensor)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_label = breed_labels[predicted_idx.item()]

    # Gemini insights
    gemini_prompt = f"Give insights about cattle breed '{predicted_label}' including its characteristics and usefulness in the region of {CURRENT_LOCATION}."
    gemini_response = gemini_model.generate_content(gemini_prompt)

    return render_template("breed.html", prediction=predicted_label, image_path=filepath, insights=gemini_response.text, location=CURRENT_LOCATION)

@app.route("/disease")
def disease_page():
    return render_template("disease.html", location=CURRENT_LOCATION)

@app.route("/disease_prediction")
def disease_prediction_page():
    return render_template("disease_prediction.html", location=CURRENT_LOCATION)

@app.route("/disease_dashboard.html")
def disease_dashboard_page():
    return render_template("disease_dashboard.html", location=CURRENT_LOCATION)

@app.route("/predict_disease", methods=["POST"])
def predict_disease():
    if not disease_model:
        return render_template("disease.html", prediction="Disease prediction model not available.", location=CURRENT_LOCATION)

    try:
        symptoms = [
            'Fever', 'Weight_Loss', 'Reduced_Milk', 'Diarrhea', 'Lameness', 'Cough', 'Swollen_Udder',
            'Skin_Nodules', 'Blisters_Mouth', 'Nasal_Discharge', 'Infertility', 'Blood_Oozing',
            'Muscle_Swelling', 'Loss_of_Appetite', 'Abortion', 'Sudden_Death'
        ]
        features = [float(request.form.get(symptom, 0)) for symptom in symptoms]
        input_data = np.array(features).reshape(1, -1)
        prediction = disease_model.predict(input_data)

        gemini_prompt = f"Give treatment and care advice for cattle disease '{prediction[0]}' considering the common practices and availability of resources in {CURRENT_LOCATION}."
        gemini_response = gemini_model.generate_content(gemini_prompt)

        return render_template("disease.html", prediction=prediction[0], advice=gemini_response.text, location=CURRENT_LOCATION)
    except Exception as e:
        return render_template("disease.html", prediction=f"Error: {str(e)}", location=CURRENT_LOCATION)

if __name__ == "__main__":
    app.run(debug=True)