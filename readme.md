# 🐄 COWSHALA

Cowshala is a Flask-based web application designed to assist farmers and veterinarians by offering two key functionalities:
Cattle Breed Prediction: Identify cattle breeds from uploaded images using a trained ResNet18 model.
Cattle Disease Diagnosis: Predict cattle diseases based on symptom inputs using a trained Random Forest classifier.

Additionally, the application uses Google Gemini AI to generate insights and provide treatment and care recommendations based on regional practices.

## Features
Breed Prediction: Upload cattle images to identify their breed using a customized ResNet18 model.
Disease Diagnosis: Input various symptoms to receive disease predictions from a Random Forest model.
AI-Generated Insights: Uses Google Gemini AI to provide detailed insights on cattle breed characteristics and treatment suggestions.
Responsive Design: Templates and static files have been structured for an intuitive user experience.

## Tech Stack
Backend: Flask, Python
Machine Learning: PyTorch (ResNet18), Scikit-learn (RandomForestClassifier)
Data Handling: Pandas, NumPy, joblib
Image Processing: PIL (Python Imaging Library)
AI Integration: Google Generative AI (Gemini)
Frontend: HTML, CSS

## 📁 Project Structure
```
cowshala/
├── .idea/                        # IDE configuration files
├── data/
│   └── Cow_Disease_Train.xlsx    # Dataset for disease prediction
├── models/
│   ├── cattle_breed_classifier_full_model.pth  # Pre-trained breed classifier
│   ├── breed_labels.txt         # Corresponding breed labels
│   └── disease_model.pkl        # Trained disease prediction model (saved after training)
├── static/
│   └── uploads/                 # Uploaded images for breed prediction
├── styles/                      # CSS files for styling
├── templates/                   # HTML templates
│   ├── index.html
│   ├── breed.html
│   ├── disease.html
│   ├── disease_prediction.html
│   └── disease_dashboard.html
├── app.py                       # Main Flask application
└── requirements.txt             # Project dependencies
```

### Setup & Installation

Clone the Repository:
```
git clone https://github.com/MoniDeWotah/cowshala.git
cd cowshala
```

Create a Virtual Environment (Optional but Recommended):
```
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate     # For Windows
```

Install Dependencies:
```
pip install -r requirements.txt
```

Set your Gemini API key as an environment variable:
```
export GEMINI_API_KEY="your-api-key-here"  # For Linux/Mac
set GEMINI_API_KEY=your-api-key-here       # For Windows CMD
```

Run the Application:
```
python app.py
# The app will be accessible at http://127.0.0.1:5000.
```
