PneumoScan — Pneumonia Detection System
#---------Project Structure-----------#
PNEUMONIA_DSP/
├── app.py                          # Flask backend and prediction pipeline
├── train.py                        # Deep Learning training script
├── requirements.txt               # Python dependencies
├── templates/
│   └── index.html                 # Frontend UI
├── static/
│   └── uploads/                   # Uploaded and ROI images
├── models/
│   ├── pneumonia_3class_model.h5
│   └── class_indices.txt
└── chest_xray/
    ├── train/
    ├── val/
    └── test/

#------Project Overview-----#
PneumoScan is a Deep Learning-based healthcare application developed to classify chest X-ray images into Normal, Pneumonia, and Invalid Image categories. The system performs image preprocessing, model prediction, confidence-based decision handling, ROI visualization, and healthcare assistance through a Gemini 2.0-powered chatbot.

#------Features-----#
-Deep Learning Model (train.py)
-Developed using EfficientNetB0/EfficientNetB3
-Supports 3-class classification:
   Normal  Pneumonia  Invalid Image
-Image preprocessing and augmentation
-Class balancing using class weights
-Two-phase transfer learning and fine-tuning
-Confidence-based prediction handling
-Validation and testing pipeline

#----Backend (app.py)-----#
-Flask-based prediction API
-Real-time chest X-ray classification
-Confidence-based decision handling
-ROI visualization for pneumonia cases
-Invalid image detection
-Rule-based validation for:
    QR codes   Documents    Random images   Non-chest X-ray inputs
-Gemini 2.0 healthcare chatbot integration

#-----Frontend (templates/index.html)-----#
-Drag-and-drop image upload
-Responsive user interface
-Real-time prediction results
-Confidence score visualization
-ROI image display for pneumonia cases
-Healthcare chatbot support
-Mobile-friendly design

#------Gemini Healthcare Chatbot------#
-The application integrates Google Gemini 2.0 to provide:
-Medical awareness information
-Pneumonia-related explanations
-Symptoms and precaution guidance
-Prediction explanations
-General healthcare assistance

Note: This project uses the free version of Google Gemini API. Due to free-tier usage limitations, the chatbot can answer only a limited number of questions. Once the free quota is exhausted, the chatbot functionality becomes temporarily unavailable until the quota resets.

#-------Technologies Used------#
-Python
-TensorFlow
-Keras
-EfficientNetB0 / EfficientNetB3
-Flask
-NumPy
-Pillow (PIL)
-HTML
-CSS
-JavaScript
-Google Gemini API

#-------Local Development--------#
pip install -r requirements.txt
python app.py
Visit: http://localhost:5000
Set your Gemini API key:
export GOOGLE_API_KEY="your-api-key"

#------Model Training----#
Run:
python train.py
Dataset structure:
chest_xray/
├── train/
│   ├── INVALID IMAGES/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── INVALID IMAGES/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── INVALID IMAGES/
    ├── NORMAL/
    └── PNEUMONIA/

#--------Project Workflow----------#
Upload Image
        ↓
Image Validation
        ↓
Image Preprocessing
        ↓
Deep Learning Prediction
        ↓
Normal / Pneumonia / Invalid
        ↓
Confidence Score Generation
        ↓
ROI Visualization
        ↓
Gemini Healthcare Assistance

#-------Future Improvements-----------#
Improve invalid image detection using larger datasets
Add ultrasound, CT, MRI, and document datasets
Improve confidence calibration
Deploy to cloud platforms
Enhance ROI localization accuracy

Disclaimer:
This project is developed for educational and research purposes only and is not intended to replace professional medical diagnosis. Users should consult qualified healthcare professionals for medical decisions.
