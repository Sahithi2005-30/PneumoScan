# 🫁 PnemoScan – Pneumonia Detection System

## Overview

**PnemoScan** is a deep learning-based healthcare application that detects pneumonia from chest X-ray images. It classifies uploaded images into **Normal**, **Pneumonia**, and **Invalid Image** categories. The system rejects non-X-ray images, highlights the suspected pneumonia region (ROI), and integrates a Gemini-powered healthcare chatbot for general guidance.

## Key Features

- Real-time pneumonia detection
- Three-class classification (Normal, Pneumonia, Invalid)
- Invalid image detection
- ROI highlighting
- Flask web application
- Gemini AI chatbot
- User-friendly interface

## Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB3
- Flask
- HTML
- CSS
- JavaScript
- NumPy
- Pillow (PIL)
- Gemini API

## Dataset

- Normal
- Pneumonia
- Invalid

## Data Preprocessing

- Resize images (224×224)
- Normalize pixel values
- Convert to NumPy arrays
- Data augmentation (rotation, zoom, flip, brightness)

## Model

EfficientNetB3 with Transfer Learning and Fine-Tuning.

## Workflow

1. Upload chest X-ray.
2. Flask receives the image.
3. Image preprocessing.
4. EfficientNetB3 predicts the class.
5. Display prediction.
6. Highlight ROI.
7. Gemini chatbot provides healthcare guidance.

## Installation

```bash
git clone https://github.com/your-username/PnemoScan.git
cd PnemoScan
pip install -r requirements.txt
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Future Enhancements

- Multi-disease detection
- Cloud deployment
- Mobile application
- Hospital integration

## Author

**Sahithi Kondeti**

## License

This project is intended for educational and research purposes only.
