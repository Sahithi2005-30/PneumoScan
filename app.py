import os
import uuid
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image as PILImage

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from google import genai


# ─── LOAD ENV ─────────────────────────────────────────────
load_dotenv()


# ─── CONFIG ───────────────────────────────────────────────
IMG_SIZE = 224

MODEL_PATH = os.environ.get("MODEL_PATH", "models/pneumonia_3class_model.h5")

UPLOAD_DIR = "static/uploads"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

# Gemini API key only for chatbot
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY", "")

# Thresholds
INVALID_THRESHOLD = 0.95
PNEUMONIA_THRESHOLD = 0.65
UNCERTAIN_MARGIN = 0.15

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ─── GEMINI CLIENT ────────────────────────────────────────
gemini_client = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("[OK] Gemini client created for chatbot.")
    except Exception as e:
        print("[WARN] Gemini client error:", e)
else:
    print("[INFO] Gemini API key not configured. Chatbot will not work.")


# ─── FLASK APP ────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
CORS(app)


# ─── LOAD MODEL ───────────────────────────────────────────
def load_pneumonia_model():
    try:
        if not os.path.exists(MODEL_PATH):
            print(f"[WARN] Model not found at {MODEL_PATH}")
            return None

        loaded_model = tf.keras.models.load_model(MODEL_PATH)
        print("[OK] Pneumonia 3-class model loaded successfully!")
        return loaded_model

    except Exception as e:
        print(f"[ERR] Error loading model: {e}")
        return None


model = load_pneumonia_model()


# ─── HELPERS ──────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(pil_image):
    """
    Preprocess full image for prediction.
    Do not crop here because model was trained on full X-ray images.
    ROI is only for display when pneumonia is detected.
    """
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))

    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)


def save_roi_image(pil_image, original_filename):
    """
    Saves ROI image with a visible red box using PIL only.
    No cv2 / OpenCV needed.
    ROI is shown only when Pneumonia is detected.
    This is NOT medical lung segmentation.
    """

    from PIL import ImageDraw, ImageFont

    img = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    w, h = img.size

    # ROI box around central lung/chest area
    x1 = int(w * 0.12)
    y1 = int(h * 0.15)
    x2 = int(w * 0.88)
    y2 = int(h * 0.82)

    thickness = max(5, int(min(w, h) * 0.012))

    # Draw thick red rectangle
    for i in range(thickness):
        draw.rectangle(
            [x1 - i, y1 - i, x2 + i, y2 + i],
            outline=(255, 0, 0)
        )

    label = "Suspected Pneumonia ROI"

    try:
        font = ImageFont.truetype("arial.ttf", max(18, int(w * 0.035)))
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(label, font=font)

    label_x = x1
    label_y = max(5, y1 - text_h - 16)

    # Red label background
    draw.rectangle(
        [label_x, label_y, label_x + text_w + 18, label_y + text_h + 12],
        fill=(255, 0, 0)
    )

    # White label text
    draw.text(
        (label_x + 9, label_y + 5),
        label,
        fill=(255, 255, 255),
        font=font
    )

    # Save as PNG
    base_name = os.path.splitext(original_filename)[0]
    roi_filename = "roi_" + base_name + ".png"

    roi_path = os.path.join(UPLOAD_DIR, roi_filename)
    img.save(roi_path)

    print("[ROI SAVED]", roi_path)

    # Direct path for frontend
    return f"/static/uploads/{roi_filename}"

def basic_xray_validation(pil_image):
    """
    Basic rule-based validation to reject obvious non-X-ray images.
    """

    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    rg_diff = np.mean(np.abs(r - g))
    rb_diff = np.mean(np.abs(r - b))
    gb_diff = np.mean(np.abs(g - b))

    color_score = (rg_diff + rb_diff + gb_diff) / 3

    gray = np.mean(arr, axis=2)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    binary_dark = np.mean(gray < 50)
    binary_light = np.mean(gray > 205)

    print("Validation check:", {
        "color_score": float(color_score),
        "brightness": float(brightness),
        "contrast": float(contrast),
        "binary_dark": float(binary_dark),
        "binary_light": float(binary_light)
    })

    # Reject colorful images
    if color_score > 25:
        return False

    # Reject blank images
    if contrast < 18:
        return False

    # Reject extremely dark or bright images
    if brightness < 15 or brightness > 240:
        return False

    # Reject QR/barcode/document-like images
    if binary_dark > 0.20 and binary_light > 0.20 and contrast > 60:
        print("Rejected: QR/Barcode/Document")
        return False

    return True


# ─── PREDICTION ───────────────────────────────────────────
def predict_pneumonia(pil_img):
    """
    3-class prediction:
    INVALID IMAGES / NORMAL / PNEUMONIA

    Class order from your training:
    {'INVALID IMAGES': 0, 'NORMAL': 1, 'PNEUMONIA': 2}
    """

    if model is None:
        raise Exception("Model not loaded")

    img_array = preprocess(pil_img)
    preds = model.predict(img_array, verbose=0)[0]

    invalid_score = float(preds[0])
    normal_score = float(preds[1])
    pneumonia_score = float(preds[2])

    print("Predictions:", {
        "INVALID": invalid_score,
        "NORMAL": normal_score,
        "PNEUMONIA": pneumonia_score
    })

    # Thresholds
    INVALID_THRESHOLD_LOCAL = 0.90
    PNEUMONIA_THRESHOLD_LOCAL = 0.70
    NORMAL_THRESHOLD_LOCAL = 0.60
    MIN_DIFFERENCE = 0.10

    # 1. Invalid image
    if invalid_score >= INVALID_THRESHOLD_LOCAL:
        return {
            "prediction": "INVALID",
            "confidence": round(invalid_score * 100, 2),
            "raw_score": round(pneumonia_score, 4),
            "threshold": PNEUMONIA_THRESHOLD_LOCAL,
            "scores": preds
        }

    # 2. Pneumonia only if pneumonia score is clearly high
    if (
        pneumonia_score >= PNEUMONIA_THRESHOLD_LOCAL
        and pneumonia_score > normal_score
        and (pneumonia_score - normal_score) >= MIN_DIFFERENCE
    ):
        return {
            "prediction": "Pneumonia",
            "confidence": round(pneumonia_score * 100, 2),
            "raw_score": round(pneumonia_score, 4),
            "threshold": PNEUMONIA_THRESHOLD_LOCAL,
            "scores": preds
        }
    # 3. Normal if normal score is clearly high
    if (
        normal_score >= NORMAL_THRESHOLD_LOCAL
        and normal_score > pneumonia_score
        and (normal_score - pneumonia_score) >= MIN_DIFFERENCE
    ):
        return {
            "prediction": "Normal",
            "confidence": round(normal_score * 100, 2),
            "raw_score": round(pneumonia_score, 4),
            "threshold": PNEUMONIA_THRESHOLD_LOCAL,
            "scores": preds
        }

    # 4. Handle uncertain predictions
    best_score = max(invalid_score, normal_score, pneumonia_score)

    # If model is highly confused, mark as uncertain
    if best_score < 0.50:
        return {
        "prediction": "Uncertain",
        "confidence": round(best_score * 100, 2),
        "raw_score": round(pneumonia_score, 4),
        "threshold": PNEUMONIA_THRESHOLD_LOCAL,
        "scores": preds
    }

    # Otherwise return uncertain
    return {
        "prediction": "Uncertain",
        "confidence": round(best_score * 100, 2),
        "raw_score": round(pneumonia_score, 4),
        "threshold": PNEUMONIA_THRESHOLD_LOCAL,
        "scores": preds
    }

# ─── GEMINI CHATBOT ───────────────────────────────────────
def chat_with_gemini(text, image=None):
    """
    Gemini is used only for chatbot.
    It answers medical-awareness and pneumonia-project related questions.
    It is not used for prediction or validation.
    """
    try:
        if gemini_client is None:
            return "Gemini API key not configured."

        user_text = text.strip() if text else ""

        if not user_text and image is None:
            return "Please ask a medical awareness or pneumonia-related question."

        prompt = f"""
You are a medical awareness chatbot for a Pneumonia Detection System.

Answer medical awareness, pneumonia, chest X-ray, model confidence, invalid image, and ROI explanation questions.
Keep answers short and simple.
Do not diagnose or prescribe medicines.
For serious symptoms, advise consulting a doctor.
If unrelated, reply: "I can help only with medical awareness and pneumonia detection related questions."

User question: {user_text}
"""

        contents = [prompt]

        if image:
            contents.append(image)

        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents
        )

        return response.text.strip()

    except Exception as e:
        error_msg = str(e)
        print(f"[WARN] Gemini chat error: {error_msg}")

        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
            return "Gemini chatbot is temporarily unavailable due to API quota limit. Please try again after some time."

        return "Sorry, I could not process your question right now. Please try again."


# ─── ROUTES ───────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction route:
    - accepts image
    - validates obvious non-X-ray images
    - predicts INVALID / Normal / Pneumonia / Uncertain
    - shows ROI only for Pneumonia
    """
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please train or add the model file."
            }), 500

        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No selected file"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type. Use JPG, JPEG, PNG, BMP, or WEBP."
            }), 400

        filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_DIR, filename)
        file.save(file_path)

        pil_img = PILImage.open(file_path).convert("RGB")

        # Basic validation before model prediction
        if not basic_xray_validation(pil_img):
            try:
                os.remove(file_path)
            except Exception:
                pass

            return jsonify({
                "error": "Invalid image. Please upload a clear chest X-ray. Ultrasound, CT, MRI, QR codes and non-X-ray images are not supported."
            }), 400

        result_data = predict_pneumonia(pil_img)

        prediction = result_data["prediction"]
        confidence = result_data["confidence"]
        raw_score = result_data["raw_score"]
        threshold = result_data["threshold"]
        raw_pred = result_data["scores"]

        # If model strongly says invalid
        if prediction == "INVALID":
            try:
                os.remove(file_path)
            except Exception:
                pass

            return jsonify({
                "error": "Invalid image. Please upload a clear chest X-ray.",
                "scores": {
                    "invalid": round(float(raw_pred[0]) * 100, 2),
                    "normal": round(float(raw_pred[1]) * 100, 2),
                    "pneumonia": round(float(raw_pred[2]) * 100, 2)
                }
            }), 400

        response = {
            "prediction": prediction,
            "confidence": confidence,
            "raw_score": raw_score,
            "threshold": threshold,
            "scores": {
                "invalid": round(float(raw_pred[0]) * 100, 2),
                "normal": round(float(raw_pred[1]) * 100, 2),
                "pneumonia": round(float(raw_pred[2]) * 100, 2)
            },
            "original_image": f"uploads/{filename}",
            "roi_image": None,
            "show_roi": False
        }

        # ROI should be shown only for Pneumonia
        if prediction == "Pneumonia":
            roi_image_path = save_roi_image(pil_img, filename)
            response["roi_image"] = roi_image_path
            response["show_roi"] = True

        # Normal and Uncertain will not show ROI
        return jsonify(response)

    except Exception as e:
        print("[ERR] Prediction error:", e)
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/gemini-chat", methods=["POST"])
def gemini_chat():
    """
    Gemini chatbot route.
    Gemini is not used for prediction.
    """
    try:
        question = request.form.get("question", "")
        image_file = request.files.get("file")

        img = None
        if image_file:
            img = PILImage.open(image_file).convert("RGB")

        reply = chat_with_gemini(question, img)

        return jsonify({
            "reply": reply
        })

    except Exception as e:
        return jsonify({
            "reply": f"Error: {str(e)}"
        }), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/health")
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_usage": "chatbot_only",
        "explainability": "ROI shown only for Pneumonia",
        "invalid_threshold": INVALID_THRESHOLD,
        "pneumonia_threshold": PNEUMONIA_THRESHOLD,
        "uncertain_margin": UNCERTAIN_MARGIN
    })


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ─── MAIN ─────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "production") == "development"

    print(f"[START] Pneumonia Detection App on port {port}")
    print(f"[INFO] Model path: {MODEL_PATH}")
    print(f"[INFO] Model loaded: {model is not None}")
    print(f"[INFO] Gemini configured for chatbot: {bool(GEMINI_API_KEY)}")
    print("[INFO] Gemini is NOT used for prediction or validation.")
    print("[INFO] ROI is shown only when Pneumonia is detected.")
    print(f"[INFO] Pneumonia threshold: {PNEUMONIA_THRESHOLD}")
    print(f"[INFO] Invalid threshold: {INVALID_THRESHOLD}")

    app.run(host="0.0.0.0", port=port, debug=debug)