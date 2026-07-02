from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import requests

# -----------------------------
# Flask App Setup
# -----------------------------
app = Flask(__name__)
app.secret_key = "plant_disease_secret"

# -----------------------------
# Login Credentials (Demo)
# -----------------------------
USERNAME = "admin"
PASSWORD = "admin123"

# -----------------------------
# Load TFLite Model
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# Load Class Names & Disease Info
# -----------------------------
with open("classes.json", "r") as f:
    class_names = json.load(f)

with open("disease_info.json", "r") as f:
    disease_info = json.load(f)

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# Telegram Bot Setup
# -----------------------------
TELEGRAM_TOKEN = "8439362949:AAGxBjY3i1xA921X_jQSHw7noO-PRk9qnnE"   # Replace with your bot token
CHAT_ID = "6350009893"  # Replace with your chat ID

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Error sending Telegram message:", e)

# -----------------------------
# Login Page
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        if username == USERNAME and password == PASSWORD:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            error = "Invalid username or password"
    return render_template("login.html", error=error)

# -----------------------------
# Home / Prediction Page
# -----------------------------
@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    info = {}
    confidence = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image = Image.open(file).convert("RGB")
            image = preprocess_image(image)

            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            predicted_index = np.argmax(output)
            prediction = class_names[predicted_index]
            confidence = round(float(np.max(output)) * 100, 2)

            info = disease_info.get(prediction, {})

            # -----------------------------
            # Send Telegram message if disease detected
            # -----------------------------
            healthy_classes = ["Healthy", "Tomato_healthy"]  # Add any other healthy class names here
            if prediction not in healthy_classes:
                message_lines = [
                    "🚨 Plant Disease Alert! 🚨",
                    f"Disease: {info.get('Name', prediction)}",
                    f"Confidence: {confidence}%"
                ]

                for key in ["Causes", "Prevention", "Treatment", "Plant Health"]:
                    if key in info:
                        message_lines.append(f"{key}: {info[key]}")

                message = "\n".join(message_lines)
                send_telegram_message(message)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        info=info
    )

# -----------------------------
# Logout
# -----------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
