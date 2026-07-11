from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

from model import Inception

app = Flask(__name__)
app.secret_key = "medical_ai_secret"


UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = Inception(
    in_channels=3,
    use_auxiliary=True,
    num_classes=3
)

model.load_state_dict(
    torch.load(
        "googlenet_medical.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Class Names
class_names = [
    "COVID-19",
    "NORMAL",
    "PNEUMONIA"
]


@app.route("/")
def home():

    prediction = session.pop("prediction", None)
    confidence = session.pop("confidence", None)
    image_url = session.pop("image_url", None)
    filename = session.pop("filename", None)
    error = session.pop("error", None)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_url=image_url,
        filename=filename,
        error=error
    )

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        session["error"] = "No file uploaded. Please select an X-ray image."
        return redirect(url_for("home"))

    file = request.files["image"]

    if file.filename == "":
        session["error"] = "Please choose an image before submitting."
        return redirect(url_for("home"))

    # Clear old uploads so the app does not retain previous files.
    for existing_file in os.listdir(app.config["UPLOAD_FOLDER"]):
        existing_path = os.path.join(app.config["UPLOAD_FOLDER"], existing_file)
        if os.path.isfile(existing_path):
            try:
                os.remove(existing_path)
            except OSError:
                pass

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _, _ = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    disease = class_names[predicted.item()]
    confidence = round(confidence.item() * 100, 2)
    image_url = url_for("static", filename=f"uploads/{filename}")

    session["prediction"] = disease
    session["confidence"] = confidence
    session["image_url"] = image_url
    session["filename"] = filename

    return redirect(url_for("home"))


import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT",7860))
    )