import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request, redirect
from PIL import Image
from werkzeug.utils import secure_filename

import CNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "plant_disease_model_1_latest.pt")
ALLOWED = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

disease_info = pd.read_csv(os.path.join(BASE_DIR, "disease_info.csv"), encoding="cp1252")
supplement_info = pd.read_csv(os.path.join(BASE_DIR, "supplement_info1.csv"), encoding="cp1252")

model = CNN.CNN(39)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def prediction(image_path):
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    data = TF.to_tensor(image).view(-1, 3, 224, 224)
    with torch.no_grad():
        output = model(data)
    return int(np.argmax(output.numpy()))


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/index")
def ai_engine_page():
    return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method != "POST":
        return redirect("/index")

    image = request.files.get("image")
    if not image or not image.filename:
        return redirect("/index")

    name = secure_filename(image.filename)
    if name.rsplit(".", 1)[-1].lower() not in ALLOWED:
        return "Please upload a PNG, JPG or WEBP image.", 400

    file_path = os.path.join(UPLOAD_DIR, name)
    image.save(file_path)
    pred = prediction(file_path)

    return render_template(
        "submit.html",
        title=disease_info["disease_name"][pred],
        desc=disease_info["description"][pred],
        prevent=disease_info["Possible Steps"][pred],
        image_url=disease_info["image_url"][pred],
        pred=pred,
        sname=supplement_info["supplement name"][pred],
        simage=supplement_info["supplement image"][pred],
        buy_link=supplement_info["buy link"][pred],
    )


@app.route("/market")
def market():
    return render_template(
        "market.html",
        supplement_image=list(supplement_info["supplement image"]),
        supplement_name=list(supplement_info["supplement name"]),
        disease=list(disease_info["disease_name"]),
        buy=list(supplement_info["buy link"]),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
