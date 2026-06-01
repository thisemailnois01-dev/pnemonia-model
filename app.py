import os, base64
import numpy as np
from PIL import Image

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Grad-CAM imports disabled for Railway speed
# import cv2
# from pytorch_grad_cam import GradCAMPlusPlus
# from pytorch_grad_cam.utils.image import show_cam_on_image

app = Flask(__name__)
CORS(app)

DEVICE = torch.device("cpu")
IMG_SIZE = 224


class PneumoniaNet(nn.Module):
    def __init__(self, model_name="tf_efficientnetv2_s", dropout=0.4):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0
        )

        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x.mean(dim=[-2, -1])
        return self.head(x)

    def get_cam_target_layer(self):
        return self.backbone.blocks[-1]


model = PneumoniaNet().to(DEVICE)

ckpt = torch.load("pneumonia_model.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Model loaded on CPU")

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])


def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    tensor = transform(image=img)["image"]
    return tensor


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "model": "PneumoniaNet-EfficientNetV2",
        "device": "cpu"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Empty file name"}), 400

        image = Image.open(file.stream)
        tensor = preprocess(image)

        with torch.no_grad():
            output = model(tensor.unsqueeze(0).to(DEVICE))
            prob = torch.sigmoid(output).item()

        prediction = "PNEUMONIA" if prob > 0.5 else "NORMAL"
        confidence = prob if prob > 0.5 else 1 - prob

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "probability_pneumonia": round(prob * 100, 2),
            "probability_normal": round((1 - prob) * 100, 2),
            "gradcam_image": None,
            "risk_level": "HIGH" if prob > 0.8 else "MEDIUM" if prob > 0.5 else "LOW"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
