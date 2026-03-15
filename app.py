from flask import Flask, render_template, request
import os
import cv2
from ultralytics import YOLO
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# ✅ FORCE CPU (Important for Render Free Plan)
device = "cpu"

# ✅ Load YOLOv8 model
model = YOLO("yolov8n.pt")
model.to(device)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run YOLO detection
    results = model(filepath)

    result = results[0]
    boxes = result.boxes

    # Count objects
    object_count = len(boxes)

    # Draw results
    annotated_frame = result.plot()

    # Save result image
    result_path = os.path.join(RESULT_FOLDER, file.filename)
    cv2.imwrite(result_path, annotated_frame)

    return render_template(
        "index.html",
        result_image=result_path,
        count=object_count
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)