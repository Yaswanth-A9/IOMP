from flask import Flask, render_template, request, redirect, url_for
from ultralytics import YOLO
import cv2
import os
import uuid

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ================= LOAD MODEL ONLY ONCE =================
model = YOLO("yolov8n.pt")   # small model (best for free tier)

# ========================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("home"))

    # Save uploaded file
    unique_name = str(uuid.uuid4()) + ".jpg"
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    # Read image
    frame = cv2.imread(filepath)

    # Run YOLO on CPU
    results = model(frame, conf=0.25, device="cpu")

    # Get detections
    boxes = results[0].boxes
    count = len(boxes)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save result image
    output_path = os.path.join(UPLOAD_FOLDER, "result_" + unique_name)
    cv2.imwrite(output_path, annotated_frame)

    return render_template(
        "index.html",
        input_image=filepath,
        output_image=output_path,
        count=count
    )


if __name__ == "__main__":
    app.run(debug=True)