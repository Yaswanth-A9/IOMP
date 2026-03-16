from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load smallest YOLO model
model = YOLO("yolov8n.pt")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("index"))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    frame = cv2.imread(file_path)

    # 🔥 Reduce image size (important for memory)
    frame = cv2.resize(frame, (640, 480))

    # 🔥 Lower image size + higher confidence (less processing)
    results = model(frame, conf=0.4, imgsz=320, verbose=False)

    annotated = results[0].plot()

    output_path = os.path.join(STATIC_FOLDER, "result.jpg")
    cv2.imwrite(output_path, annotated)

    return render_template("index.html", result=True)

if __name__ == "__main__":
    app.run()