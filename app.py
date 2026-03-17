from flask import Flask, render_template, request, Response, redirect, url_for, session
import cv2
import os
from ultralytics import YOLO
from collections import Counter

app = Flask(__name__)
app.secret_key = "yolo_secret_key"

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Load YOLOv8 nano model
model = YOLO("yolov8n.pt")

video_path = None

# ================= HOME =================
@app.route("/")
def index():
    return render_template(
        "index.html",
        image_result=session.pop("image_result", False),
        video_result=session.pop("video_result", False),
        counts=session.pop("counts", None),
        total=session.pop("total", None)
    )

# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    global video_path

    # Clear old result image
    result_img = os.path.join(STATIC_FOLDER, "result.jpg")
    if os.path.exists(result_img):
        os.remove(result_img)

    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # ================= IMAGE =================
    if file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        frame = cv2.imread(file_path)
        results = model(frame, conf=0.15)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.tolist()
            names = model.names
            detected_classes = [names[int(i)] for i in class_ids]

            count_dict = dict(Counter(detected_classes))
            total_objects = len(class_ids)
        else:
            count_dict = {}
            total_objects = 0

        session["counts"] = count_dict
        session["total"] = total_objects
        session["image_result"] = True
        session["video_result"] = False

        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f"Total Objects: {total_objects}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        output_image = os.path.join(STATIC_FOLDER, "result.jpg")
        cv2.imwrite(output_image, annotated_frame)

        return redirect(url_for("index"))

    # ================= VIDEO =================
    elif file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        video_path = file_path
        session["image_result"] = False
        session["video_result"] = True
        return redirect(url_for("index"))

    return redirect(url_for("index"))

# ================= VIDEO STREAM =================
def generate():
    global video_path
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.15)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.tolist()
            names = model.names
            detected_classes = [names[int(i)] for i in class_ids]
            count_dict = dict(Counter(detected_classes))
            total_objects = len(class_ids)
        else:
            count_dict = {}
            total_objects = 0

        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f"Total Objects: {total_objects}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Overlay class-wise counts
        y_offset = 80
        for cls, count in count_dict.items():
            cv2.putText(
                annotated_frame,
                f"{cls}: {count}",
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )
            y_offset += 30

        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ================= RUN =================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Use Render's port
    app.run(host="0.0.0.0", port=port, debug=False)  # Expose externally for Render