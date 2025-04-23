from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import csv
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

MODEL_PATH = "defect_detection_cnn.h5"
model = tf.keras.models.load_model(MODEL_PATH)

UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CSV_FILE = "detections_log.csv"


def get_last_id():
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r") as file:
            reader = list(csv.reader(file))
            if len(reader) > 1:
                return int(reader[-1][0])
    return 0

detection_id = get_last_id()

camera = cv2.VideoCapture(0)

DETECTION_INTERVAL = 10
last_detection_time = time.time()


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (300, 300))
    img = np.expand_dims(img, axis=[0, -1]) / 255.0
    return img


def log_detection(status, confidence):
    global detection_id
    detection_id += 1
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["ID", "Status", "Confidence", "Timestamp"])

        writer.writerow([detection_id, status, f"{confidence:.2f}%", timestamp])



def generate_frames():
    global last_detection_time, DETECTION_INTERVAL

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            current_time = time.time()
            if current_time - last_detection_time >= DETECTION_INTERVAL:
                frame = detect_defects(frame)
                last_detection_time = current_time

            _, buffer = cv2.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/")
def index():
    return render_template("index.html", detection_interval=DETECTION_INTERVAL)

@app.route("/inspector")
def inspector():
    return render_template("inspector.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = any(500 < cv2.contourArea(cnt) < 500000 for cnt in contours)

    if not detected:
        print("No impeller detected! Skipping classification.")
        return jsonify({"message": "No impeller detected", "class": "None", "confidence": 0})

    img_resized = cv2.resize(image, (300, 300))
    img_processed = np.expand_dims(img_resized, axis=[0, -1]) / 255.0

    prediction = model.predict(img_processed)[0]
    confidence = float(prediction[0]) * 100

    result = {
        "class": "OK" if confidence > 80 else "Defective",
        "confidence": round(confidence, 2)
    }

    log_detection(result["class"], result["confidence"])

    return jsonify(result)


@app.route("/set_interval", methods=["POST"])
def set_interval():
    global DETECTION_INTERVAL
    new_interval = request.json.get("interval", 10)
    DETECTION_INTERVAL = max(1, int(new_interval))
    return jsonify({"message": "Interval updated", "new_interval": DETECTION_INTERVAL})

PREDICTION_ENABLED = True

@app.route("/toggle_prediction", methods=["POST"])
def toggle_prediction():
    global PREDICTION_ENABLED
    PREDICTION_ENABLED = request.json.get("enabled", True)
    return jsonify({"prediction_enabled": PREDICTION_ENABLED})

import cv2
import numpy as np

def detect_defects(frame):
    global PREDICTION_ENABLED, detection_id

    if not PREDICTION_ENABLED:
        cv2.putText(frame, "Prediction Stopped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5000 < area < 50000:
            detected = True
            break

    if not detected:
        print("No impeller detected! Skipping classification.")
        cv2.putText(frame, "No impeller detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame

    img_resized = cv2.resize(gray, (300, 300))
    img_processed = np.expand_dims(img_resized, axis=[0, -1]) / 255.0

    prediction = model.predict(img_processed)[0]
    confidence = float(prediction[0]) * 100

    label = "OK" if confidence > 80 else "Defective"
    color = (0, 255, 0) if confidence > 80 else (0, 0, 255)

    log_detection(label, confidence)

    cv2.putText(frame, f"{label} ({confidence:.2f}%)", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return frame


@app.route("/get_detections", methods=["GET"])
def get_detections():
    detections = []
    with open(CSV_FILE, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            detections.append(row)
    return jsonify(detections)



def variables():
        
    import pandas as pd

    if os.path.exists("detections_log.csv"):
        data = pd.read_csv("detections_log.csv")
    else:
        data = pd.DataFrame({"status": []})


    ok_count = int(data["status"].value_counts().get("OK", 0))
    defect_count = int(data["status"].value_counts().get("Defective", 0))

    data["date"] = pd.to_datetime(data["date"])

    data["hourly"] = data["date"].dt.floor("h")
    status_counts = data.groupby(["hourly", "status"]).size().unstack(fill_value=0)
    def_time = status_counts.get("Defective", pd.Series([])).tolist()
    ok_time = status_counts.get("OK", pd.Series([])).tolist()

    data["confidence"] = data["confidence"].astype(str).str.replace("%", "").astype(float)

    bins = [0, 10, 30, 50, 75, 100]
    old_labels = ["<10", "10-30", "30-50", "50-75", "75-100"]
    new_labels = ["Major Defect", "Minor Defect", "Cosmetic Defects", "Acceptable", "Excellent"]


    category_counts = pd.cut(data["confidence"], bins=bins, labels=old_labels, right=False).value_counts()

    category_counts.index = category_counts.index.map(dict(zip(old_labels, new_labels)))

    count_list = category_counts.reindex(new_labels).tolist()

    a=[]
    for index, row in data.iterrows():
        
        k=[]
        k.append(row.iloc[0])
        k.append(row.iloc[1])
        k.append(row.iloc[2])
        k.append(row.iloc[3])
        a.append(k)
    a= list(reversed(a))
    return ok_count,defect_count,status_counts,def_time,ok_time,new_labels,count_list,a 

@app.route("/history")
def history():
    ok_count,defect_count,status_counts,def_time,ok_time,new_labels,count_list,a = variables()
    return render_template("history.html",data=a)


@app.route("/dash")
def dashboard():
    ok_count,defect_count,status_counts,def_time,ok_time,new_labels,count_list,a = variables()
    Line_labels = status_counts.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
    
    return render_template(
        "dash.html",
        Labels_bar=["OK", "Defective"],
        Data_bar=[ok_count, defect_count],
        Line_labels=Line_labels,
        line_def=def_time,
        line_ok=ok_time,
        pie_labels=new_labels,
        pie_values=count_list,
    )
    

if __name__ == "__main__":
    app.run(debug=True)
