import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from flask import Flask, Response, render_template
from flask_sock import Sock
import json
import threading
import time

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 224
CLASS_NAMES = ['cardboard', 'plastic', 'metal', 'glass']

CONF_THRESHOLD = 0.6
HISTORY_SIZE = 10

# -----------------------------
# LOAD MODEL
# -----------------------------
interpreter = tf.lite.Interpreter(model_path="waste_classifier.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# TEMPORAL SMOOTHING
# -----------------------------
history = deque(maxlen=HISTORY_SIZE)

# -----------------------------
# FLASK SETUP
# -----------------------------
app = Flask(__name__)
sock = Sock(app)

latest_frame = None
latest_result = {"label": "Detecting...", "confidence": 0.0}
frame_lock = threading.Lock()

# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(img):
    img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
    img = cv2.resize(img, (256, 256))
    img = img[16:240, 16:240]  # center crop
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# CLASSIFY
# -----------------------------
def classify(img):
    input_data = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]['index'])[0]

    # Bias (optional tuning)
    bias = np.array([0.0, 1.0, 1.0, 1.0])
    probs = probs * bias
    probs = probs / probs.sum()

    class_id = np.argmax(probs)
    confidence = float(probs[class_id])

    return CLASS_NAMES[class_id], confidence

# -----------------------------
# FORCE 4:3 ASPECT RATIO
# -----------------------------
def to_laptop_aspect(frame):
    h, w, _ = frame.shape
    target_ratio = 4 / 3
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = (w - new_w) // 2
        frame = frame[:, x1:x1+new_w]
    else:
        new_h = int(w / target_ratio)
        y1 = (h - new_h) // 2
        frame = frame[y1:y1+new_h, :]

    return frame

# -----------------------------
# CAMERA CAPTURE LOOP (runs in background thread)
# -----------------------------
def capture_loop():
    global latest_frame, latest_result

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_count = 0
    conf = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = to_laptop_aspect(frame)
        h, w, _ = frame.shape

        # -----------------------------
        # CENTER SQUARE ROI
        # -----------------------------
        box_size = int(min(h, w) * 0.5)
        x1 = (w - box_size) // 2
        y1 = (h - box_size) // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        roi_full = frame[y1:y2, x1:x2]
        h2, w2, _ = roi_full.shape
        roi = roi_full[int(h2*0.1):int(h2*0.9), int(w2*0.1):int(w2*0.9)]

        # -----------------------------
        # FRAME SKIPPING + CLASSIFY
        # -----------------------------
        frame_count += 1
        if frame_count % 3 == 0:
            pred, conf = classify(roi)
            if conf > CONF_THRESHOLD:
                history.append(pred)

        final_label = max(set(history), key=history.count) if history else "Detecting..."
        latest_result = {"label": final_label, "confidence": round(conf, 2)}

        # -----------------------------
        # DRAW OVERLAY ON FRAME
        # -----------------------------
        COLORS = {
            'cardboard': (139, 69, 19),
            'plastic':   (0, 215, 255),
            'metal':     (180, 180, 180),
            'glass':     (255, 200, 0)
        }
        color = COLORS.get(final_label, (0, 255, 0))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        corner_len = 25
        thickness = 4
        cv2.line(frame, (x1, y1), (x1+corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1+corner_len), color, thickness)
        cv2.line(frame, (x2, y1), (x2-corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1+corner_len), color, thickness)
        cv2.line(frame, (x1, y2), (x1+corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2-corner_len), color, thickness)
        cv2.line(frame, (x2, y2), (x2-corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2-corner_len), color, thickness)

        cv2.putText(frame, "AI Waste Classifier",
                    (int(w*0.05), int(h*0.08)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.putText(frame, "Place object in center box",
                    (int(w*0.05), int(h*0.14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 2)

        if conf < CONF_THRESHOLD:
            display_text = "Analyzing..."
        else:
            display_text = f"{final_label.upper()}  {conf:.2f}"

        (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        padding = 8
        badge_x1 = x1
        badge_y1 = y1 - text_h - 15
        badge_x2 = x1 + text_w + padding*2
        badge_y2 = y1 - 5
        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), (0,0,0), -1)
        cv2.putText(frame, display_text,
                    (badge_x1 + padding, badge_y2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        bar_x1 = int(w*0.1)
        bar_x2 = int(w*0.9)
        bar_y  = int(h*0.92)
        bar_width = bar_x2 - bar_x1
        filled = int(bar_width * conf)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y+20), (50,50,50), -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1+filled, bar_y+20), color, -1)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y+20), (255,255,255), 2)

        # -----------------------------
        # ENCODE FRAME AS JPEG FOR STREAM
        # -----------------------------
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()

# -----------------------------
# MJPEG GENERATOR
# -----------------------------
def generate_mjpeg():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.05)

# -----------------------------
# FLASK ROUTES
# -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@sock.route('/ws')
def predictions(ws):
    while True:
        ws.send(json.dumps(latest_result))
        time.sleep(0.1)

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == '__main__':
    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True)