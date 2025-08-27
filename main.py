import os
import time
import threading
import base64
import pickle
from datetime import datetime
from flask import Flask, Response
from pymongo import MongoClient
import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import serial
from insightface.app import FaceAnalysis

# === GPIO Setup ===
RELAY_PIN = 17
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.HIGH)

# === MongoDB Setup ===
client = MongoClient("mongodb://admin:tembo123@172.27.243.149:27017/admin")
db = client["face_access"]
users_collection = db["registered_users"]
logs_collection = db["access_logs"]

# === Serial Setup ===
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
arduino_connected = False
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    arduino_connected = True
except serial.SerialException as e:
    print(f"[WARNING] Could not connect to Arduino: {e}")

# === Load Known Encodings ===
known_names = []
known_embeddings = []
print("[INFO] Loading face encodings from MongoDB...")
for doc in users_collection.find():
    try:
        encoding = pickle.loads(base64.b64decode(doc["encoding"]))
        arr = np.array(encoding)
        if arr.shape == (512,):  # Ensure proper shape
            known_embeddings.append(arr)
            known_names.append(doc["name"])
            print(f"[OK] Loaded: {doc['name']}")
        else:
            print(f"[SKIP] Invalid embedding shape for {doc['name']}: {arr.shape}")
    except Exception as e:
        print(f"[ERROR] Failed to load {doc.get('name', 'Unknown')}: {e}")

# Convert to array if not empty
if known_embeddings:
    known_embeddings = np.array(known_embeddings)
    print(f"[INFO] Loaded {len(known_embeddings)} known embeddings with shape {known_embeddings.shape}")
else:
    known_embeddings = np.empty((0, 512))
    print("[WARNING] No valid face embeddings loaded.")

# === Camera Setup ===
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()

# === InsightFace Setup ===
print("[INFO] Initializing InsightFace...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

# === Relay Control ===
def trigger_unlock():
    GPIO.output(RELAY_PIN, GPIO.LOW)
    time.sleep(7)
    GPIO.output(RELAY_PIN, GPIO.HIGH)

def send_access_granted():
    if arduino_connected:
        try:
            ser.write(b"ACCESS_GRANTED\n")
        except:
            pass

def send_access_denied():
    if arduino_connected:
        try:
            ser.write(b"ACCESS_DENIED\n")
        except:
            pass

# === Background Detection ===
latest_frame = None
def detection_loop():
    global latest_frame
    while True:
        frame = picam2.capture_array()
        latest_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if faces:
            print(f"[{timestamp}] {len(faces)} face(s) detected.")
            for face in faces:
                embedding = face.embedding
                name = "Unknown"
                matched = False

                if known_embeddings.size > 0:
                    print(f"[DEBUG] Embedding shape: {embedding.shape}")
                    print(f"[DEBUG] Known embeddings shape: {known_embeddings.shape}")

                    try:
                        # Normalize live embedding before distance calculation
                        norm_embedding = embedding / np.linalg.norm(embedding)

                        distances = np.linalg.norm(known_embeddings - norm_embedding, axis=1)
                        best_idx = np.argmin(distances)
                        print(f"[DEBUG] Distances: {distances}")
                        print(f"[DEBUG] Closest match: {distances[best_idx]:.4f} ({known_names[best_idx]})")

                        if distances[best_idx] < 1.2:
                            name = known_names[best_idx]
                            matched = True
                            print(f"[{timestamp}] ✅ Matched: {name}")
                            send_access_granted()
                            trigger_unlock()
                        else:
                            print(f"[{timestamp}] ❌ No match found (distance too high).")
                            send_access_denied()
                    except Exception as e:
                        print(f"[ERROR] Distance computation failed: {e}")
                        send_access_denied()
                else:
                    print(f"[{timestamp}] ⚠️ No known embeddings to compare.")
                    send_access_denied()

                # Draw rectangle and label
                x1, y1, x2, y2 = face.bbox.astype(int)
                color = (0, 255, 0) if matched else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Encode detected face image as JPEG in memory (no file saving)
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                # Log to MongoDB with image stored as binary
                logs_collection.insert_one({
                    "timestamp": datetime.now(),
                    "status": "granted" if matched else "denied",
                    "name": name,
                    "image": img_bytes  # Store raw image bytes in DB
                })
        else:
            print(f"[{timestamp}] No face detected.")
        time.sleep(0.3)

# Start detection thread
threading.Thread(target=detection_loop, daemon=True).start()

# === Flask Live Stream ===
app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>Live Video Feed</h1><img src="/video">'

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    global latest_frame
    while True:
        if latest_frame is not None:
            _, buffer = cv2.imencode('.jpg', latest_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.1)

# === Run App ===
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8000, debug=False)
    finally:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        GPIO.cleanup()
        picam2.close()
        if arduino_connected:
            ser.close()
