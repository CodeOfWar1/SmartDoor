import numpy as np
import os
import cv2
import base64
import pickle
from pymongo import MongoClient
from datetime import datetime
from insightface.app import FaceAnalysis

# === MongoDB setup ===
client = MongoClient("mongodb://admin:tembo123@10.127.191.149:27017/admin")
db = client["face_access"]
users_col = db["registered_users"]

# === Initialize InsightFace ===
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0)

def encode_face_insight(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = face_app.get(rgb)

    if not faces:
        raise ValueError("No face detected in the image.")

    return faces[0].embedding

def store_user(name, embedding):
    # Optional: normalize for consistency
    embedding = embedding / np.linalg.norm(embedding)

    pickled = pickle.dumps(embedding)
    encoded = base64.b64encode(pickled).decode('utf-8')

    doc = {
        "name": name,
        "encoding": encoded,
        "timestamp": datetime.utcnow()
    }

    users_col.insert_one(doc)
    print(f"[SUCCESS] Registered user: {name}")

def main():
    image_path = input("Enter path to image file: ").strip()
    name = input("Enter user name: ").strip()

    if not name or not image_path:
        print("❌ Name and image path are required.")
        return

    try:
        embedding = encode_face_insight(image_path)
        store_user(name, embedding)
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()
