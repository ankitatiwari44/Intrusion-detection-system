import os
import cv2
import numpy as np
import threading
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from insightface.app import FaceAnalysis
from playsound import playsound

# Initialize models
embedder = FaceNet()
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# Load known face embeddings
path = "known_faces"
known_embeddings = []
classNames = []

for filename in os.listdir(path):
    img_path = os.path.join(path, filename)
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    person_name = os.path.splitext(filename)[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Cannot read image: {filename}")
        continue

    faces = face_app.get(img)
    if not faces:
        print(f"[!] No face detected in {filename}")
        continue

    face_crop = faces[0].crop_bgr
    if face_crop is None or face_crop.size == 0:
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        face_crop = img[y1:y2, x1:x2]

    if face_crop is None or face_crop.size == 0:
        print(f"[!] Failed to crop {filename}")
        continue

    resized = cv2.resize(face_crop, (160, 160))
    embedding = embedder.embeddings([resized])[0]
    known_embeddings.append(embedding)
    classNames.append(person_name.lower())
    print(f"[+] Encoded: {filename}")

known_embeddings = normalize(np.array(known_embeddings))
print("Loaded embeddings:", len(known_embeddings))

# Recognition threshold
THRESHOLD = 0.5
siren_played = False  # Global flag for one-time siren

def play_siren_once():
    try:
        playsound("siren.mp3")  # Ensure this file exists in the same directory
    except Exception as e:
        print("Failed to play siren:", e)

def recognize_faces(frame: np.ndarray) -> np.ndarray:
    global siren_played

    faces = face_app.get(frame)
    intruder_detected = False

    for face in faces:
        face_crop = face.crop_bgr
        if face_crop is None or face_crop.size == 0:
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_crop = frame[y1:y2, x1:x2]

        if face_crop is None or face_crop.size == 0:
            continue

        try:
            resized = cv2.resize(face_crop, (160, 160))
            embedding = embedder.embeddings([resized])[0]
            embedding = normalize(embedding.reshape(1, -1))[0]
        except:
            continue

        if not known_embeddings.any():
            continue

        sims = cosine_similarity(embedding.reshape(1, -1), known_embeddings)[0]
        best_idx = np.argmax(sims)
        score = sims[best_idx]

        name = classNames[best_idx] if score > THRESHOLD else "Intruder"

        if name == "Intruder":
            intruder_detected = True

        # Draw label and box
        x1, y1, x2, y2 = face.bbox.astype(int)
        color = (0, 0, 255) if name == "Intruder" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 🔊 Play siren only once when intruder detected
    if intruder_detected and not siren_played:
        threading.Thread(target=play_siren_once, daemon=True).start()
        siren_played = True
    elif not intruder_detected:
        siren_played = False

    return frame
