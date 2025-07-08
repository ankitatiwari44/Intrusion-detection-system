import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

detector = MTCNN()
embedder = FaceNet()

def extract_face(img, required_size=(160, 160)):
    results = detector.detect_faces(img)
    if results:
        x1, y1, width, height = results[0]['box']
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = x1 + width, y1 + height
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, required_size)
        face = face.astype('float32') / 255.0
        return face
    return None

def get_embedding(face_pixels):
    return embedder.embeddings([face_pixels])[0]

def is_match(known_embedding, candidate_embedding, threshold=0.6):
    distance = np.linalg.norm(known_embedding - candidate_embedding)
    return distance < threshold
