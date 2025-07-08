#  Intrusion Detection System in Restricted Areas

This project is a real-time **Intrusion Detection System** built using Deep Learning and Computer Vision techniques. It uses **YOLOv8** for detecting humans in video frames, and **FaceNet** with **MTCNN** for recognizing known individuals. If an unknown person (intruder) is detected, a **siren sound is played**, and the image of the intruder is saved for record-keeping.

---

##  Technologies Used

- **YOLOv8** – for person detection in real-time
- **FaceNet** – for generating facial embeddings
- **MTCNN** – for face detection before recognition
- **Streamlit** – to create an interactive web app interface
- **OpenCV** – for video processing and image handling
- **Pygame** – to play siren sound when an intruder is detected

---

## Features

- Detects people in **real-time webcam** or **video file**
- Identifies **known faces** from a folder
- Labels unknown individuals as **“Intruder”**
- Plays a **siren alert** when intruder is detected
- Saves **intruder face images** in a folder
- Streamlit-based user interface for ease of use

---

## Folder Structure


Intrusion-Detection-System/
├── app.py # Main Streamlit app
├── known_faces/ # Folder with known person images
├── saved_intruders/ # Stores captured intruder images
├── models/
│ ├── yolov8.pt # YOLOv8 model weights
│ └── facenet_model.h5 # Pretrained FaceNet model
├── siren.mp3 # Siren audio file
├── utils.py # Supporting functions
├── requirements.txt # Python dependencies
├── .gitignore # Files/folders to ignore in Git
└── README.md # Project documentation

YOLOv8 detects all persons in each video frame.

Detected faces are cropped and passed to FaceNet.

FaceNet generates embeddings, which are compared with saved known faces.

If the similarity score is above a threshold, the person is recognized.

If not recognized:

The person is labeled as “Intruder”

A siren is played

The image is saved in the saved_intruders/ folder
