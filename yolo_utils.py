import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

def detect_people_yolo(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == 0 and confidence > 0.5:  # classID 0 = person
                box = detection[0:4] * np.array([w, h, w, h])
                centerX, centerY, bw, bh = box.astype("int")
                x = int(centerX - bw / 2)
                y = int(centerY - bh / 2)
                boxes.append((x, y, x + int(bw), y + int(bh)))
    return boxes
