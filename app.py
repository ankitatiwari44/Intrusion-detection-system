import streamlit as st
import cv2
from av import VideoFrame
import numpy as np
from PIL import Image
from face_utils import recognize_faces
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("🎯 Face Recognition System with Intruder Alert")

# Option selector
option = st.radio("Choose input mode:", ["📷 Live Webcam", "🖼️ Upload Photo"])

# Webcam recognition class
from av import VideoFrame  # required

class FaceRecognitionTransformer(VideoTransformerBase):
    def recv(self, frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        output = recognize_faces(img)
        return VideoFrame.from_ndarray(output, format="bgr24")


# Webcam mode
if option == "📷 Live Webcam":
    st.markdown("#### Live Detection (press 'Stop' to end)")
    webrtc_streamer(
        key="face-detection",
        video_processor_factory=FaceRecognitionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Upload photo mode
# Upload photo mode
elif option == "🖼️ Upload Photo":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        frame = np.array(image)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        result = recognize_faces(frame_bgr)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        st.image(result_rgb, caption="Detected Faces", use_container_width=True)  # ✅ Updated here

