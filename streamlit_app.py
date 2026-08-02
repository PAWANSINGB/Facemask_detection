"""
Real-Time Face Mask Detector - Web Version (Streamlit)
---------------------------------------------------------
Same detection logic as the desktop app, but uses streamlit-webrtc
so it can be deployed and accessed from any browser (camera runs
in the USER's browser, not on the server).
 
Local test:
    pip install streamlit streamlit-webrtc opencv-python-headless tensorflow-cpu av
    streamlit run streamlit_app.py
 
Deploy: push this folder to GitHub, then deploy on share.streamlit.io
"""
 
import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
 
# ------------------- 1. Path & Parameters -------------------
MODEL_PATH = "mask_detector_model.keras"     # apne model ka naam yahan match karo
IMAGE_SIZE = (128, 128)
 
# IMPORTANT: yeh order apne training dataset ke class_names se match karna hai.
# Notebook mein: print(raw_ds.class_names) chalao aur waisa hi order yahan set karo.
CLASS_INFO = [
    ("With Mask",    (0, 200, 0)),   # class index 0 -> green box
    ("Without Mask", (0, 0, 255)),   # class index 1 -> red box
]
UNSURE_COLOR = (0, 255, 255)
CONFIDENCE_THRESHOLD = 0.60
 
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
 
# ------------------- 2. Load Model & Face Detector (cached) -------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
 
 
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
 
 
model = load_model()
face_cascade = load_face_cascade()
 
 
# ------------------- 3. Frame Processor -------------------
class MaskDetector(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
 
        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
 
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, IMAGE_SIZE)
            face_array = np.expand_dims(face_resized.astype("float32"), axis=0)
            # NOTE: manually /255 mat karna, model ke andar Rescaling layer already hai
 
            preds = model(face_array, training=False).numpy()[0]
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx])
            display_text, box_color = CLASS_INFO[class_idx]
 
            if confidence < CONFIDENCE_THRESHOLD:
                text = f"Unsure ({confidence * 100:.0f}%)"
                color = UNSURE_COLOR
            else:
                text = f"{display_text}: {confidence * 100:.0f}%"
                color = box_color
 
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                img, text, (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )
 
        return av.VideoFrame.from_ndarray(img, format="bgr24")
 
 
# ------------------- 4. Streamlit UI -------------------
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")
st.title("😷 Real-Time Face Mask Detector")
st.caption("Camera access allow karo neeche 'START' dabane ke baad.")
 
webrtc_streamer(
    key="mask-detector",
    video_processor_factory=MaskDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)