"""
Real-Time Face Mask Detector - Streamlit + WebRTC version
-----------------------------------------------------------
Phone aur desktop dono ke browser se camera access karta hai.
Local run: streamlit run app.py
Deploy: Streamlit Community Cloud (share.streamlit.io) par free hosted ho sakta hai.
"""

import os
import urllib.request

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# ------------------- Config -------------------
MODEL_PATH = "mask_detector_model.keras"
IMAGE_SIZE = (128, 128)

CLASS_INFO = [
    ("With Mask",    (0, 200, 0)),    # BGR -> green
    ("Without Mask", (0, 0, 255)),    # BGR -> red
]
UNSURE_COLOR = (0, 255, 255)
CONFIDENCE_THRESHOLD = 0.60
FACE_DETECT_CONFIDENCE = 0.5   # DNN face detector ka apna confidence threshold

# OpenCV's standard DNN face detector files (small, ~2.7MB caffemodel)
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
PROTOTXT_PATH = "deploy.prototxt"
FACE_MODEL_PATH = "res10_300x300_ssd_iter_140000_fp16.caffemodel"


def _download_if_missing(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)


# ------------------- Load model once (cached) -------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def load_face_detector():
    _download_if_missing(PROTOTXT_URL, PROTOTXT_PATH)
    _download_if_missing(MODEL_URL, FACE_MODEL_PATH)
    return cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, FACE_MODEL_PATH)


model = load_model()
face_net = load_face_detector()


def detect_faces_dnn(img):
    """Returns list of (x, y, w, h) boxes. Robust to occlusion (masks, angles)."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < FACE_DETECT_CONFIDENCE:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


# ------------------- Frame processor -------------------
class MaskDetectionProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        faces = detect_faces_dnn(img)

        for (x, y, w, h) in faces:
            face_img = img[y:y + h, x:x + w]
            if face_img.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, IMAGE_SIZE)
            face_array = np.expand_dims(face_resized.astype("float32"), axis=0)

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


# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")
st.title("😷 Real-Time Face Mask Detector")
st.write("Camera access allow karo aur apna face frame ke saamne rakho.")

webrtc_streamer(
    key="mask-detector",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=MaskDetectionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

st.caption("Note: Pehli baar browser camera permission maangega — Allow dabao.")
