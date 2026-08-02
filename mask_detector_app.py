"""
Real-Time Face Mask Detector
------------------------------
Loads the trained CNN model (mask_detector_model.keras) and uses the
webcam to detect faces in real time, classifying each as
"With Mask" / "Without Mask" and drawing a labeled bounding box.

Requirements:
    pip install opencv-python

Run:
    python mask_detector_app.py
Press 'q' to quit.
"""

import cv2
import numpy as np
import tensorflow as tf

# ------------------- 1. Path & Parameters -------------------
MODEL_PATH = "mask_detector_model.keras"   # apne model ka path yahan set karein
IMAGE_SIZE = (128, 128)                    # training wale size se match hona chahiye

# IMPORTANT: yeh order apne training dataset ke class_names se match karna hai.
# Notebook mein: print(raw_ds.class_names) chalao aur waisa hi order yahan set karo.
# Format: (display_text, box_color_in_BGR) -> list index = model's class index
CLASS_INFO = [
    ("With Mask",    (0, 200, 0)),   # class index 0 -> green box
    ("Without Mask", (0, 0, 255)),   # class index 1 -> red box
]

UNSURE_COLOR = (0, 255, 255)                # yellow
CONFIDENCE_THRESHOLD = 0.60                 # isse kam confidence par "Unsure" dikhega

# ------------------- 2. Load Model & Face Detector -------------------
print("Model load ho raha hai...")
model = tf.keras.models.load_model(MODEL_PATH)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------- 3. Webcam Loop -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam access nahi mila. Camera index (0) ya permissions check karo.")

print("Camera ON hai. Quit karne ke liye 'q' dabao.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read nahi hua, camera disconnect ho sakta hai.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        face_img = frame[y:y + h, x:x + w]

        # OpenCV BGR deta hai, training RGB images par hui thi -> conversion zaroori
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMAGE_SIZE)
        face_array = np.expand_dims(face_resized.astype("float32"), axis=0)  # (1,128,128,3)
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

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, text, (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
        )

    cv2.imshow("Face Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Camera band ho gaya.")