import cv2
import numpy as np
from keras.models import load_model
import time

# Load trained model
model = load_model('emotion.h5')

# Emotion labels with colors
emotion_labels = [
    ('Angry', (0, 0, 255)),
    ('Disgust', (0, 102, 0)),
    ('Fear', (102, 102, 255)),
    ('Happy', (0, 255, 255)),
    ('Sad', (255, 0, 0)),
    ('Surprise', (255, 153, 255)),
    ('Neutral', (200, 200, 200))
]

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS tracking
prev_frame_time = 0

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # FPS calculation
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time + 1e-6))
    prev_frame_time = new_frame_time

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotion_counts = {}

    # Prepare emotion detection if faces found
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype("float") / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))
        preds = model.predict(roi, verbose=0)
        emotion_idx = np.argmax(preds)
        label, color = emotion_labels[emotion_idx]

        emotion_counts[label] = emotion_counts.get(label, 0) + 1

        # Draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 30), (x + w, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # ---------- Draw Stats Box ----------
    box_x, box_y = 20, 20
    line_height = 35
    base_height = 160
    extra_height = len(emotion_counts) * line_height
    box_height = base_height + extra_height
    box_width = 350

    # Box background and border
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (40, 40, 40), -1)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 200, 255), 2)

    # Title and subtitle
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + 40), (30, 30, 30), -1)
    cv2.putText(frame, "EMOTION DETECTION STATS", (box_x + 10, box_y + 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
    cv2.putText(frame, "Enhanced Emotion Detection", (box_x + 10, box_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Divider line
    cv2.line(frame, (box_x + 10, box_y + 70), (box_x + box_width - 10, box_y + 70), (0, 200, 255), 1)

    # FPS and face count
    cv2.putText(frame, f"FPS: {fps}", (box_x + 20, box_y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2)
    cv2.putText(frame, f"Faces Detected: {len(faces)}", (box_x + 20, box_y + 135), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 255, 100), 2)

    # Emotion breakdown inside the box
    for i, (emotion, count) in enumerate(emotion_counts.items()):
        color = next((col for lbl, col in emotion_labels if lbl == emotion), (255, 255, 255))
        cv2.putText(frame, f"{emotion}: {count}", 
                    (box_x + 20, box_y + 170 + i * line_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Show frame
    cv2.imshow('Enhanced Emotion Detection', frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
