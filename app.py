import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ── Load emotion model ────────────────────────────────────────────────────────
model = None
model_error = None

for loader in [
    lambda: __import__('tensorflow.keras.models', fromlist=['load_model']).load_model('emotion.h5', compile=False),
    lambda: __import__('keras.models',             fromlist=['load_model']).load_model('emotion.h5', compile=False),
    lambda: __import__('tensorflow', fromlist=['keras']).keras.models.load_model('emotion.h5', compile=False),
]:
    try:
        model = loader()
        print("Emotion model loaded successfully.")
        break
    except Exception as e:
        model_error = str(e)

if model is None:
    print(f"WARNING: Emotion model failed to load — {model_error}")

# ── Face detection: MediaPipe (preferred) → Haar fallback ────────────────────
USE_MEDIAPIPE = False
_mp_detector  = None

try:
    import mediapipe as mp
    _mp_detector  = mp.solutions.face_detection.FaceDetection(
        model_selection=0,          # 0 = short-range (≤2 m), best for webcam
        min_detection_confidence=0.3
    )
    USE_MEDIAPIPE = True
    print("MediaPipe face detection ready.")
except Exception as e:
    print(f"MediaPipe unavailable ({e}), falling back to Haar cascade.")

# Haar fallback (used only if MediaPipe is not available)
_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def detect_faces(frame):
    """Return list of (x, y, w, h) tuples for every detected face."""
    h, w = frame.shape[:2]

    if USE_MEDIAPIPE and _mp_detector is not None:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _mp_detector.process(rgb)
        boxes   = []
        if results.detections:
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x  = max(0, int(bb.xmin * w))
                y  = max(0, int(bb.ymin * h))
                bw = min(int(bb.width  * w), w - x)
                bh = min(int(bb.height * h), h - y)
                if bw > 20 and bh > 20:
                    boxes.append((x, y, bw, bh))
        return boxes

    # Haar fallback
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = _haar.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20)
    )
    return list(found) if len(found) else []


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status':       'ok' if model is not None else 'degraded',
        'model_loaded': model is not None,
        'model_error':  model_error,
        'face_detector': 'mediapipe' if USE_MEDIAPIPE else 'haar',
    })


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': f'Model not loaded: {model_error}', 'faces': []})

    data = request.get_json(silent=True)
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided', 'faces': []})

    try:
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr       = np.frombuffer(image_bytes, np.uint8)
        frame       = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Could not decode image', 'faces': []})

        faces   = detect_faces(frame)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float32') / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            preds = model.predict(roi, verbose=0)[0]
            idx   = int(np.argmax(preds))

            results.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'emotion':    EMOTION_LABELS[idx],
                'confidence': float(preds[idx]),
                'scores': {
                    EMOTION_LABELS[i]: round(float(preds[i]) * 100, 1)
                    for i in range(len(EMOTION_LABELS))
                }
            })

        return jsonify({'faces': results})

    except Exception as e:
        return jsonify({'error': str(e), 'faces': []})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
