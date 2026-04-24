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


def _rebuild_and_load(path):
    import tensorflow as tf
    m = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax'),
    ])
    m.load_weights(path)
    return m


for _loader in [
    lambda: __import__('tensorflow.keras.models', fromlist=['load_model']).load_model('emotion.h5', compile=False),
    lambda: __import__('keras.models', fromlist=['load_model']).load_model('emotion.h5', compile=False),
    lambda: _rebuild_and_load('emotion.h5'),
]:
    try:
        model = _loader()
        print("Emotion model loaded successfully.")
        break
    except Exception as e:
        model_error = str(e)

if model is None:
    print(f"WARNING: Emotion model failed to load — {model_error}")

# ── Face detection: MTCNN → Haar fallback ────────────────────────────────────
_mtcnn    = None
USE_MTCNN = False

try:
    from mtcnn import MTCNN
    _mtcnn    = MTCNN()
    USE_MTCNN = True
    print("MTCNN face detection ready.")
except Exception as e:
    print(f"MTCNN unavailable ({e}), falling back to Haar cascade.")

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Warm-up: compile TF graphs at startup, not on the first request ───────────
try:
    _dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    if USE_MTCNN and _mtcnn is not None:
        _mtcnn.detect_faces(_dummy_frame)
        print("MTCNN warmed up.")
    if model is not None:
        _dummy_roi = np.zeros((1, 48, 48, 1), dtype=np.float32)
        model.predict(_dummy_roi, verbose=0)
        print("Emotion model warmed up.")
except Exception as e:
    print(f"Warmup error (non-fatal): {e}")


def _nms(boxes, iou_thresh=0.35):
    if len(boxes) <= 1:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dup = False
        for kx, ky, kw, kh in kept:
            ix  = max(x1, kx);       iy  = max(y1, ky)
            ix2 = min(x1+w1, kx+kw); iy2 = min(y1+h1, ky+kh)
            if ix2 <= ix or iy2 <= iy:
                continue
            inter = (ix2 - ix) * (iy2 - iy)
            union = w1*h1 + kw*kh - inter
            if inter / union > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(box)
    return kept


def detect_faces(frame):
    h, w   = frame.shape[:2]
    min_px = max(40, int(w * 0.05))

    if USE_MTCNN and _mtcnn is not None:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _mtcnn.detect_faces(rgb)
        boxes   = []
        for r in results:
            if r['confidence'] < 0.85:
                continue
            x, y, bw, bh = r['box']
            x  = max(0, x);   y  = max(0, y)
            bw = min(bw, w - x); bh = min(bh, h - y)
            if bw >= min_px and bh >= min_px:
                boxes.append((x, y, bw, bh))
        return _nms(boxes)

    # Haar fallback
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = _haar.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 40)
    )
    return _nms(list(found) if len(found) else [])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status':        'ok' if model is not None else 'degraded',
        'model_loaded':  model is not None,
        'model_error':   model_error,
        'face_detector': 'mtcnn' if USE_MTCNN else 'haar',
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
