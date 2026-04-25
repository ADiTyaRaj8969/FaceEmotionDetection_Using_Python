import os
import base64
import logging
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

_BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_BASE_DIR, 'emotion.tflite')
_YUNET_PATH = os.path.join(_BASE_DIR, 'face_detection_yunet.onnx')

# ── Load TF Lite emotion model ────────────────────────────────────────────────
_interp     = None
_inp_idx    = None
_out_idx    = None
model_error = None

if not os.path.exists(_MODEL_PATH):
    model_error = f'emotion.tflite not found at {_MODEL_PATH}'
    log.error(model_error)
else:
    for _loader in [
        lambda: __import__('tflite_runtime.interpreter', fromlist=['Interpreter']).Interpreter,
        lambda: __import__('ai_edge_litert.interpreter', fromlist=['Interpreter']).Interpreter,
        lambda: __import__('tensorflow', fromlist=['lite']).lite.Interpreter,
    ]:
        try:
            Interpreter = _loader()
            _interp = Interpreter(model_path=_MODEL_PATH, num_threads=2)
            _interp.allocate_tensors()
            _inp_idx = _interp.get_input_details()[0]['index']
            _out_idx = _interp.get_output_details()[0]['index']
            log.info('TF Lite emotion model loaded OK')
            break
        except Exception as _e:
            log.warning('TFLite loader attempt failed: %s', _e)
    if _interp is None:
        model_error = 'All TFLite loading methods failed'
        log.error(model_error)


def predict_emotion(gray_roi):
    roi = cv2.resize(gray_roi, (48, 48)).astype('float32') / 255.0
    roi = np.reshape(roi, (1, 48, 48, 1))
    _interp.set_tensor(_inp_idx, roi)
    _interp.invoke()
    preds = _interp.get_tensor(_out_idx)[0]
    idx   = int(np.argmax(preds))
    return (
        EMOTION_LABELS[idx],
        float(preds[idx]),
        {EMOTION_LABELS[i]: round(float(preds[i]) * 100, 1) for i in range(7)},
    )


# ── Face detection: YuNet → Haar fallback ────────────────────────────────────
_yunet    = None
USE_YUNET = False

if os.path.exists(_YUNET_PATH) and hasattr(cv2, 'FaceDetectorYN'):
    try:
        _yunet = cv2.FaceDetectorYN.create(
            _YUNET_PATH, '', (320, 320),
            score_threshold=0.3,   # lowered from 0.5 — catches more faces
            nms_threshold=0.3,
            top_k=5000,
        )
        USE_YUNET = True
        log.info('YuNet face detection ready')
    except Exception as e:
        log.warning('YuNet init failed (%s), using Haar', e)
else:
    log.info('YuNet unavailable, using Haar cascade')

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)


def _nms(boxes, iou_thresh=0.35):
    if len(boxes) <= 1:
        return boxes
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept  = []
    for box in boxes:
        x1, y1, w1, h1 = box
        dup = False
        for kx, ky, kw, kh in kept:
            ix  = max(x1, kx);        iy  = max(y1, ky)
            ix2 = min(x1+w1, kx+kw);  iy2 = min(y1+h1, ky+kh)
            if ix2 <= ix or iy2 <= iy:
                continue
            inter = (ix2-ix) * (iy2-iy)
            union = w1*h1 + kw*kh - inter
            if inter / union > iou_thresh:
                dup = True
                break
        if not dup:
            kept.append(box)
    return kept


def _pad_box(x, y, w, h, fw, fh, pad=0.15):
    """Expand face bounding box by pad% on each side for better emotion context."""
    px, py = int(w * pad), int(h * pad)
    x2 = max(0, x - px)
    y2 = max(0, y - py)
    w2 = min(fw - x2, w + 2 * px)
    h2 = min(fh - y2, h + 2 * py)
    return x2, y2, w2, h2


def detect_faces(frame):
    fh, fw = frame.shape[:2]
    min_px  = max(40, int(fw * 0.04))

    if USE_YUNET and _yunet is not None:
        try:
            _yunet.setInputSize((fw, fh))
            _, det = _yunet.detect(frame)
            boxes  = []
            if det is not None:
                for d in det:
                    x, y, bw, bh = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                    x  = max(0, x);    y  = max(0, y)
                    bw = min(bw, fw-x); bh = min(bh, fh-y)
                    if bw >= min_px and bh >= min_px:
                        boxes.append((x, y, bw, bh))
            return _nms(boxes)
        except Exception as e:
            log.warning('YuNet error: %s', e)

    # Haar fallback — more sensitive parameters
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = _haar.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=3,
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return _nms(list(found) if len(found) else [])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({
        'status':        'ok' if _interp is not None else 'degraded',
        'model_loaded':  _interp is not None,
        'model_error':   model_error,
        'face_detector': 'yunet' if USE_YUNET else 'haar',
        'yunet_file':    os.path.exists(_YUNET_PATH),
        'tflite_file':   os.path.exists(_MODEL_PATH),
    })


@app.route('/predict', methods=['POST'])
def predict():
    if _interp is None:
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

        # Resize very large frames for speed (cap at 640px wide)
        fh, fw = frame.shape[:2]
        if fw > 640:
            scale = 640 / fw
            frame = cv2.resize(frame, (640, int(fh * scale)))
            fh, fw = frame.shape[:2]

        faces   = detect_faces(frame)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = []

        for (x, y, w, h) in faces:
            # Pad the ROI slightly so the model gets more context
            px, py, pw, ph = _pad_box(x, y, w, h, fw, fh)
            emotion, confidence, scores = predict_emotion(gray[py:py+ph, px:px+pw])
            results.append({
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'emotion':    emotion,
                'confidence': round(confidence, 4),
                'scores':     scores,
            })

        log.info('predict: %d face(s) detected', len(results))
        return jsonify({'faces': results})

    except Exception as e:
        log.exception('predict error')
        return jsonify({'error': str(e), 'faces': []})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
