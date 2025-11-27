# omr_server.py
import json, math
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

# --- must match single_note_cnn.py ---
IMG_SIZE = 64

def preprocess_bgr_to_tensor(img_bgr, target=IMG_SIZE):
    """Same as in single_note_cnn.py."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    m = cv2.moments(gray)
    if abs(m["mu02"]) > 1e-3:
        skew = m["mu11"] / m["mu02"]
        angle = math.degrees(math.atan(skew))
    else:
        angle = 0.0
    if abs(angle) > 0.3 and abs(angle) < 10:
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    bw = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    ys, xs = np.where(bw > 0)
    if len(xs) < 10 or len(ys) < 10:
        roi = bw
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        roi = bw[y0:y1 + 1, x0:x1 + 1]

    h, w = roi.shape[:2]
    pad = abs(h - w)
    if h > w:
        left = pad // 2
        right = pad - left
        roi = cv2.copyMakeBorder(
            roi, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0
        )
    elif w > h:
        top = pad // 2
        bottom = pad - top
        roi = cv2.copyMakeBorder(
            roi, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0
        )

    roi = cv2.resize(roi, (target, target), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    roi = np.expand_dims(roi, axis=-1)
    return roi

def build_single_note_musicxml(
    pitch_name,
    divisions=8,
    duration_type="whole",
    beats=4,
    beat_type=4,
    clef_sign="G",
    clef_line=2,
    key_fifths=0,
):
    """Return a MusicXML string for a single-note, single-measure score."""
    DUR = {"whole": 4, "half": 2, "quarter": 1, "eighth": 0.5, "16th": 0.25}
    dur_divs = int(DUR[duration_type] * divisions)

    def prettify(elem):
        rough = tostring(elem, encoding="utf-8")
        return minidom.parseString(rough).toprettyxml(indent="  ")

    score = Element("score-partwise", version="3.1")
    part_list = SubElement(score, "part-list")
    sp = SubElement(part_list, "score-part", id="P1")
    SubElement(sp, "part-name").text = "OMR"

    part = SubElement(score, "part", id="P1")
    measure = SubElement(part, "measure", number="1")
    attr = SubElement(measure, "attributes")
    SubElement(attr, "divisions").text = str(divisions)
    ks = SubElement(attr, "key")
    SubElement(ks, "fifths").text = str(key_fifths)
    ts = SubElement(attr, "time")
    SubElement(ts, "beats").text = str(beats)
    SubElement(ts, "beat-type").text = str(beat_type)
    clef = SubElement(attr, "clef")
    SubElement(clef, "sign").text = clef_sign
    SubElement(clef, "line").text = str(clef_line)

    note = SubElement(measure, "note")
    p = SubElement(note, "pitch")
    step, octave = pitch_name[:-1], int(pitch_name[-1])
    SubElement(p, "step").text = step
    SubElement(p, "octave").text = str(octave)
    SubElement(note, "duration").text = str(dur_divs)
    SubElement(note, "voice").text = "1"
    SubElement(note, "type").text = duration_type
    SubElement(note, "staff").text = "1"

    return prettify(score)

# --- Load model + labels once at startup ---

MODEL_PATH = Path("runs/seven_notes/best.keras")  # adjust if needed
LABELS_PATH = Path("runs/seven_notes/labels.json")

print("Loading model...")
model = tf.keras.models.load_model(str(MODEL_PATH))
with open(LABELS_PATH, "r") as f:
    classes = json.load(f)["classes"]
print("Loaded model with classes:", classes)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/omr", response_class=PlainTextResponse)
async def omr_endpoint(file: UploadFile = File(...)):
    """
    Accept an uploaded measure image, classify the note, and return MusicXML text.
    """
    contents = await file.read()
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    x = preprocess_bgr_to_tensor(img, target=IMG_SIZE)
    x = np.expand_dims(x, axis=0)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    pitch_name = classes[idx]

    xml_str = build_single_note_musicxml(pitch_name)

    # Frontend can both preview this text and offer it as a .musicxml download
    return xml_str