import os, sys, json, math, glob, argparse, random
from pathlib import Path

import numpy as np
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

# -----------------------------
# Config defaults
# -----------------------------
IMG_SIZE = 64  # square input to CNN
SEED     = 42
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# -----------------------------
# OpenCV-based preprocessing
# -----------------------------
def preprocess_bgr_to_tensor(img_bgr, target=IMG_SIZE):
    """
    Robust-ish preprocessing for printed sheet music crops:
      1) grayscale
      2) light deskew (via moments)
      3) adaptive threshold (binarize)
      4) center + pad to square
      5) resize -> target x target
      6) normalize to [0,1], shape (H,W,1)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- deskew using image moments (works when notehead/lines dominate)
    m = cv2.moments(gray)
    if abs(m["mu02"]) > 1e-3:
        skew = m["mu11"] / m["mu02"]
        angle = math.degrees(math.atan(skew))
    else:
        angle = 0.0
    if abs(angle) > 0.3 and abs(angle) < 10:  # avoid wild rotations
        (h, w) = gray.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # --- adaptive threshold for consistent contrasts
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 10)

    # --- get tight bbox of ink, center it, pad to square
    ys, xs = np.where(bw > 0)
    if len(xs) < 10 or len(ys) < 10:
        # fallback if almost blank: just resize gray
        roi = bw
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        roi = bw[y0:y1+1, x0:x1+1]

    h, w = roi.shape[:2]
    pad = abs(h - w)
    if h > w:
        left = pad//2; right = pad - left
        roi = cv2.copyMakeBorder(roi, 0, 0, left, right, cv2.BORDER_CONSTANT, value=0)
    elif w > h:
        top = pad//2; bottom = pad - top
        roi = cv2.copyMakeBorder(roi, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)

    # --- final resize and normalize
    roi = cv2.resize(roi, (target, target), interpolation=cv2.INTER_AREA)
    roi = roi.astype(np.float32) / 255.0
    roi = np.expand_dims(roi, axis=-1)  # (H,W,1)
    return roi

def load_image_as_tensor(path, target=IMG_SIZE):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return preprocess_bgr_to_tensor(img, target=target)

# -----------------------------
# tf.data builders
# -----------------------------
def list_dataset(root_dir):
    """
    Scans root_dir for class subfolders, returns (paths, labels, class_names).
    """
    root = Path(root_dir)
    class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    paths, labels = [], []
    for ci, cn in enumerate(class_names):
        for p in glob.glob(str(root / cn / "*")):
            if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                paths.append(p); labels.append(ci)
    return paths, labels, class_names

def make_tf_dataset(paths, labels, batch_size=64, shuffle=True, augment=True):
    x = np.array(paths)
    y = np.array(labels, dtype=np.int32)

    def _parse(path, label):
        path = path.numpy().decode("utf-8")
        arr = load_image_as_tensor(path)  # (H,W,1)
        return arr, label

    def _tf_parse(path, label):
        arr, lab = tf.py_function(_parse, [path, label], [tf.float32, tf.int32])
        arr.set_shape((IMG_SIZE, IMG_SIZE, 1))
        lab.set_shape(())
        return arr, lab

    ds = tf.data.Dataset.from_tensor_slices((x, y)).map(_tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

    # Lightweight data augmentation (good for generalization on scans/photos)
    if augment:
        def _aug(img, lab):
            # random slight affine & elastic-ish blur
            img = tf.image.random_flip_left_right(img)  # harmless
            img = tf.image.random_brightness(img, 0.15)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            return img, lab
        ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(4096, seed=SEED)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------
# Model (compact CNN)
# -----------------------------
def build_cnn(n_classes):
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)  # 32 -> 32

    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)  # 32 -> 16

    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)  # 16 -> 8

    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)  # 8 -> 4

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="SingleNoteOMR_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

# -----------------------------
# MusicXML (single note helper)
# -----------------------------
def write_single_note_musicxml(pitch_name, out_path="single_note.xml", divisions=8, duration_type="whole",
                               beats=4, beat_type=4, clef_sign="G", clef_line=2, key_fifths=0):
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom

    DUR = {"whole":4,"half":2,"quarter":1,"eighth":0.5,"16th":0.25}
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
    ks = SubElement(attr, "key"); SubElement(ks, "fifths").text = str(key_fifths)
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

    xml = prettify(score)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    return out_path

# -----------------------------
# Train / Evaluate
# -----------------------------
def train(args):
    train_paths, train_labels, classes = list_dataset(args.train_dir)
    if len(train_paths) == 0:
        raise RuntimeError("No training images found.")
    val_paths, val_labels, classes_val = list_dataset(args.val_dir)
    if classes != classes_val:
        raise RuntimeError("Class folders in train and val do not match.\n"
                           f"train classes: {classes}\nval classes: {classes_val}")

    print(f"Classes ({len(classes)}): {classes}")
    bs = args.batch_size

    ds_train = make_tf_dataset(train_paths, train_labels, batch_size=bs, shuffle=True, augment=True)
    ds_val   = make_tf_dataset(val_paths,   val_labels,   batch_size=bs, shuffle=False, augment=False)

    model = build_cnn(n_classes=len(classes))
    model.summary()

    ckpt_path = Path(args.out_dir) / "best.keras"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
    ]

    history = model.fit(ds_train, validation_data=ds_val, epochs=args.epochs, callbacks=callbacks)

    # Evaluate and print sklearn report
    y_true, y_pred = [], []
    for xb, yb in ds_val:
        probs = model.predict(xb, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(yb.numpy().tolist())

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)

    # Save label map
    label_map_path = Path(args.out_dir) / "labels.json"
    with open(label_map_path, "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    print(f"Saved best model to: {ckpt_path}")
    print(f"Saved labels to:     {label_map_path}")

def predict_image(args):
    # Load model + labels
    model = tf.keras.models.load_model(args.model_path)
    with open(args.labels_path, "r") as f:
        classes = json.load(f)["classes"]

    # Preprocess
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.image)
    x = preprocess_bgr_to_tensor(img, target=IMG_SIZE)
    x = np.expand_dims(x, axis=0)  # batch

    # Predict
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    pred_class = classes[idx]
    conf = float(probs[idx])

    print(f"Prediction: {pred_class}  (conf {conf:.3f})")

    # Optional: write a one-note MusicXML with that pitch
    if args.write_musicxml:
        out_xml = Path(args.out_xml or "predicted_note.xml")
        path = write_single_note_musicxml(
            pred_class,
            out_path=str(out_xml),
            duration_type="whole"
        )
        print(f"Wrote MusicXML: {path}")

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Single-note OMR with TensorFlow (CNN).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train a CNN on single-note images.")
    tr.add_argument("--train_dir", required=True, help="Path to train/CLASS/*.png")
    tr.add_argument("--val_dir",   required=True, help="Path to val/CLASS/*.png")
    tr.add_argument("--epochs", type=int, default=30)
    tr.add_argument("--batch_size", type=int, default=64)
    tr.add_argument("--out_dir", default="omr_runs/singlenote")
    tr.set_defaults(func=train)

    pr = sub.add_parser("predict", help="Predict a single image and (optionally) emit MusicXML.")
    pr.add_argument("--image", required=True)
    pr.add_argument("--model_path", required=True)
    pr.add_argument("--labels_path", required=True)
    pr.add_argument("--write_musicxml", action="store_true")
    pr.add_argument("--out_xml", default=None)
    pr.set_defaults(func=predict_image)

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.func(args)