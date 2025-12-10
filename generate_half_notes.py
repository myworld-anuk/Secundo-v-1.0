import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 64

def make_half_note(img):
    """
    Convert a whole-note image (solid notehead) into a synthetic half-note
    (open notehead).
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Threshold to isolate ink
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours to locate the notehead
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img  # fallback

    # Assume largest contour = notehead region
    cnt = max(contours, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(cnt)

    # Create a synthetic open notehead at the same location
    out = bw.copy()

    # Erase the filled notehead
    cv2.rectangle(out, (x, y), (x+w, y+h), 0, -1)

    # Draw an ellipse (open notehead)
    center = (x + w // 2, y + h // 2)
    axes = (max(4, w // 2), max(3, h // 2))
    cv2.ellipse(out, center, axes, 0, 0, 360, 255, 2)

    # Invert back so white background, black ink
    result = cv2.bitwise_not(out)
    return result


def process_folder(set_root):
    """
    set_root = data/single_note/train
           or data/single_note/val
    """
    root = Path(set_root)

    for pitch_dir in root.iterdir():
        if not pitch_dir.is_dir():
            continue

        whole_dir = pitch_dir / "whole"
        half_dir  = pitch_dir / "half"

        if not whole_dir.exists():
            print(f"[WARN] No whole folder in {pitch_dir}, skipping.")
            continue

        half_dir.mkdir(exist_ok=True)

        pngs = list(whole_dir.glob("*.png"))
        if len(pngs) == 0:
            print(f"[WARN] No images in {whole_dir}, skipping.")
            continue

        print(f"Processing {pitch_dir.name}: {len(pngs)} whole notes")

        for i, img_path in enumerate(pngs):
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            half = make_half_note(img)

            out_name = f"{pitch_dir.name}-half-{i+1}.png"
            cv2.imwrite(str(half_dir / out_name), half)

        print(f" → Generated {len(pngs)} half notes for {pitch_dir.name}")


if __name__ == "__main__":
    print("Generating synthetic half notes…")
    process_folder("data/single_note/train")
    process_folder("data/single_note/val")
    print("Done!")