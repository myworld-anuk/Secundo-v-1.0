
#!/usr/bin/env python3
"""
Deskew an image (sheet music or document) to ~0° regardless of tilt direction.
Uses multiple estimators (Hough lines, connected-components orientation,
and projection sweep) and combines them robustly.

Usage:
  python deskew_any_angle.py --input path/to/image.png --outdir outputs/

Optional flags:
  --debug           Save intermediate images (helpful to tune).
  --maxsweep 20     Max abs degrees for projection sweep (default 20).
  --step 0.5        Step size for projection sweep (default 0.5).
  --prefer hough    Force method: hough | rect | proj | auto (default auto).

Requirements:
  pip install opencv-python numpy
"""
import argparse
import math
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def to_gray_contrast(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale and apply CLAHE to boost contrast."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive binarization robust to lighting; returns 0/255 image."""
    # Otsu and adaptive; choose the one that preserves more black (staff lines/text)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )
    return adap if np.sum(adap == 0) >= np.sum(otsu == 0) else otsu


def normalize_angle_deg(a: float) -> float:
    """Map any angle to the range [-45, 45]."""
    while a <= -45.0:
        a += 90.0
    while a > 45.0:
        a -= 90.0
    return a


def detect_angle_hough(binary: np.ndarray, debug: bool, outdir: Path) -> float:
    """
    Estimate skew using Hough transform emphasizing horizontal lines.
    Returns angle in degrees (positive=CCW). NaN if not found.
    """
    # Emphasize long horizontal structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (61, 3))
    horiz = cv2.morphologyEx(255 - binary, cv2.MORPH_OPEN, kernel)
    edges = cv2.Canny(horiz, 30, 120, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=120)
    if lines is None or len(lines) == 0:
        return float("nan")

    angles = []
    for rho_theta in lines[:300]:
        rho, theta = rho_theta[0]
        # For horizontal lines, theta≈0 or π. Convert to angle around 0.
        angle_deg = (theta * 180.0 / np.pi) - 90.0
        angles.append(normalize_angle_deg(angle_deg))

    if debug:
        dbg = cv2.cvtColor(horiz, cv2.COLOR_GRAY2BGR)
        for rho_theta in (lines[:150] if lines is not None else []):
            rho, theta = rho_theta[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 2000 * (-b)), int(y0 + 2000 * (a))
            x2, y2 = int(x0 - 2000 * (-b)), int(y0 - 2000 * (a))
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(str(outdir / "hough_overlay.png"), dbg)

    return float(np.median(angles)) if angles else float("nan")


def detect_angle_rect(binary: np.ndarray) -> float:
    """Dominant orientation from minAreaRect of connected components."""
    inv = 255 - binary  # components white
    cnts, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    angles: List[float] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:  # filter tiny noise
            continue
        (cx, cy), (w, h), a = cv2.minAreaRect(c)
        if w < 5 or h < 5:
            continue
        # minAreaRect angle is (-90, 0]; adjust so that "along the long edge"
        angle = a if w >= h else a + 90
        angles.append(normalize_angle_deg(angle))
    return float(np.median(angles)) if angles else float("nan")


def detect_angle_projection(binary: np.ndarray, sweep: Tuple[float, float], step: float) -> float:
    """
    Try a grid of small rotations; pick the angle that maximizes horizontal-band sharpness
    (row-wise black-pixel variance). Slow but robust.
    """
    h, w = binary.shape
    best_angle, best_score = 0.0, -1.0
    a0, a1 = sweep
    angles = np.arange(a0, a1 + step, step)
    for a in angles:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), a, 1.0)
        rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        proj = np.mean(rot == 0, axis=1)  # fraction of black per row
        score = proj.var()
        if score > best_score:
            best_score, best_angle = score, a
    return float(best_angle)


def rotate_expand(img: np.ndarray, angle_deg: float, border: int = 30) -> np.ndarray:
    """Rotate image with expanded canvas (white background)."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int((h * sin) + (w * cos)) + 2 * border
    new_h = int((h * cos) + (w * sin)) + 2 * border
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=255)


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = max(a.shape[0], b.shape[0])
    w = a.shape[1] + b.shape[1]
    comp = 255 * np.ones((h, w, 3), dtype=np.uint8)
    comp[: a.shape[0], : a.shape[1]] = a
    comp[: b.shape[0], a.shape[1] : a.shape[1] + b.shape[1]] = b
    return comp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input image")
    ap.add_argument("--outdir", default="outputs", help="Directory for outputs")
    ap.add_argument("--debug", action="store_true", help="Save extra debug images")
    ap.add_argument("--maxsweep", type=float, default=20.0, help="Projection sweep ±degrees")
    ap.add_argument("--step", type=float, default=0.5, help="Projection sweep step (deg)")
    ap.add_argument("--prefer", choices=["auto", "hough", "rect", "proj"], default="auto",
                    help="Force a method or choose auto (median of available).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    ensure_dir(out_dir)

    img = cv2.imread(str(in_path))
    if img is None:
        raise SystemExit(f"Could not read image: {in_path}")

    gray = to_gray_contrast(img)
    if args.debug:
        cv2.imwrite(str(out_dir / "0_gray.png"), gray)

    binary = binarize(gray)
    if args.debug:
        cv2.imwrite(str(out_dir / "1_binary.png"), binary)

    # Angle estimates
    angle_h = detect_angle_hough(binary, args.debug, out_dir)
    angle_r = detect_angle_rect(binary)
    angle_p = detect_angle_projection(binary, (-args.maxsweep, args.maxsweep), args.step)

    # Combine
    candidates = []
    names = []
    if not math.isnan(angle_h):
        candidates.append(angle_h); names.append(("hough", angle_h))
    if not math.isnan(angle_r):
        candidates.append(angle_r); names.append(("rect", angle_r))
    if not math.isnan(angle_p):
        candidates.append(angle_p); names.append(("proj", angle_p))

    if args.prefer != "auto":
        chosen = {"hough": angle_h, "rect": angle_r, "proj": angle_p}[args.prefer]
    else:
        chosen = float(np.median(candidates)) if candidates else 0.0

    print("Angle estimates (deg):", ", ".join([f"{n}={a:.3f}" for n, a in names]))
    print(f"Chosen angle: {chosen:.3f} (positive=CCW). Applying correction...")

    # Rotate by the negative of the detected skew to bring to ~0°
    corrected = rotate_expand(img, -chosen, border=30)
    comp = side_by_side(img, corrected)
    cv2.imwrite(str(out_dir / "2_corrected.png"), corrected)
    cv2.imwrite(str(out_dir / "3_comparison.png"), comp)
    print(f"Saved: {out_dir/'2_corrected.png'}, {out_dir/'3_comparison.png'}")

if __name__ == "__main__":
    main()
