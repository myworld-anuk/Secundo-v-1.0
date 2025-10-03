#!/usr/bin/env python3
"""
Deskew (rotation correction) for scanned/photographed sheet music.

Usage:
  python deskew_starter.py --input path/to/image.png --outdir outputs/

What it does:
  1) Loads an image (gray + binarize).
  2) Estimates skew angle two ways:
       A) Hough-line method (good when staff lines are visible).
       B) minAreaRect on connected components (fallback).
  3) Rotates the image to deskew and saves intermediate results.
Requires: opencv-python, numpy
"""
import argparse
import math
import os
from pathlib import Path

import cv2
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Adaptive binarization robust to lighting; invert so notes are white on black if needed."""
    # Otsu as a baseline
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Adaptive (often better on uneven lighting)
    adap = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
    )   
    # Heuristic: pick the one with more black pixels (staff lines)
    if np.sum(otsu == 0) > np.sum(adap == 0):
        return otsu
    return adap


def detect_angle_hough(binary: np.ndarray) -> float:
    """
    Estimate skew using Hough transform on horizontal staff lines.
    Returns angle in degrees. Positive angle means rotate CCW to correct.
    """
    # Emphasize horizontal lines using morphological opening on a wide, short kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3))  # was (51,3)
    horiz = cv2.morphologyEx(255 - binary, cv2.MORPH_OPEN, kernel)  # invert to make lines bright
    edges = cv2.Canny(horiz, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=120)

    if lines is None or len(lines) == 0:
        return float("nan")

    # Convert rho,theta to angles relative to horizontal
    angles = []
    for rho_theta in lines[:200]:  # cap for speed
        rho, theta = rho_theta[0]
        # For perfectly horizontal lines, theta ≈ 0 or π. Convert to slope angle in degrees.
        # Angle of the *line* relative to x-axis:
        angle_deg = (theta * 180.0 / np.pi) - 90.0  # normalize around 0 for horizontal
        # bring to [-45, 45] for robustness
        while angle_deg <= -45: angle_deg += 90
        while angle_deg > 45: angle_deg -= 90
        angles.append(angle_deg)

    if not angles:
        return float("nan")
    
    dbg = cv2.cvtColor(horiz, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for rho_theta in lines[:100]:
            rho, theta = rho_theta[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            x1, y1 = int(x0 + 2000*(-b)), int(y0 + 2000*(a))
            x2, y2 = int(x0 - 2000*(-b)), int(y0 - 2000*(a))
            cv2.line(dbg, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.imwrite(str(Path("outputs") / "hough_overlay.png"), dbg)

    # Use median to reduce outliers
    return float(np.median(angles))


def detect_angle_minarearect(binary: np.ndarray) -> float:
    """
    Fallback angle using minAreaRect on connected components.
    Computes the dominant orientation of text/lines blobs.
    """
    # Invert: components should be white
    inv = 255 - binary
    cnts, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for c in cnts:
        if cv2.contourArea(c) < 150:  # skip tiny noise
            continue
        rect = cv2.minAreaRect(c)  # (center, (w,h), angle in degrees)
        (cx, cy), (w, h), a = rect
        if w < 5 or h < 5:
            continue
        # minAreaRect angle is in (-90, 0]; convert to near-horizontal
        angle = a if w >= h else a + 90
        # bring to [-45, 45]
        while angle <= -45: angle += 90
        while angle > 45: angle -= 90
        angles.append(angle)

    if not angles:
        return float("nan")

    return float(np.median(angles))


def rotate_image(img: np.ndarray, angle_deg: float, border: int = 10) -> np.ndarray:
    """Rotate around center with expanded canvas to avoid clipping."""
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Compute new bounds to avoid cropping
    cos = abs(rot_mat[0, 0])
    sin = abs(rot_mat[0, 1])
    new_w = int((h * sin) + (w * cos)) + 2 * border
    new_h = int((h * cos) + (w * sin)) + 2 * border

    # Adjust rotation matrix for translation
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        img, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=255
    )
    return rotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image (png/jpg)")
    parser.add_argument("--outdir", default="outputs", help="Directory to save results")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.outdir)
    ensure_dir(out_dir)

    img = cv2.imread(str(in_path))
    if img is None:
        raise SystemExit(f"Could not read image: {in_path}")

    # 1) convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) improve contrast BEFORE thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    cv2.imwrite(str(out_dir / "0_gray.png"), gray)

    binary = binarize(gray)
    cv2.imwrite(str(out_dir / "1_binary.png"), binary)

    angle_hough = detect_angle_hough(binary)
    angle_rect = detect_angle_minarearect(binary)

    # Choose angle: prefer Hough (staff lines), else fallback
    chosen = angle_rect
    if math.isnan(chosen):
        print("Warning: Could not reliably estimate angle; defaulting to 0.")
        chosen = 0.0

    print(f"Estimated angle (Hough): {angle_hough:.3f} deg")
    print(f"Estimated angle (minAreaRect): {angle_rect:.3f} deg")
    print(f"Chosen angle: {chosen:.3f} deg (rotate by -chosen to deskew)")

    # Rotate opposite to the detected skew to correct
    # rotate in the SAME sign as the estimated angle (so negative => CW correction)
    deskewed = rotate_image(img, chosen, border=20)
    cv2.imwrite(str(out_dir / "2_deskewed.png"), deskewed)

    # Save a side-by-side comparison for quick QA
    h = max(img.shape[0], deskewed.shape[0])
    w = img.shape[1] + deskewed.shape[1]
    comp = 255 * np.ones((h, w, 3), dtype=np.uint8)
    comp[: img.shape[0], : img.shape[1]] = img
    comp[: deskewed.shape[0], img.shape[1] : img.shape[1] + deskewed.shape[1]] = deskewed
    cv2.imwrite(str(out_dir / "3_comparison.png"), comp)

    print(f"Saved results to: {out_dir}")

if __name__ == "__main__":
    main()
