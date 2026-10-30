"""Image normalisation: binarisation, quality metrics, deskew and rotation."""
from __future__ import annotations

import cv2
import numpy as np

from .config import INK_THRESHOLD, MAX_DESKEW_DEG


def binarize(gray: np.ndarray) -> np.ndarray:
    """Foreground (ink + rules) = 255. Adaptive so uneven lighting / stains don't flood it."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)


def ink_ratio(gray: np.ndarray) -> float:
    return float((gray < INK_THRESHOLD).mean())


def blur_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_skew(gray: np.ndarray) -> float:
    """Dominant angle (deg) of long near-horizontal strokes, i.e. the table rules.

    Robust to handwriting because only segments longer than 1/4 page width vote,
    and the vote is the length-weighted median.
    """
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    if edges.mean() / 255 > 0.15:
        return 0.0  # noise / texture, not a document: Hough would be slow and meaningless
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=100,
                            minLineLength=w // 4, maxLineGap=8)
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(ang) <= MAX_DESKEW_DEG:
            angles.append(ang)
            weights.append(np.hypot(x2 - x1, y2 - y1))
    if not angles:
        return 0.0
    order = np.argsort(angles)
    cum = np.cumsum(np.asarray(weights)[order])
    return float(np.asarray(angles)[order][np.searchsorted(cum, cum[-1] / 2)])


def rotate(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate about the centre, expanding the canvas and padding with white."""
    if abs(angle_deg) < 0.05:
        return gray
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(gray, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)


def rotate90(gray: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(gray, k=-quarter_turns % 4))


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    angle = estimate_skew(gray)
    return rotate(gray, angle), angle
