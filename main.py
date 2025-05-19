"""
Object-color classifier
───────────────────────
For every input image, assign one of three labels:

• black        → conveyor belt A
• transparent  → conveyor belt B
• colorful     → conveyor belt C
"""

import sys
from pathlib import Path

import cv2
import numpy as np


# ── Tunable thresholds ───────────────────────────────────────────────────────
BLACK_PCT_THRESHOLD = 0.30   # ≥30 % very-dark pixels ⇒ label as “black”
DARK_PIXEL_THRESH   = 60     # V-channel value below which a pixel is “dark”

TRANSP_V_MEAN       = 190    # High brightness …
TRANSP_S_MEAN       = 35     # … plus low saturation ⇒ “transparent”

CROP_FRAC           = 0.30   # Keep only the central 30 % of the image

BELT = {"black": "A", "transparent": "B", "colorful": "C"}  # belt mapping


# ── Helper functions ─────────────────────────────────────────────────────────
def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk; raise if it doesn’t exist."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return img


def center_crop(img: np.ndarray, frac: float = CROP_FRAC) -> np.ndarray:
    """
    Return the central *frac* region of the image.
    This reduces the influence of bright or cluttered backgrounds.
    """
    h, w = img.shape[:2]
    ch, cw = int(h * frac / 2), int(w * frac / 2)  # half-sizes of crop
    cy, cx = h // 2, w // 2                        # image centre
    return img[cy - ch : cy + ch, cx - cw : cx + cw]


def preprocess(img: np.ndarray, max_side: int = 512) -> np.ndarray:
    """
    Down-scale very large images so that their longest side is ≤ *max_side*.
    Makes the algorithm faster without hurting accuracy.
    """
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)                                       # scale factor
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


# ── Core classification logic ───────────────────────────────────────────────
def classify(hsv: np.ndarray, *, debug: bool = False) -> str:
    """
    Decide whether the HSV image is predominantly black, transparent, or colorful.
    Uses simple brightness/saturation heuristics.
    """
    v, s = hsv[..., 2], hsv[..., 1]          # value (brightness) and saturation
    v_mean, s_mean = v.mean(), s.mean()
    dark_pct = (v < DARK_PIXEL_THRESH).mean()  # fraction of very-dark pixels
    v_median = np.median(v)

    if debug:
        print(
            f"[DEBUG] v_mean={v_mean:.1f}  s_mean={s_mean:.1f}  "
            f"v_median={v_median:.1f}  dark_pct={dark_pct:.2f}"
        )

    # ----- rule set -----
    if dark_pct >= BLACK_PCT_THRESHOLD or v_median < 40:
        return "black"
    if v_mean >= TRANSP_V_MEAN and s_mean <= TRANSP_S_MEAN:
        return "transparent"
    return "colorful"


def detect(path: str | Path, *, debug: bool = False) -> tuple[str, str]:
    """
    Full pipeline: load → crop → resize → HSV convert → classify.
    Returns both the label and its conveyor-belt letter.
    """
    img = load_image(path)
    img = center_crop(img)
    img = preprocess(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    label = classify(hsv, debug=debug)
    return label, BELT[label]


# ── Command-line interface ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Expect: python main.py image.jpg [--debug]
    if len(sys.argv) < 2:
        sys.exit("Usage:  python main.py image.jpg [--debug]")

    img_path = sys.argv[1]
    debug = "--debug" in sys.argv

    label, belt = detect(img_path, debug=debug)
    print(f"Detected **{label}** object → send to conveyor belt **{belt}**")
