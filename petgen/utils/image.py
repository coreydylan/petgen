"""Image processing utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_image(path: Path) -> np.ndarray:
    """Load an image as RGB numpy array."""
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: Path) -> Path:
    """Save an RGB numpy array as an image file."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return path


def auto_crop_face(image: np.ndarray, bbox: tuple[int, int, int, int], padding: float = 0.3) -> np.ndarray:
    """Crop image around a face bounding box with padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    px, py = int(bw * padding), int(bh * padding)
    cx1 = max(0, x1 - px)
    cy1 = max(0, y1 - py)
    cx2 = min(w, x2 + px)
    cy2 = min(h, y2 + py)
    return image[cy1:cy2, cx1:cx2]


def remove_background(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove background from an image. Returns (foreground_rgb, alpha_mask)."""
    # Placeholder — will use rembg or SAM2 in implementation
    raise NotImplementedError("TODO: implement background removal")


def ensure_resolution(
    image: np.ndarray,
    min_width: int = 512,
    min_height: int = 512,
) -> np.ndarray:
    """Upscale image if below minimum resolution."""
    import cv2

    h, w = image.shape[:2]
    if w >= min_width and h >= min_height:
        return image
    scale = max(min_width / w, min_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
