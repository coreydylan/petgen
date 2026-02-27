"""Semantic segmentation of animal faces into regions.

Replaces MuseTalk's BiSeNet human face parser with a landmark-based
approach using convex hulls for animal face region mapping.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from petgen.config import PetGenConfig
from petgen.models import Landmark

logger = logging.getLogger(__name__)

# Semantic region IDs
REGIONS = {
    0: "background",
    1: "nose",
    2: "left_eye",
    3: "right_eye",
    4: "left_ear",
    5: "right_ear",
    6: "mouth",
    7: "forehead",
    8: "jaw",
    9: "neck",
}

# Landmark index ranges for CatFLW (48 landmarks) and DogFLW (46 landmarks).
# These map landmark groups to semantic regions.
_LANDMARK_GROUPS_CAT = {
    "left_ear": (0, 2),      # landmarks 0-1
    "right_ear": (2, 4),     # landmarks 2-3
    "left_eye": (4, 8),      # landmarks 4-7
    "right_eye": (8, 12),    # landmarks 8-11
    "nose": (26, 33),        # landmarks 26-32
    "mouth": (33, 48),       # landmarks 33-47
}

_LANDMARK_GROUPS_DOG = {
    "left_ear": (0, 2),
    "right_ear": (2, 4),
    "left_eye": (4, 8),
    "right_eye": (8, 12),
    "nose": (26, 33),
    "mouth": (33, 46),       # landmarks 33-45
}

_GROUP_TO_REGION_ID = {
    "left_ear": 4,
    "right_ear": 5,
    "left_eye": 2,
    "right_eye": 3,
    "nose": 1,
    "mouth": 6,
}

# Colors for visualization (BGR order for cv2)
_REGION_COLORS = {
    0: (0, 0, 0),        # background
    1: (0, 200, 200),     # nose - yellow
    2: (200, 0, 0),       # left eye - blue
    3: (200, 0, 0),       # right eye - blue
    4: (0, 200, 0),       # left ear - green
    5: (0, 200, 0),       # right ear - green
    6: (0, 0, 200),       # mouth - red
    7: (200, 200, 0),     # forehead - cyan
    8: (200, 0, 200),     # jaw - magenta
    9: (100, 100, 100),   # neck - gray
}


class AnimalFaceParser:
    """Semantic segmentation of animal faces into regions.

    Replaces MuseTalk's BiSeNet human face parser.
    Segments: nose, eyes, ears, mouth, forehead, jaw, background.
    """

    REGIONS = REGIONS

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._face_detector = None

    def _get_face_detector(self):
        """Lazy-load PetFaceDetector for species detection."""
        if self._face_detector is None:
            from petgen.modules.face_detect import PetFaceDetector

            self._face_detector = PetFaceDetector(self.config)
        return self._face_detector

    def parse(
        self, image: np.ndarray, landmarks: list[Landmark] | None = None
    ) -> np.ndarray:
        """Generate semantic segmentation map for an animal face.

        Uses landmarks + convex hull regions. If no landmarks are provided,
        attempts to detect face and generate heuristic landmarks.

        Args:
            image: RGB image array (H, W, 3).
            landmarks: Optional list of facial landmarks.

        Returns:
            (H, W) array of region IDs (see REGIONS dict).
        """
        if landmarks is None or len(landmarks) == 0:
            # Try to detect and generate landmarks
            detector = self._get_face_detector()
            bbox = detector.detect_face(image)
            if bbox is None:
                return np.zeros(image.shape[:2], dtype=np.uint8)
            landmarks = detector.detect_landmarks(image, bbox)

        return self._landmarks_to_regions(image.shape[:2], landmarks)

    def _landmarks_to_regions(
        self, image_shape: tuple[int, int], landmarks: list[Landmark]
    ) -> np.ndarray:
        """Map landmark groups to semantic regions using convex hulls.

        Args:
            image_shape: (H, W) of the image.
            landmarks: Full set of facial landmarks.

        Returns:
            (H, W) array of region IDs.
        """
        seg_map = np.zeros(image_shape, dtype=np.uint8)
        n = len(landmarks)

        # Choose landmark groups based on landmark count
        if n >= 48:
            groups = _LANDMARK_GROUPS_CAT
        elif n >= 46:
            groups = _LANDMARK_GROUPS_DOG
        else:
            # For short landmark lists, use proportional splits
            groups = self._approximate_groups(n)

        for group_name, (start, end) in groups.items():
            if start >= n:
                continue
            actual_end = min(end, n)
            group_lms = landmarks[start:actual_end]
            if len(group_lms) < 3:
                # Need at least 3 points for a convex hull; use a circle instead
                if group_lms:
                    cx = int(np.mean([lm.x for lm in group_lms]))
                    cy = int(np.mean([lm.y for lm in group_lms]))
                    radius = max(5, int(np.std([lm.x for lm in group_lms] +
                                                [lm.y for lm in group_lms]) * 2))
                    region_id = _GROUP_TO_REGION_ID.get(group_name, 0)
                    cv2.circle(seg_map, (cx, cy), radius, int(region_id), -1)
                continue

            pts = np.array([[int(lm.x), int(lm.y)] for lm in group_lms])
            hull = cv2.convexHull(pts)
            region_id = _GROUP_TO_REGION_ID.get(group_name, 0)
            cv2.fillPoly(seg_map, [hull], int(region_id))

        return seg_map

    def _approximate_groups(self, n: int) -> dict[str, tuple[int, int]]:
        """Create approximate landmark groups for short landmark lists."""
        # Split proportionally: ears 10%, eyes 20%, nose 15%, mouth 30%, rest
        return {
            "left_ear": (0, max(1, n // 10)),
            "right_ear": (max(1, n // 10), max(2, n // 5)),
            "left_eye": (max(2, n // 5), max(3, n * 3 // 10)),
            "right_eye": (max(3, n * 3 // 10), max(4, n * 2 // 5)),
            "nose": (max(4, n * 2 // 5), max(5, n * 11 // 20)),
            "mouth": (max(5, n * 11 // 20), n),
        }

    def get_mouth_region(self, seg_map: np.ndarray) -> np.ndarray:
        """Extract binary mouth mask from segmentation map.

        Args:
            seg_map: Semantic segmentation map (H, W) of region IDs.

        Returns:
            Binary mask (H, W), uint8, 255 where mouth, 0 elsewhere.
        """
        return ((seg_map == 6).astype(np.uint8)) * 255

    def visualize(self, image: np.ndarray, seg_map: np.ndarray) -> np.ndarray:
        """Overlay colored segmentation map on image for debugging.

        Args:
            image: RGB image (H, W, 3).
            seg_map: Segmentation map (H, W).

        Returns:
            RGB image with colored overlay (H, W, 3), uint8.
        """
        overlay = np.zeros_like(image)
        for region_id, color in _REGION_COLORS.items():
            mask = seg_map == region_id
            if region_id == 0:
                continue  # skip background
            overlay[mask] = color

        # Blend with original at 40% opacity for overlay regions
        alpha = 0.4
        result = image.copy()
        non_bg = seg_map > 0
        result[non_bg] = (
            image[non_bg].astype(np.float32) * (1 - alpha)
            + overlay[non_bg].astype(np.float32) * alpha
        ).astype(np.uint8)

        return result
