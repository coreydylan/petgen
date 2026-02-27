"""Pet face detection and landmark extraction using YOLOv8 + CatFLW/DogFLW + SAM2."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig
from petgen.models import BoundingBox, Landmark


class PetFaceDetector:
    """Detects pet faces, extracts landmarks, and segments mouth regions."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._yolo_model = None
        self._landmark_model = None
        self._sam2_predictor = None

    def detect_face(self, image: np.ndarray) -> BoundingBox | None:
        """Detect the primary pet face in an image using YOLOv8."""
        raise NotImplementedError("TODO: implement in Task #4")

    def detect_landmarks(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> list[Landmark]:
        """Extract facial landmarks (46-48 points) from a detected face region."""
        raise NotImplementedError("TODO: implement in Task #4")

    def segment_mouth(
        self,
        image: np.ndarray,
        landmarks: list[Landmark],
    ) -> np.ndarray:
        """Generate a binary mask of the mouth region using SAM2."""
        raise NotImplementedError("TODO: implement in Task #4")

    def track_mouth_video(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Track and segment mouth across video frames using SAM2 Video Predictor."""
        raise NotImplementedError("TODO: implement in Task #4")

    def extract_mouth_landmarks(self, landmarks: list[Landmark]) -> list[Landmark]:
        """Extract just the mouth/jaw subset from full facial landmarks."""
        raise NotImplementedError("TODO: implement in Task #4")

    def get_species(self, image: np.ndarray) -> str:
        """Classify whether the detected face is cat or dog."""
        raise NotImplementedError("TODO: implement in Task #4")
