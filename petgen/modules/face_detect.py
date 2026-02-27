"""Pet face detection and landmark extraction using YOLOv8 + CatFLW/DogFLW + SAM2."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig
from petgen.models import BoundingBox, Landmark

logger = logging.getLogger(__name__)

# YOLO class IDs for pets (COCO dataset)
_YOLO_CAT_CLASS = 15
_YOLO_DOG_CLASS = 16

# Landmark index ranges for mouth/jaw region.
# CatFLW uses 48 landmarks; mouth/jaw are the last 15 (indices 33-47).
# DogFLW uses 46 landmarks; mouth/jaw are the last 13 (indices 33-45).
_CAT_MOUTH_START = 33
_DOG_MOUTH_START = 33
_CAT_NUM_LANDMARKS = 48
_DOG_NUM_LANDMARKS = 46


class PetFaceDetector:
    """Detects pet faces, extracts landmarks, and segments mouth regions."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._yolo_model = None
        self._landmark_model = None
        self._sam2_predictor = None
        self._sam2_video_predictor = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_yolo(self):
        """Lazy-load YOLOv8 model for pet face detection."""
        if self._yolo_model is not None:
            return self._yolo_model

        from ultralytics import YOLO

        weights_path = self.config.weights_dir / "yolov8n.pt"
        if weights_path.exists():
            self._yolo_model = YOLO(str(weights_path))
        else:
            # Download default model; ultralytics handles caching
            self._yolo_model = YOLO("yolov8n.pt")
        logger.info("YOLOv8 model loaded on device=%s", self.config.device)
        return self._yolo_model

    def _load_landmark_model(self):
        """Lazy-load a lightweight CNN landmark detector.

        For the prototype we use a simple regression CNN that maps a cropped
        face region to N landmark coordinates.  Weights are expected at
        ``config.weights_dir / 'pet_landmarks.pt'``.  If unavailable we fall
        back to a heuristic grid estimator so the rest of the pipeline can
        still run.
        """
        if self._landmark_model is not None:
            return self._landmark_model

        weights_path = self.config.weights_dir / "pet_landmarks.pt"
        if weights_path.exists():
            import torch

            self._landmark_model = torch.jit.load(
                str(weights_path),
                map_location=self.config.device,
            )
            self._landmark_model.eval()
            logger.info("Landmark model loaded from %s", weights_path)
        else:
            logger.warning(
                "Landmark weights not found at %s — using heuristic fallback",
                weights_path,
            )
            self._landmark_model = None
        return self._landmark_model

    def _load_sam2(self):
        """Lazy-load SAM2 image predictor for mouth segmentation."""
        if self._sam2_predictor is not None:
            return self._sam2_predictor

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = self.config.weights_dir / "sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        sam2_model = build_sam2(model_cfg, str(checkpoint), device=self.config.device)
        self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        logger.info("SAM2 image predictor loaded")
        return self._sam2_predictor

    def _load_sam2_video(self):
        """Lazy-load SAM2 video predictor for temporal mouth tracking."""
        if self._sam2_video_predictor is not None:
            return self._sam2_video_predictor

        from sam2.build_sam import build_sam2_video_predictor

        checkpoint = self.config.weights_dir / "sam2_hiera_small.pt"
        model_cfg = "sam2_hiera_s.yaml"
        self._sam2_video_predictor = build_sam2_video_predictor(
            model_cfg, str(checkpoint), device=self.config.device
        )
        logger.info("SAM2 video predictor loaded")
        return self._sam2_video_predictor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_face(self, image: np.ndarray) -> BoundingBox | None:
        """Detect the primary pet face in an image using YOLOv8.

        Filters detections to cat/dog classes and returns the highest-confidence
        bounding box, or ``None`` when no pet face is found.
        """
        model = self._load_yolo()
        results = model.predict(
            source=image,
            device=self.config.device,
            verbose=False,
            conf=0.25,
        )

        if not results or len(results) == 0:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        # Filter to pet classes only
        best_box = None
        best_conf = 0.0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            if cls_id in (_YOLO_CAT_CLASS, _YOLO_DOG_CLASS) and conf > best_conf:
                best_conf = conf
                coords = boxes.xyxy[i].cpu().numpy()
                best_box = BoundingBox(
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                    confidence=conf,
                )

        return best_box

    def detect_landmarks(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
    ) -> list[Landmark]:
        """Extract facial landmarks (46-48 points) from a detected face region.

        Crops the face region defined by *bbox*, resizes to the landmark model's
        expected input, and regresses N landmark coordinates.  Falls back to a
        heuristic grid if no trained model is available.
        """
        model = self._load_landmark_model()
        species = self._species_from_last_detection()
        num_landmarks = _CAT_NUM_LANDMARKS if species == "cat" else _DOG_NUM_LANDMARKS

        # Crop face region with a small padding
        h, w = image.shape[:2]
        pad = 0.1
        x1 = max(0, int(bbox.x1 - bbox.width * pad))
        y1 = max(0, int(bbox.y1 - bbox.height * pad))
        x2 = min(w, int(bbox.x2 + bbox.width * pad))
        y2 = min(h, int(bbox.y2 + bbox.height * pad))
        crop = image[y1:y2, x1:x2]

        if model is not None:
            return self._predict_landmarks(model, crop, x1, y1, x2, y2, num_landmarks)

        # Heuristic fallback: distribute points in a grid within the bounding box
        return self._heuristic_landmarks(bbox, num_landmarks)

    def segment_mouth(
        self,
        image: np.ndarray,
        landmarks: list[Landmark],
    ) -> np.ndarray:
        """Generate a binary mask of the mouth region using SAM2.

        Computes a tight bounding box around mouth landmarks, then uses SAM2's
        image predictor with that box as a prompt to produce a pixel-precise mask.
        """
        mouth_lms = self.extract_mouth_landmarks(landmarks)
        if not mouth_lms:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        predictor = self._load_sam2()

        # Compute mouth bounding box
        xs = [lm.x for lm in mouth_lms]
        ys = [lm.y for lm in mouth_lms]
        pad = 10
        mouth_box = np.array([
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(image.shape[1], max(xs) + pad),
            min(image.shape[0], max(ys) + pad),
        ])

        predictor.set_image(image)
        masks, scores, _ = predictor.predict(
            box=mouth_box[None, :],
            multimask_output=False,
        )

        # Return the best mask as uint8
        mask = (masks[0] > 0.5).astype(np.uint8) * 255
        return mask

    def track_mouth_video(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Track and segment mouth across video frames using SAM2 Video Predictor.

        Detects the mouth in the first frame, then propagates the segmentation
        mask forward through the remaining frames for temporal consistency.
        """
        if not frames:
            return []

        # Detect face and landmarks in the first frame
        first = frames[0]
        bbox = self.detect_face(first)
        if bbox is None:
            return [np.zeros(first.shape[:2], dtype=np.uint8)] * len(frames)

        landmarks = self.detect_landmarks(first, bbox)
        mouth_lms = self.extract_mouth_landmarks(landmarks)

        if not mouth_lms:
            return [np.zeros(first.shape[:2], dtype=np.uint8)] * len(frames)

        # Compute initial mouth box prompt
        xs = [lm.x for lm in mouth_lms]
        ys = [lm.y for lm in mouth_lms]
        pad = 10
        mouth_box = np.array([
            max(0, min(xs) - pad),
            max(0, min(ys) - pad),
            min(first.shape[1], max(xs) + pad),
            min(first.shape[0], max(ys) + pad),
        ])

        video_predictor = self._load_sam2_video()
        state = video_predictor.init_state(frames=frames)

        # Add mouth prompt on frame 0
        video_predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=0,
            obj_id=0,
            box=mouth_box,
        )

        # Propagate masks across the video
        masks_out: list[np.ndarray] = [np.zeros(first.shape[:2], dtype=np.uint8)] * len(frames)
        for frame_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(state):
            mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
            masks_out[frame_idx] = mask

        return masks_out

    def extract_mouth_landmarks(self, landmarks: list[Landmark]) -> list[Landmark]:
        """Extract just the mouth/jaw subset from full facial landmarks.

        CatFLW (48 points): mouth indices 33-47 (15 points).
        DogFLW (46 points): mouth indices 33-45 (13 points).
        For shorter landmark lists (e.g. test fixtures), returns the last third.
        """
        n = len(landmarks)
        if n >= _CAT_NUM_LANDMARKS:
            return landmarks[_CAT_MOUTH_START:]
        if n >= _DOG_NUM_LANDMARKS:
            return landmarks[_DOG_MOUTH_START:]
        # Fallback for short lists: return the last third
        start = max(0, n - n // 3)
        return landmarks[start:]

    def get_species(self, image: np.ndarray) -> str:
        """Classify whether the detected face is cat or dog using YOLOv8 class prediction."""
        model = self._load_yolo()
        results = model.predict(
            source=image,
            device=self.config.device,
            verbose=False,
            conf=0.25,
        )

        if not results or len(results) == 0:
            return "cat"  # default fallback

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return "cat"

        # Find highest-confidence pet detection and return its species
        best_cls = _YOLO_CAT_CLASS
        best_conf = 0.0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            if cls_id in (_YOLO_CAT_CLASS, _YOLO_DOG_CLASS) and conf > best_conf:
                best_conf = conf
                best_cls = cls_id

        return "cat" if best_cls == _YOLO_CAT_CLASS else "dog"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _species_from_last_detection(self) -> str:
        """Return species from cached YOLO state.  Falls back to 'cat'."""
        # In a full implementation this would cache the last detect_face result.
        # For now, default to cat (48 landmarks — superset of dog).
        return "cat"

    def _predict_landmarks(
        self,
        model,
        crop: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        num_landmarks: int,
    ) -> list[Landmark]:
        """Run the landmark regression model on a cropped face image."""
        import cv2
        import torch

        input_size = 256
        resized = cv2.resize(crop, (input_size, input_size))
        tensor = (
            torch.from_numpy(resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.config.device)
            / 255.0
        )

        with torch.no_grad():
            coords = model(tensor)  # (1, N*2) normalized [0,1]

        coords = coords.cpu().numpy().reshape(-1, 2)
        crop_h, crop_w = y2 - y1, x2 - x1

        landmarks = []
        for cx, cy in coords[:num_landmarks]:
            landmarks.append(Landmark(
                x=float(cx * crop_w + x1),
                y=float(cy * crop_h + y1),
                visibility=1.0,
            ))
        return landmarks

    def _heuristic_landmarks(
        self,
        bbox: BoundingBox,
        num_landmarks: int,
    ) -> list[Landmark]:
        """Generate a heuristic grid of landmark points within a bounding box.

        This allows the rest of the pipeline to function when no trained
        landmark model is available.
        """
        cx, cy = bbox.center
        w, h = bbox.width, bbox.height

        landmarks = []
        # Distribute points in concentric ellipses centred on the face
        for i in range(num_landmarks):
            angle = 2 * np.pi * i / num_landmarks
            # Radial distance grows for later (mouth/jaw) landmarks
            r = 0.2 + 0.15 * (i / num_landmarks)
            lx = cx + w * r * np.cos(angle)
            ly = cy + h * r * np.sin(angle)
            landmarks.append(Landmark(x=float(lx), y=float(ly), visibility=0.5))
        return landmarks
