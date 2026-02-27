"""MuseTalk animal adaptation layer.

Replaces human-specific components in MuseTalk with animal equivalents:
- Face detection: human detector -> PetFaceDetector (YOLOv8)
- Face parsing: BiSeNet -> AnimalFaceParser (landmark convex hulls)
- Keeps: wav2vec2 audio encoder, latent inpainting decoder
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig

logger = logging.getLogger(__name__)


class MuseTalkAnimalAdapter:
    """Adapter layer for running MuseTalk with animal faces.

    Replaces:
    - Face detection: human -> animal (PetFaceDetector)
    - Face parsing: BiSeNet -> AnimalFaceParser
    - Keeps: wav2vec2 audio encoder, latent inpainting decoder
    """

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self.face_detector = None
        self.face_parser = None
        self._audio_encoder = None
        self._inpainter = None
        self._model_loaded = False

    def _ensure_detector(self):
        """Lazy-load PetFaceDetector."""
        if self.face_detector is None:
            from petgen.modules.face_detect import PetFaceDetector

            self.face_detector = PetFaceDetector(self.config)

    def _ensure_parser(self):
        """Lazy-load AnimalFaceParser."""
        if self.face_parser is None:
            from petgen.research.face_parser import AnimalFaceParser

            self.face_parser = AnimalFaceParser(self.config)

    def _load_audio_encoder(self):
        """Load wav2vec2 audio encoder for feature extraction.

        Uses the same model path as JoyVASA for consistency.
        Falls back gracefully if weights are not available.
        """
        if self._audio_encoder is not None:
            return self._audio_encoder

        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor

            wav2vec_path = self.config.weights_dir / "joyvasa" / "wav2vec2"
            if wav2vec_path.exists():
                processor = Wav2Vec2Processor.from_pretrained(str(wav2vec_path))
                model = Wav2Vec2Model.from_pretrained(str(wav2vec_path))
            else:
                processor = Wav2Vec2Processor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                model = Wav2Vec2Model.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )

            model.eval()
            device = self.config.device
            if device != "cpu":
                model = model.to(device)

            self._audio_encoder = {"processor": processor, "model": model}
            logger.info("wav2vec2 audio encoder loaded")
        except ImportError:
            logger.warning("transformers not available; audio encoding disabled")
            self._audio_encoder = None
        except Exception as e:
            logger.warning("Failed to load audio encoder: %s", e)
            self._audio_encoder = None

        return self._audio_encoder

    def _load_inpainter(self):
        """Load the VAE-based inpainting model.

        NOTE: This is a scaffold. Actual model requires trained weights
        specific to animal faces. Returns None if not available.
        """
        if self._inpainter is not None:
            return self._inpainter

        inpainter_path = self.config.weights_dir / "musetalk_animal" / "inpainter.pt"
        if not inpainter_path.exists():
            logger.info(
                "MuseTalk animal inpainter weights not found at %s — "
                "using passthrough mode",
                inpainter_path,
            )
            self._model_loaded = False
            return None

        try:
            import torch

            self._inpainter = torch.jit.load(
                str(inpainter_path), map_location=self.config.device
            )
            self._inpainter.eval()
            self._model_loaded = True
            logger.info("MuseTalk animal inpainter loaded from %s", inpainter_path)
        except Exception as e:
            logger.warning("Failed to load inpainter: %s", e)
            self._model_loaded = False

        return self._inpainter

    def preprocess_frame(self, frame: np.ndarray) -> dict:
        """Detect face, parse regions, prepare input tensors.

        Args:
            frame: RGB image (H, W, 3), uint8.

        Returns:
            Dict with: face_crop, face_mask, mouth_mask, landmarks, bbox.
            Returns empty dict if no face detected.
        """
        self._ensure_detector()
        self._ensure_parser()

        bbox = self.face_detector.detect_face(frame)
        if bbox is None:
            return {}

        landmarks = self.face_detector.detect_landmarks(frame, bbox)
        seg_map = self.face_parser.parse(frame, landmarks)
        mouth_mask = self.face_parser.get_mouth_region(seg_map)

        # Crop face region with padding
        h, w = frame.shape[:2]
        pad = 0.15
        x1 = max(0, int(bbox.x1 - bbox.width * pad))
        y1 = max(0, int(bbox.y1 - bbox.height * pad))
        x2 = min(w, int(bbox.x2 + bbox.width * pad))
        y2 = min(h, int(bbox.y2 + bbox.height * pad))

        face_crop = frame[y1:y2, x1:x2]
        face_mask = (seg_map[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
        mouth_crop = mouth_mask[y1:y2, x1:x2]

        return {
            "face_crop": face_crop,
            "face_mask": face_mask,
            "mouth_mask": mouth_crop,
            "landmarks": landmarks,
            "bbox": bbox,
            "crop_coords": (x1, y1, x2, y2),
        }

    def encode_audio(self, audio_path: Path) -> np.ndarray:
        """Extract wav2vec2 features from audio.

        Uses the same wav2vec2 model as JoyVASA for consistency.

        Args:
            audio_path: Path to WAV audio file.

        Returns:
            Audio feature array (T, D) where T is the number of time steps
            and D is the feature dimension (768 for wav2vec2-base).
            Returns empty array if encoder not available.
        """
        encoder = self._load_audio_encoder()
        if encoder is None:
            logger.warning("Audio encoder not available, returning empty features")
            return np.array([], dtype=np.float32).reshape(0, 768)

        import torch
        from scipy.io import wavfile as scipy_wav
        from scipy.signal import resample_poly

        sr, audio_data = scipy_wav.read(str(audio_path))

        # Normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        # Resample to 16kHz
        if sr != 16000:
            audio_data = resample_poly(audio_data, 16000, sr).astype(np.float32)

        processor = encoder["processor"]
        model = encoder["model"]
        device = next(model.parameters()).device

        inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            features = model(input_values).last_hidden_state

        return features.cpu().numpy()[0]  # (T, 768)

    def inpaint_mouth(
        self,
        face_crop: np.ndarray,
        mouth_mask: np.ndarray,
        audio_features: np.ndarray,
        frame_idx: int,
    ) -> np.ndarray:
        """Run latent-space inpainting on the mouth region.

        This is the core MuseTalk operation:
        1. Encode face_crop to latent space (VAE encoder)
        2. Mask the mouth region in latent space
        3. Condition on audio features
        4. Decode inpainted latent (VAE decoder)

        NOTE: This is a scaffold -- actual inference requires trained weights.
        Returns the input face_crop unchanged if no weights available.

        Args:
            face_crop: Cropped face image (H, W, 3), uint8.
            mouth_mask: Binary mouth mask (H, W), uint8.
            audio_features: Audio features for this frame (D,) or (1, D).
            frame_idx: Frame index (for temporal conditioning).

        Returns:
            Inpainted face crop (H, W, 3), uint8.
        """
        inpainter = self._load_inpainter()
        if inpainter is None:
            # Passthrough: return the original face crop
            return face_crop.copy()

        import torch

        device = self.config.device

        # Prepare inputs
        face_tensor = (
            torch.from_numpy(face_crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ).to(device)
        mask_tensor = (
            torch.from_numpy(mouth_mask).float().unsqueeze(0).unsqueeze(0) / 255.0
        ).to(device)

        if audio_features.ndim == 1:
            audio_features = audio_features[np.newaxis, :]
        audio_tensor = torch.from_numpy(audio_features).float().to(device)

        with torch.no_grad():
            output = inpainter(face_tensor, mask_tensor, audio_tensor)

        result = (
            output[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        )
        return result

    def generate_frames(
        self, source_image: np.ndarray, audio_path: Path, fps: int = 25
    ) -> list[np.ndarray]:
        """Full pipeline: audio -> per-frame mouth inpainting.

        Falls back to passthrough (repeated source image) if no trained
        model is available.

        Args:
            source_image: Source pet image (H, W, 3), uint8, RGB.
            audio_path: Path to audio WAV file.
            fps: Target frames per second.

        Returns:
            List of RGB frames (H, W, 3), uint8.
        """
        # Encode audio
        audio_features = self.encode_audio(audio_path)

        # Preprocess source frame
        preprocessed = self.preprocess_frame(source_image)
        if not preprocessed:
            logger.warning("No face detected in source image, returning static frames")
            num_frames = max(1, len(audio_features))
            return [source_image.copy() for _ in range(num_frames)]

        face_crop = preprocessed["face_crop"]
        mouth_mask = preprocessed["mouth_mask"]
        bbox = preprocessed["bbox"]
        x1, y1, x2, y2 = preprocessed["crop_coords"]

        num_frames = len(audio_features) if len(audio_features) > 0 else fps
        frames = []

        for t in range(num_frames):
            feat = audio_features[t] if t < len(audio_features) else np.zeros(768)
            inpainted_crop = self.inpaint_mouth(face_crop, mouth_mask, feat, t)

            # Composite back into full frame
            output = source_image.copy()
            # Resize inpainted crop back to original crop dimensions
            crop_h, crop_w = y2 - y1, x2 - x1
            if inpainted_crop.shape[:2] != (crop_h, crop_w):
                import cv2

                inpainted_crop = cv2.resize(
                    inpainted_crop, (crop_w, crop_h)
                )
            output[y1:y2, x1:x2] = inpainted_crop
            frames.append(output)

        return frames
