"""High-quality alpha matting for fur boundaries using ViTMatte."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from petgen.config import PetGenConfig

logger = logging.getLogger(__name__)

# HuggingFace model identifier for ViTMatte
_VITMATTE_MODEL_ID = "hustvl/vitmatte-small-composition-1k"

# Trimap dilation kernel sizes
_TRIMAP_DILATE_FG = 5
_TRIMAP_DILATE_UNKNOWN = 25


class FurMatter:
    """High-quality alpha matting for fur boundaries using ViTMatte."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._processor = None
        self._model = None

    @property
    def vitmatte_available(self) -> bool:
        """Check whether ViTMatte dependencies (transformers) are importable."""
        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_model(self):
        """Lazy-load ViTMatte model and processor from HuggingFace."""
        if self._model is not None:
            return self._model, self._processor

        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            self._processor = VitMatteImageProcessor.from_pretrained(_VITMATTE_MODEL_ID)
            self._model = VitMatteForImageMatting.from_pretrained(_VITMATTE_MODEL_ID)

            if self.config.device != "cpu":
                import torch

                device = torch.device(self.config.device)
                self._model = self._model.to(device)

            self._model.eval()
            logger.info("ViTMatte loaded from %s", _VITMATTE_MODEL_ID)
            return self._model, self._processor
        except ImportError:
            raise ImportError(
                "transformers package not installed or ViTMatte not available. "
                "Install with: pip install transformers"
            )

    def generate_mattes(
        self,
        frames: list[np.ndarray],
        trimaps: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Generate per-frame alpha mattes for fur boundary blending.

        If trimaps not provided, generates them from edge detection.
        Falls back to Canny-edge approach if ViTMatte not available.

        Args:
            frames: List of RGB frames (H, W, 3) as uint8 arrays.
            trimaps: Optional per-frame trimaps (H, W) with values in {0, 128, 255}
                     representing definite-bg, unknown, and definite-fg.

        Returns:
            List of alpha mattes as float32 arrays in [0, 1], shape (H, W).
        """
        if not frames:
            return []

        if trimaps is None:
            trimaps = [self._generate_trimap(f) for f in frames]

        if not self.vitmatte_available:
            logger.info("ViTMatte not available -- falling back to edge-detection matting")
            return self._fallback_edge_mattes(frames)

        try:
            return self._vitmatte_mattes(frames, trimaps)
        except Exception as exc:
            logger.warning("ViTMatte inference failed (%s) -- falling back to edge detection", exc)
            return self._fallback_edge_mattes(frames)

    def _vitmatte_mattes(
        self,
        frames: list[np.ndarray],
        trimaps: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Run ViTMatte inference on each frame."""
        import torch
        from PIL import Image

        model, processor = self._load_model()
        device = next(model.parameters()).device
        mattes: list[np.ndarray] = []

        for i, (frame, trimap) in enumerate(zip(frames, trimaps)):
            pil_image = Image.fromarray(frame)

            # ViTMatte expects a trimap as a grayscale PIL image
            trimap_resized = trimap
            if trimap.shape[:2] != frame.shape[:2]:
                trimap_resized = cv2.resize(
                    trimap, (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            pil_trimap = Image.fromarray(trimap_resized.astype(np.uint8))

            inputs = processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                output = model(**inputs)

            alpha = output.alphas.squeeze().cpu().numpy()
            # Resize alpha back to frame dimensions if needed
            if alpha.shape[:2] != frame.shape[:2]:
                alpha = cv2.resize(
                    alpha, (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
            mattes.append(alpha)

        return mattes

    def _generate_trimap(self, frame: np.ndarray) -> np.ndarray:
        """Generate a trimap from a frame using edge detection and morphological ops.

        Produces a trimap with:
        - 255 = definite foreground (interior of subject)
        - 128 = unknown (boundary region for alpha estimation)
        - 0   = definite background

        Returns:
            Trimap array (H, W) with uint8 values in {0, 128, 255}.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to get a connected boundary
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(edges, kernel_small, iterations=3)

        # Flood fill from corners to find background
        filled = dilated.copy()
        h, w = filled.shape
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        foreground = cv2.bitwise_not(filled)

        # Combine with edges to get a solid foreground region
        foreground = cv2.bitwise_or(dilated, foreground)

        # Erode for definite foreground
        kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_TRIMAP_DILATE_FG, _TRIMAP_DILATE_FG))
        definite_fg = cv2.erode(foreground, kernel_fg, iterations=2)

        # Dilate for unknown region
        kernel_unk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_TRIMAP_DILATE_UNKNOWN, _TRIMAP_DILATE_UNKNOWN))
        expanded = cv2.dilate(foreground, kernel_unk, iterations=1)

        # Build trimap
        trimap = np.zeros((h, w), dtype=np.uint8)
        trimap[expanded > 0] = 128  # unknown
        trimap[definite_fg > 0] = 255  # definite foreground
        # 0 stays as definite background

        return trimap

    def _fallback_edge_mattes(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Fallback alpha matte generation using Canny edge + flood fill.

        Returns mattes as float32 in [0, 1].
        """
        mattes: list[np.ndarray] = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dilated = cv2.dilate(edges, kernel, iterations=3)

            filled = dilated.copy()
            h, w = filled.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(filled, flood_mask, (0, 0), 255)
            filled_inv = cv2.bitwise_not(filled)

            matte = cv2.bitwise_or(dilated, filled_inv)
            matte = cv2.GaussianBlur(matte, (21, 21), 0)

            # Normalize to [0, 1] float
            matte_f = matte.astype(np.float32) / 255.0
            mattes.append(matte_f)

        return mattes
