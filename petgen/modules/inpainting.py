"""Video inpainting using ProPainter for mouth artifact repair."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from petgen.config import PetGenConfig

logger = logging.getLogger(__name__)

# ProPainter inference chunk size (frames) to manage VRAM
_CHUNK_SIZE = 50


class ProPainterInpainter:
    """Video inpainting using ProPainter for mouth artifact repair."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._model = None

    @property
    def propainter_available(self) -> bool:
        """Check whether ProPainter weights are present on disk."""
        weights_path = self.config.weights_dir / "propainter"
        return weights_path.exists() and any(weights_path.iterdir())

    def _load_model(self):
        """Lazy-load the ProPainter model from weights directory."""
        if self._model is not None:
            return self._model

        weights_path = self.config.weights_dir / "propainter"
        if not self.propainter_available:
            raise FileNotFoundError(
                f"ProPainter weights not found at {weights_path}. "
                "Run scripts/setup_weights.sh to download them."
            )

        try:
            import torch
            from propainter.inference import InpaintGenerator

            device = torch.device(self.config.device)
            model = InpaintGenerator().to(device)
            ckpt = torch.load(
                weights_path / "ProPainter.pth",
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(ckpt, strict=True)
            model.eval()
            self._model = model
            logger.info("ProPainter model loaded on %s", device)
            return model
        except ImportError:
            raise ImportError(
                "ProPainter package not installed. "
                "Install with: pip install propainter"
            )

    def repair_mouth_artifacts(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
        flow_threshold: float = 0.5,
    ) -> list[np.ndarray]:
        """Use ProPainter's dual-domain propagation to fix warp artifacts.

        Falls back to Gaussian blur approach if ProPainter not available.

        Args:
            frames: List of RGB frames (H, W, 3) as uint8 arrays.
            mouth_masks: Per-frame binary masks of the mouth region.
            flow_threshold: Optical flow magnitude threshold for artifact detection.

        Returns:
            List of repaired frames with same shapes as input.
        """
        if not frames or not mouth_masks:
            return list(frames)

        if not self.propainter_available:
            logger.info("ProPainter not available — falling back to Gaussian blur repair")
            return self._fallback_gaussian_repair(frames, mouth_masks)

        try:
            return self._propainter_repair(frames, mouth_masks, flow_threshold)
        except Exception as exc:
            logger.warning("ProPainter inference failed (%s) — falling back to Gaussian blur", exc)
            return self._fallback_gaussian_repair(frames, mouth_masks)

    def _propainter_repair(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
        flow_threshold: float,
    ) -> list[np.ndarray]:
        """Run ProPainter inpainting in chunks."""
        import torch

        model = self._load_model()
        device = next(model.parameters()).device

        result_frames: list[np.ndarray] = []

        for chunk_start in range(0, len(frames), _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, len(frames))
            chunk_frames = frames[chunk_start:chunk_end]
            chunk_masks = mouth_masks[chunk_start:chunk_end]

            # Compute optical flow between consecutive frames
            flows_forward, flows_backward = self._compute_optical_flows(chunk_frames)

            # Prepare tensors: (T, C, H, W)
            h, w = chunk_frames[0].shape[:2]
            frames_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                for f in chunk_frames
            ]).unsqueeze(0).to(device)  # (1, T, 3, H, W)

            masks_tensor = torch.stack([
                torch.from_numpy(self._normalize_mask(m, h, w)).float()
                for m in chunk_masks
            ]).unsqueeze(0).unsqueeze(2).to(device)  # (1, T, 1, H, W)

            flows_f_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1).float()
                for f in flows_forward
            ]).unsqueeze(0).to(device) if flows_forward else torch.zeros(1, max(len(chunk_frames) - 1, 1), 2, h, w).to(device)

            flows_b_tensor = torch.stack([
                torch.from_numpy(f).permute(2, 0, 1).float()
                for f in flows_backward
            ]).unsqueeze(0).to(device) if flows_backward else torch.zeros(1, max(len(chunk_frames) - 1, 1), 2, h, w).to(device)

            with torch.no_grad():
                output = model(frames_tensor, flows_f_tensor, flows_b_tensor, masks_tensor)

            # Convert back to numpy
            if isinstance(output, tuple):
                output = output[0]
            output = output.squeeze(0)  # (T, 3, H, W)
            for t in range(output.shape[0]):
                frame_out = (output[t].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                result_frames.append(frame_out)

        return result_frames

    def _compute_optical_flows(
        self, frames: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute forward and backward optical flow between consecutive frames.

        Returns:
            Tuple of (forward_flows, backward_flows), each a list of (H, W, 2) arrays.
        """
        if len(frames) < 2:
            return [], []

        forward_flows: list[np.ndarray] = []
        backward_flows: list[np.ndarray] = []

        for i in range(len(frames) - 1):
            gray_prev = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
            gray_next = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)

            flow_fwd = cv2.calcOpticalFlowFarneback(
                gray_prev, gray_next, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            flow_bwd = cv2.calcOpticalFlowFarneback(
                gray_next, gray_prev, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )

            forward_flows.append(flow_fwd)
            backward_flows.append(flow_bwd)

        return forward_flows, backward_flows

    def _fallback_gaussian_repair(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Gaussian blur boundary smoothing as fallback when ProPainter is unavailable.

        This mirrors the existing Compositor.repair_artifacts approach.
        """
        result = []
        for i, frame in enumerate(frames):
            if i >= len(mouth_masks):
                result.append(frame.copy())
                continue

            mask = mouth_masks[i]
            if mask.max() == 0:
                result.append(frame.copy())
                continue

            h, w = frame.shape[:2]

            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            if mask.dtype == np.uint8:
                mask_f = mask.astype(np.float32) / 255.0
            else:
                mask_f = mask.astype(np.float32)

            if mask_f.ndim == 3:
                mask_f = mask_f[:, :, 0]

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(mask_f, kernel, iterations=2)
            boundary = np.clip(dilated - mask_f * 0.5, 0, 1)

            blurred = cv2.GaussianBlur(frame, (21, 21), 0)
            boundary_3ch = boundary[:, :, np.newaxis]
            repaired = (
                frame.astype(np.float32) * (1 - boundary_3ch)
                + blurred.astype(np.float32) * boundary_3ch
            ).astype(np.uint8)

            result.append(repaired)

        return result

    @staticmethod
    def _normalize_mask(mask: np.ndarray, h: int, w: int) -> np.ndarray:
        """Normalize a mask to float [0, 1] with target dimensions."""
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if mask.dtype == np.uint8:
            mask = mask.astype(np.float32) / 255.0
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask
