"""Compositing and post-processing — artifact repair, fur matting, deflickering, encoding."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from petgen.config import PetGenConfig
from petgen.utils.video import frames_to_video, mux_audio_video, resize_frames


class Compositor:
    """Post-processing pipeline for final video output."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config

    def repair_artifacts(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Fix warping artifacts around the animated mouth region.

        Prototype approach: Gaussian blur the mouth boundary region and alpha-blend
        it back for smooth transitions. Uses mouth_masks to identify the repair region.
        Later upgrade path: integrate ProPainter for dual-domain propagation.
        """
        if not frames or not mouth_masks:
            return list(frames)

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

            # Ensure mask matches frame dimensions
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Normalize mask to [0, 1] float
            if mask.dtype == np.uint8:
                mask_f = mask.astype(np.float32) / 255.0
            else:
                mask_f = mask.astype(np.float32)

            # Ensure mask is 2D
            if mask_f.ndim == 3:
                mask_f = mask_f[:, :, 0]

            # Dilate the mask to capture the boundary region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            dilated = cv2.dilate(mask_f, kernel, iterations=2)

            # The boundary region is the dilated mask minus the original
            boundary = np.clip(dilated - mask_f * 0.5, 0, 1)

            # Gaussian blur the frame
            blurred = cv2.GaussianBlur(frame, (21, 21), 0)

            # Blend: use blurred pixels in the boundary, original elsewhere
            boundary_3ch = boundary[:, :, np.newaxis]
            repaired = (
                frame.astype(np.float32) * (1 - boundary_3ch)
                + blurred.astype(np.float32) * boundary_3ch
            ).astype(np.uint8)

            result.append(repaired)

        return result

    def blend_fur_boundaries(
        self,
        frames: list[np.ndarray],
        alpha_mattes: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Blend animated region with static background using alpha mattes.

        Simple alpha compositing: alpha * animated + (1-alpha) * original.
        Matte edges are feathered with Gaussian blur for smooth transitions.
        """
        if not frames or not alpha_mattes:
            return list(frames)

        # Use the first frame as the "static background" reference
        background = frames[0].copy()
        result = []

        for i, frame in enumerate(frames):
            if i >= len(alpha_mattes):
                result.append(frame.copy())
                continue

            matte = alpha_mattes[i]
            h, w = frame.shape[:2]

            # Ensure matte matches frame dimensions
            if matte.shape[:2] != (h, w):
                matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)

            # Normalize to [0, 1] float
            if matte.dtype == np.uint8:
                alpha = matte.astype(np.float32) / 255.0
            else:
                alpha = matte.astype(np.float32)

            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]

            # Feather matte edges
            alpha_2d = alpha[:, :, 0]
            alpha_2d = cv2.GaussianBlur(alpha_2d, (11, 11), 0)
            alpha = alpha_2d[:, :, np.newaxis]

            # Alpha composite
            blended = (
                frame.astype(np.float32) * alpha
                + background.astype(np.float32) * (1 - alpha)
            ).astype(np.uint8)

            result.append(blended)

        return result

    def generate_alpha_mattes(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Generate per-frame alpha mattes for fur boundary blending.

        Prototype approach: edge detection + dilation + Gaussian blur for soft matte.
        Later upgrade: integrate ViTMatte for production quality.
        """
        if not frames:
            return []

        mattes = []
        for frame in frames:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)

            # Dilate edges to create a broader matte region
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            dilated = cv2.dilate(edges, kernel, iterations=3)

            # Fill the interior: flood fill from corners to find background,
            # then invert to get the subject matte
            filled = dilated.copy()
            h, w = filled.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill(filled, flood_mask, (0, 0), 255)
            filled_inv = cv2.bitwise_not(filled)

            # Combine with dilated edges
            matte = cv2.bitwise_or(dilated, filled_inv)

            # Gaussian blur for soft edges
            matte = cv2.GaussianBlur(matte, (21, 21), 0)

            mattes.append(matte)

        return mattes

    def deflicker(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Apply blind video deflickering for temporal consistency.

        Histogram matching between consecutive frames using a running average
        to smooth out brightness fluctuations.
        """
        if not frames or len(frames) < 2:
            return list(frames)

        result = [frames[0].copy()]

        # Compute running average of per-channel means
        running_mean = frames[0].astype(np.float64).mean(axis=(0, 1))
        alpha = 0.7  # smoothing factor — higher = more smoothing toward running avg

        for i in range(1, len(frames)):
            frame = frames[i].astype(np.float64)
            current_mean = frame.mean(axis=(0, 1))

            # Update running average
            running_mean = alpha * running_mean + (1 - alpha) * current_mean

            # Compute per-channel correction factors
            correction = np.where(
                current_mean > 0,
                running_mean / current_mean,
                1.0,
            )

            # Apply correction, clamp to [0, 255]
            corrected = frame * correction[np.newaxis, np.newaxis, :]
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            result.append(corrected)

        return result

    def encode_video(
        self,
        frames: list[np.ndarray],
        audio_path: Path,
        output_path: Path,
        fps: int = 25,
        resolution: tuple[int, int] = (1080, 1920),
        codec: str = "libx264",
        crf: int = 18,
    ) -> Path:
        """Encode frames + audio to final MP4 video via FFmpeg.

        Pipeline: frames -> resize if needed -> temp video -> mux with audio -> final MP4.
        """
        if not frames:
            raise ValueError("No frames to encode")

        # Resize if frames don't match target resolution
        target_w, target_h = resolution
        fh, fw = frames[0].shape[:2]
        if fw != target_w or fh != target_h:
            frames = resize_frames(frames, target_w, target_h)

        # Write frames to temp video (no audio)
        temp_dir = Path(tempfile.mkdtemp(prefix="petgen_"))
        temp_video = temp_dir / "temp_video.mp4"

        frames_to_video(frames, temp_video, fps=fps, codec=codec, crf=crf)

        # Mux audio with video
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mux_audio_video(temp_video, audio_path, output_path)

        # Clean up temp file
        temp_video.unlink(missing_ok=True)
        temp_dir.rmdir()

        return output_path

    def composite_full(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
        audio_path: Path,
        output_path: Path,
        fps: int = 25,
        resolution: tuple[int, int] = (1080, 1920),
        deflicker_enabled: bool = True,
    ) -> Path:
        """Full compositing pipeline: repair -> matte -> deflicker -> encode."""
        result = self.repair_artifacts(frames, mouth_masks)
        mattes = self.generate_alpha_mattes(result)
        result = self.blend_fur_boundaries(result, mattes)
        if deflicker_enabled:
            result = self.deflicker(result)
        return self.encode_video(result, audio_path, output_path, fps=fps, resolution=resolution)
