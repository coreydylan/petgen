"""Compositing and post-processing — artifact repair, fur matting, deflickering, encoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig


class Compositor:
    """Post-processing pipeline for final video output."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config

    def repair_artifacts(
        self,
        frames: list[np.ndarray],
        mouth_masks: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Fix warping artifacts around the animated mouth using ProPainter.

        Dual-domain propagation + mask-guided sparse video transformer.
        """
        raise NotImplementedError("TODO: implement in Task #5")

    def blend_fur_boundaries(
        self,
        frames: list[np.ndarray],
        alpha_mattes: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Blend animated region with static background using ViTMatte alpha mattes."""
        raise NotImplementedError("TODO: implement in Task #5")

    def generate_alpha_mattes(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Generate per-frame alpha mattes for fur boundary blending."""
        raise NotImplementedError("TODO: implement in Task #5")

    def deflicker(
        self,
        frames: list[np.ndarray],
    ) -> list[np.ndarray]:
        """Apply blind video deflickering for temporal consistency."""
        raise NotImplementedError("TODO: implement in Task #5")

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
        """Encode frames + audio to final MP4 video via FFmpeg."""
        raise NotImplementedError("TODO: implement in Task #5")

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
        """Full compositing pipeline: repair → matte → deflicker → encode."""
        result = self.repair_artifacts(frames, mouth_masks)
        mattes = self.generate_alpha_mattes(result)
        result = self.blend_fur_boundaries(result, mattes)
        if deflicker_enabled:
            result = self.deflicker(result)
        return self.encode_video(result, audio_path, output_path, fps=fps, resolution=resolution)
