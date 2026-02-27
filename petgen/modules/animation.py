"""Lip sync animation using JoyVASA (audio→motion) + LivePortrait (motion→frames)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig
from petgen.models import AnimationConfig, MotionSequence


class LipSyncAnimator:
    """Audio-driven face animation for pets using JoyVASA + LivePortrait."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._joyvasa = None
        self._liveportrait = None

    def generate_motion(
        self,
        audio_path: Path,
        source_image: Path,
    ) -> MotionSequence:
        """Generate motion sequences from audio using JoyVASA diffusion transformer.

        Uses wav2vec2 for audio features → diffusion transformer → identity-independent
        motion sequences compatible with LivePortrait.
        """
        raise NotImplementedError("TODO: implement in Task #4")

    def animate(
        self,
        source_image: Path,
        motion: MotionSequence,
        config: AnimationConfig | None = None,
    ) -> list[np.ndarray]:
        """Apply motion sequences to source image using LivePortrait animal mode.

        Uses implicit keypoint-based warping with SPADE generator.
        MUST use --no_flag_stitching for animals.
        """
        raise NotImplementedError("TODO: implement in Task #4")

    def animate_from_audio(
        self,
        source_image: Path,
        audio_path: Path,
        config: AnimationConfig | None = None,
    ) -> list[np.ndarray]:
        """End-to-end: audio + source image → animated frames."""
        motion = self.generate_motion(audio_path, source_image)
        if config and config.mouth_amplitude_limit < 1.0:
            motion = self._limit_mouth_amplitude(motion, config.mouth_amplitude_limit)
        return self.animate(source_image, motion, config)

    def _limit_mouth_amplitude(
        self,
        motion: MotionSequence,
        limit: float,
    ) -> MotionSequence:
        """Clamp mouth opening amplitude to reduce dark-blob artifacts."""
        raise NotImplementedError("TODO: implement in Task #4")
