"""Secondary motion system — breathing, ear twitches, blinks, head bob, tail wag."""

from __future__ import annotations

import numpy as np

from petgen.models import AnimationConfig, BreedGroup, Landmark


class SecondaryMotion:
    """Applies procedural secondary motion to animated frames for realism."""

    def apply_breathing(
        self,
        frames: list[np.ndarray],
        bpm: float = 15.0,
        amplitude: float = 0.02,
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply sinusoidal breathing motion to chest/torso region.

        Dogs: 12-18 BPM, Cats: 20-30 BPM. Amplitude 1-3% scale oscillation.
        """
        raise NotImplementedError("TODO: implement in Task #5")

    def apply_ear_twitches(
        self,
        frames: list[np.ndarray],
        audio_energy: np.ndarray,
        landmarks: list[list[Landmark]],
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply ear twitches synchronized to speech emphasis."""
        raise NotImplementedError("TODO: implement in Task #5")

    def apply_head_bob(
        self,
        frames: list[np.ndarray],
        audio_pitch: np.ndarray,
        amplitude: float = 0.01,
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply gentle head nodding on stressed syllables."""
        raise NotImplementedError("TODO: implement in Task #5")

    def apply_blinks(
        self,
        frames: list[np.ndarray],
        landmarks: list[list[Landmark]],
        interval_range: tuple[float, float] = (3.0, 6.0),
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply random blinks every 3-6 seconds.

        Quick close (2-3 frames) + slower open (4-5 frames).
        """
        raise NotImplementedError("TODO: implement in Task #5")

    def apply_tail_wag(
        self,
        frames: list[np.ndarray],
        excitement: float = 0.5,
        fps: float = 25.0,
    ) -> list[np.ndarray]:
        """Apply procedural pendulum tail motion if tail is visible."""
        raise NotImplementedError("TODO: implement in Task #5")

    def apply_all(
        self,
        frames: list[np.ndarray],
        audio_features: dict[str, np.ndarray],
        landmarks: list[list[Landmark]],
        config: AnimationConfig,
        breed_group: BreedGroup = BreedGroup.MEDIUM_DOG,
    ) -> list[np.ndarray]:
        """Apply all enabled secondary motion effects."""
        fps = float(config.fps)

        # Determine breathing rate from breed
        if config.breathing_bpm:
            bpm = config.breathing_bpm
        elif breed_group in (BreedGroup.CAT, BreedGroup.KITTEN):
            bpm = 25.0
        else:
            bpm = 15.0

        result = frames
        result = self.apply_breathing(result, bpm=bpm, fps=fps)

        if config.enable_blinks:
            result = self.apply_blinks(
                result, landmarks, interval_range=config.blink_interval, fps=fps
            )

        if config.enable_ear_twitches and "energy" in audio_features:
            result = self.apply_ear_twitches(result, audio_features["energy"], landmarks, fps=fps)

        if config.enable_head_bob and "pitch" in audio_features:
            result = self.apply_head_bob(result, audio_features["pitch"], fps=fps)

        if config.enable_tail_wag:
            excitement = audio_features.get("mean_energy", np.array([0.5])).mean()
            result = self.apply_tail_wag(result, excitement=float(excitement), fps=fps)

        return result
