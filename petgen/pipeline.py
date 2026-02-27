"""End-to-end pipeline orchestrator — chains all modules into a single workflow."""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig
from petgen.models import (
    AnimationConfig,
    AspectRatio,
    CharacterProfile,
    PipelineResult,
)

logger = logging.getLogger(__name__)

# Aspect ratio to (width, height) at 1080p
_ASPECT_RESOLUTIONS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.VERTICAL: (1080, 1920),
    AspectRatio.SQUARE: (1080, 1080),
    AspectRatio.LANDSCAPE: (1920, 1080),
}


class PetGenPipeline:
    """Orchestrates the full PetGen pipeline: photos + script -> talking pet video."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._character_creator = None
        self._voice_generator = None
        self._face_detector = None
        self._animator = None
        self._motion = None
        self._compositor = None

    @property
    def character_creator(self):
        if self._character_creator is None:
            from petgen.modules.character import CharacterCreator

            self._character_creator = CharacterCreator(self.config)
        return self._character_creator

    @property
    def voice_generator(self):
        if self._voice_generator is None:
            from petgen.modules.voice import VoiceGenerator

            self._voice_generator = VoiceGenerator(self.config)
        return self._voice_generator

    @property
    def face_detector(self):
        if self._face_detector is None:
            from petgen.modules.face_detect import PetFaceDetector

            self._face_detector = PetFaceDetector(self.config)
        return self._face_detector

    @property
    def animator(self):
        if self._animator is None:
            from petgen.modules.animation import LipSyncAnimator

            self._animator = LipSyncAnimator(self.config)
        return self._animator

    @property
    def motion(self):
        if self._motion is None:
            from petgen.modules.motion import SecondaryMotion

            self._motion = SecondaryMotion()
        return self._motion

    @property
    def compositor(self):
        if self._compositor is None:
            from petgen.modules.compositing import Compositor

            self._compositor = Compositor(self.config)
        return self._compositor

    def generate(
        self,
        character: CharacterProfile,
        script: str,
        scene_prompt: str | None = None,
        animation_config: AnimationConfig | None = None,
        output_path: Path | None = None,
    ) -> PipelineResult:
        """Run the full pipeline: character + script -> talking pet video.

        Steps:
        1. Generate or select scene image
        2. Generate voice audio from script
        3. Detect face and extract landmarks
        4. Generate lip-sync animation
        5. Apply secondary motion
        6. Composite and encode final video
        """
        t0 = time.time()
        config = animation_config or AnimationConfig()

        if output_path is None:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.config.output_dir / f"petgen_{uuid.uuid4().hex[:8]}.mp4"

        # --- Step 1: Scene image ---
        logger.info("[1/6] Preparing scene image...")
        if scene_prompt:
            scene_image = self.character_creator.generate_scene(
                character, scene_prompt
            )
        elif "front" in character.canonical_poses:
            scene_image = character.canonical_poses["front"]
        elif character.reference_images:
            scene_image = character.reference_images[0]
        else:
            raise ValueError(
                f"Character '{character.name}' has no reference images or canonical poses."
            )
        logger.info("  Scene image: %s", scene_image)

        # --- Step 2: Voice generation ---
        logger.info("[2/6] Generating voice audio...")
        audio_path = self.voice_generator.generate(
            text=script,
            voice_ref=character.voice.reference_clip,
            exaggeration=character.voice.exaggeration,
            speed=character.voice.speed,
        )
        logger.info("  Audio: %s", audio_path)

        # --- Step 3: Face detection + landmarks ---
        logger.info("[3/6] Detecting face and landmarks...")
        from petgen.utils.image import load_image

        scene_array = load_image(scene_image)
        bbox = self.face_detector.detect_face(scene_array)
        if bbox is None:
            logger.warning("No pet face detected — using full image as animation source")
            from petgen.models import BoundingBox

            h, w = scene_array.shape[:2]
            bbox = BoundingBox(x1=0, y1=0, x2=w, y2=h, confidence=0.0)

        landmarks = self.face_detector.detect_landmarks(scene_array, bbox)
        logger.info("  Detected %d landmarks", len(landmarks))

        # --- Step 4: Lip-sync animation ---
        logger.info("[4/6] Generating lip-sync animation...")
        config_with_limit = AnimationConfig(
            fps=config.fps,
            resolution=config.resolution,
            aspect_ratio=config.aspect_ratio,
            mouth_amplitude_limit=self.config.mouth_amplitude_limit,
            breathing_bpm=config.breathing_bpm,
            enable_ear_twitches=config.enable_ear_twitches,
            enable_head_bob=config.enable_head_bob,
            enable_blinks=config.enable_blinks,
            enable_tail_wag=config.enable_tail_wag,
            blink_interval=config.blink_interval,
            deflicker=config.deflicker,
        )

        frames = self.animator.animate_from_audio(
            source_image=scene_image,
            audio_path=audio_path,
            config=config_with_limit,
        )
        logger.info("  Generated %d frames", len(frames))

        # --- Step 5: Secondary motion ---
        logger.info("[5/6] Applying secondary motion...")
        audio_features = self._extract_audio_features(audio_path, len(frames), config.fps)
        per_frame_landmarks = [landmarks] * len(frames)
        frames = self.motion.apply_all(
            frames=frames,
            audio_features=audio_features,
            landmarks=per_frame_landmarks,
            config=config_with_limit,
            breed_group=character.breed_group,
        )

        # --- Step 6: Compositing + encoding ---
        logger.info("[6/6] Compositing and encoding...")
        mouth_masks = self.face_detector.track_mouth_video(frames)
        final_path = self.compositor.composite_full(
            frames=frames,
            mouth_masks=mouth_masks,
            audio_path=audio_path,
            output_path=output_path,
            fps=config.fps,
            resolution=config.resolution,
            deflicker_enabled=config.deflicker,
        )

        elapsed = time.time() - t0
        logger.info("Pipeline complete in %.1fs -> %s", elapsed, final_path)

        return PipelineResult(
            video_path=final_path,
            audio_path=audio_path,
            character=character,
            scene_image=scene_image,
            duration=len(frames) / config.fps,
            num_frames=len(frames),
            metadata={
                "elapsed_seconds": round(elapsed, 2),
                "script": script,
                "scene_prompt": scene_prompt,
            },
        )

    def create_character(
        self,
        name: str,
        photos: list[Path],
        breed: str | None = None,
    ) -> CharacterProfile:
        """Create a new character from reference photos."""
        return self.character_creator.create_character(
            reference_images=photos,
            name=name,
            breed=breed,
        )

    def _extract_audio_features(
        self, audio_path: Path, num_frames: int, fps: int
    ) -> dict[str, np.ndarray]:
        """Extract audio energy and pitch features for secondary motion.

        Returns a dict with keys: energy, pitch, mean_energy
        Each value is a 1D array of length num_frames.
        """
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(audio_path))
            waveform = waveform.mean(dim=0).numpy()  # mono
        except Exception:
            # Fallback: generate silent features
            logger.warning("Could not load audio for feature extraction — using defaults")
            return {
                "energy": np.zeros(num_frames),
                "pitch": np.zeros(num_frames),
                "mean_energy": np.array([0.5]),
            }

        # Compute per-frame energy (RMS)
        samples_per_frame = max(1, len(waveform) // num_frames)
        energy = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(waveform))
            if start < len(waveform):
                chunk = waveform[start:end]
                energy[i] = np.sqrt(np.mean(chunk ** 2))

        # Simple pitch estimation via zero-crossing rate (rough but fast)
        pitch = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * samples_per_frame
            end = min(start + samples_per_frame, len(waveform))
            if start < len(waveform) and end - start > 1:
                chunk = waveform[start:end]
                zero_crossings = np.sum(np.abs(np.diff(np.sign(chunk)))) / 2
                pitch[i] = zero_crossings / (len(chunk) / sr) if len(chunk) > 0 else 0

        return {
            "energy": energy,
            "pitch": pitch,
            "mean_energy": np.array([energy.mean()]) if energy.mean() > 0 else np.array([0.5]),
        }
