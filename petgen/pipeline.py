"""End-to-end pipeline orchestrator — chains all modules into a single workflow."""

from __future__ import annotations

from pathlib import Path

from petgen.config import PetGenConfig
from petgen.models import AnimationConfig, CharacterProfile, PipelineResult


class PetGenPipeline:
    """Orchestrates the full PetGen pipeline: photos + script → talking pet video."""

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
        """Run the full pipeline: character + script → talking pet video.

        Steps:
        1. Generate scene image (if scene_prompt provided, else use canonical front pose)
        2. Generate voice audio from script
        3. Detect face and extract landmarks from scene image
        4. Generate lip-sync animation from audio + scene image
        5. Apply secondary motion (breathing, blinks, ears, head bob, tail)
        6. Composite and encode final video
        """
        raise NotImplementedError("TODO: implement in Task #6")

    def create_character(
        self,
        name: str,
        photos: list[Path],
        breed: str | None = None,
    ) -> CharacterProfile:
        """Create a new character from reference photos."""
        raise NotImplementedError("TODO: implement in Task #6")
