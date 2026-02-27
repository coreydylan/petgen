"""Voice generation module using Chatterbox Turbo TTS."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from petgen.config import PetGenConfig
from petgen.models import BreedGroup, VoiceSettings


@dataclass
class VoicePreset:
    """A pre-configured voice reference with settings."""

    name: str
    breed_group: BreedGroup
    reference_clip: Path
    settings: VoiceSettings
    description: str = ""


# Default voice presets mapped to breed groups
DEFAULT_PRESETS: dict[BreedGroup, VoiceSettings] = {
    BreedGroup.SMALL_DOG: VoiceSettings(preset_name="small_dog", exaggeration=0.8, speed=1.1),
    BreedGroup.MEDIUM_DOG: VoiceSettings(preset_name="medium_dog", exaggeration=0.6, speed=1.0),
    BreedGroup.LARGE_DOG: VoiceSettings(preset_name="large_dog", exaggeration=0.5, speed=0.9),
    BreedGroup.PUPPY: VoiceSettings(preset_name="puppy", exaggeration=0.9, speed=1.2),
    BreedGroup.CAT: VoiceSettings(preset_name="cat", exaggeration=0.4, speed=0.95),
    BreedGroup.KITTEN: VoiceSettings(preset_name="kitten", exaggeration=0.7, speed=1.1),
}


class VoiceGenerator:
    """Generates character voice audio from text scripts."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=self.config.tts_device)
        return self._model

    def generate(
        self,
        text: str,
        voice_ref: Path | None = None,
        exaggeration: float = 0.7,
        speed: float = 1.0,
        output_path: Path | None = None,
    ) -> Path:
        """Generate speech audio from text using Chatterbox Turbo."""
        raise NotImplementedError("TODO: implement in Task #3")

    def generate_with_preset(
        self,
        text: str,
        breed_group: BreedGroup,
        output_path: Path | None = None,
    ) -> Path:
        """Generate speech using breed-aware voice preset."""
        raise NotImplementedError("TODO: implement in Task #3")

    def list_presets(self) -> list[VoicePreset]:
        """List available voice presets."""
        raise NotImplementedError("TODO: implement in Task #3")

    def create_preset(
        self,
        name: str,
        breed_group: BreedGroup,
        reference_clip: Path,
        settings: VoiceSettings | None = None,
        description: str = "",
    ) -> VoicePreset:
        """Create a custom voice preset from a reference clip."""
        raise NotImplementedError("TODO: implement in Task #3")
