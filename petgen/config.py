"""Configuration management for PetGen."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class PetGenConfig(BaseSettings):
    """PetGen configuration loaded from environment variables."""

    model_config = {"env_prefix": "PETGEN_"}

    # API Keys
    gemini_api_key: str = ""
    fish_audio_api_key: str = ""
    elevenlabs_api_key: str = ""

    # Directories
    base_dir: Path = Path.home() / ".petgen"
    characters_dir: Path = Path.home() / ".petgen" / "characters"
    output_dir: Path = Path.home() / ".petgen" / "output"
    temp_dir: Path = Path.home() / ".petgen" / "tmp"
    weights_dir: Path = Path.home() / ".petgen" / "weights"
    voice_presets_dir: Path = Path.home() / ".petgen" / "voice_presets"

    # Gemini / Nano Banana defaults
    gemini_model: str = "gemini-3.1-flash-image"

    # Animation defaults
    default_fps: int = 25
    default_resolution_w: int = 1080
    default_resolution_h: int = 1920
    default_aspect_ratio: str = "9:16"
    mouth_amplitude_limit: float = 0.8

    # TTS defaults
    default_exaggeration: float = 0.7
    default_speed: float = 1.0
    tts_device: str = "cuda"

    # Processing
    device: str = "cuda"
    half_precision: bool = True

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [
            self.base_dir,
            self.characters_dir,
            self.output_dir,
            self.temp_dir,
            self.weights_dir,
            self.voice_presets_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


def get_config() -> PetGenConfig:
    """Load configuration from environment."""
    config = PetGenConfig()
    config.ensure_dirs()
    return config
