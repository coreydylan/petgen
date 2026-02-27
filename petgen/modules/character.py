"""Character creation module using Google Gemini / Nano Banana API."""

from __future__ import annotations

from pathlib import Path

from petgen.config import PetGenConfig
from petgen.models import BreedGroup, CharacterProfile, GeminiModel


class CharacterCreator:
    """Creates and manages persistent pet characters using Gemini image generation."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    def create_character(
        self,
        reference_images: list[Path],
        name: str,
        breed: str,
        breed_group: BreedGroup | None = None,
        model: GeminiModel = GeminiModel.NANO_BANANA_2,
    ) -> CharacterProfile:
        """Create a new pet character from reference photos."""
        raise NotImplementedError("TODO: implement in Task #2")

    def generate_canonical_poses(
        self,
        profile: CharacterProfile,
        model: GeminiModel = GeminiModel.NANO_BANANA_2,
    ) -> dict[str, Path]:
        """Generate front, side, and 3/4 canonical poses for animation source frames."""
        raise NotImplementedError("TODO: implement in Task #2")

    def generate_scene(
        self,
        profile: CharacterProfile,
        prompt: str,
        style: str = "photorealistic",
        model: GeminiModel = GeminiModel.NANO_BANANA_2,
    ) -> Path:
        """Generate a scene image with the character in a described setting."""
        raise NotImplementedError("TODO: implement in Task #2")

    def detect_breed(self, image_path: Path) -> tuple[str, BreedGroup]:
        """Auto-detect breed and breed group from a pet photo."""
        raise NotImplementedError("TODO: implement in Task #2")

    def list_characters(self) -> list[CharacterProfile]:
        """List all saved character profiles."""
        profiles = []
        for path in self.config.characters_dir.glob("*.json"):
            profiles.append(CharacterProfile.load(path))
        return profiles

    def get_character(self, character_id: str) -> CharacterProfile | None:
        """Load a character by ID."""
        path = self.config.characters_dir / f"{character_id}.json"
        if path.exists():
            return CharacterProfile.load(path)
        # Try matching by name
        for profile in self.list_characters():
            if profile.name.lower() == character_id.lower():
                return profile
        return None
