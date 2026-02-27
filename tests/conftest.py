"""Shared test fixtures for PetGen."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from petgen.config import PetGenConfig
from petgen.models import (
    BoundingBox,
    BreedGroup,
    CharacterProfile,
    Landmark,
    VoiceSettings,
)


@pytest.fixture
def tmp_config(tmp_path: Path) -> PetGenConfig:
    """Create a test config with temp directories."""
    config = PetGenConfig(
        base_dir=tmp_path / "petgen",
        characters_dir=tmp_path / "petgen" / "characters",
        output_dir=tmp_path / "petgen" / "output",
        temp_dir=tmp_path / "petgen" / "tmp",
        weights_dir=tmp_path / "petgen" / "weights",
        voice_presets_dir=tmp_path / "petgen" / "voice_presets",
        gemini_api_key="test-key",
        device="cpu",
        tts_device="cpu",
    )
    config.ensure_dirs()
    return config


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample RGB image for testing."""
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def sample_character(tmp_config: PetGenConfig) -> CharacterProfile:
    """Create a sample character profile."""
    return CharacterProfile(
        id="test-char-001",
        name="Max",
        breed="Golden Retriever",
        breed_group=BreedGroup.LARGE_DOG,
        reference_images=[],
        voice=VoiceSettings(
            preset_name="large_dog",
            exaggeration=0.5,
            speed=0.9,
        ),
        personality_tags=["friendly", "goofy"],
    )


@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Create a sample bounding box."""
    return BoundingBox(x1=100, y1=80, x2=400, y2=400, confidence=0.95)


@pytest.fixture
def sample_landmarks() -> list[Landmark]:
    """Create sample facial landmarks (simplified 10-point set)."""
    return [
        Landmark(x=200 + i * 10, y=250 + (i % 3) * 5, visibility=1.0)
        for i in range(10)
    ]
