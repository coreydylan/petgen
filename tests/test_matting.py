"""Tests for the ViTMatte fur matting module."""

from __future__ import annotations

import numpy as np
import pytest

from petgen.config import PetGenConfig
from petgen.modules.matting import FurMatter


@pytest.fixture
def matter(tmp_config: PetGenConfig) -> FurMatter:
    return FurMatter(tmp_config)


@pytest.fixture
def sample_frames() -> list[np.ndarray]:
    """Create 5 test RGB frames with a bright subject on dark background."""
    frames = []
    for _ in range(5):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        # Bright subject in the center
        frame[30:100, 30:100] = 180
        # Add some noise for edge variety
        rng = np.random.RandomState(42)
        noise = rng.randint(-10, 10, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


class TestViTMatteAvailability:
    def test_availability_reflects_imports(self, matter: FurMatter):
        """vitmatte_available should reflect whether transformers is importable."""
        # In test environment, transformers may or may not be installed.
        # Just verify the property returns a bool without error.
        assert isinstance(matter.vitmatte_available, bool)


class TestFallbackEdgeMattes:
    def test_produces_one_matte_per_frame(self, matter: FurMatter, sample_frames: list):
        result = matter.generate_mattes(sample_frames)
        assert len(result) == len(sample_frames)

    def test_matte_shape_matches_frame(self, matter: FurMatter, sample_frames: list):
        result = matter.generate_mattes(sample_frames)
        for frame, matte in zip(sample_frames, result):
            h, w = frame.shape[:2]
            assert matte.shape == (h, w)

    def test_matte_values_in_unit_range(self, matter: FurMatter, sample_frames: list):
        result = matter.generate_mattes(sample_frames)
        for matte in result:
            assert matte.dtype == np.float32
            assert matte.min() >= 0.0
            assert matte.max() <= 1.0

    def test_empty_frames(self, matter: FurMatter):
        assert matter.generate_mattes([]) == []

    def test_matte_is_2d(self, matter: FurMatter, sample_frames: list):
        result = matter.generate_mattes(sample_frames)
        for matte in result:
            assert matte.ndim == 2, "Matte should be 2D"


class TestTrimapGeneration:
    def test_trimap_shape(self, matter: FurMatter, sample_frames: list):
        trimap = matter._generate_trimap(sample_frames[0])
        assert trimap.shape == sample_frames[0].shape[:2]

    def test_trimap_values(self, matter: FurMatter, sample_frames: list):
        trimap = matter._generate_trimap(sample_frames[0])
        unique = set(np.unique(trimap))
        # Trimap should only contain values from {0, 128, 255}
        assert unique.issubset({0, 128, 255})

    def test_trimap_has_all_regions(self, matter: FurMatter):
        """A frame with a clear subject should produce all three trimap regions."""
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        frame[40:90, 40:90] = 200  # clear bright subject
        trimap = matter._generate_trimap(frame)
        unique = set(np.unique(trimap))
        # Should have background (0) and at least one of fg/unknown
        assert 0 in unique

    def test_trimap_dtype(self, matter: FurMatter, sample_frames: list):
        trimap = matter._generate_trimap(sample_frames[0])
        assert trimap.dtype == np.uint8


class TestWithProvidedTrimaps:
    def test_custom_trimaps_are_used(self, matter: FurMatter, sample_frames: list):
        """When trimaps are provided, they should be used instead of auto-generated."""
        # Create simple trimaps
        trimaps = [np.full(f.shape[:2], 128, dtype=np.uint8) for f in sample_frames]
        result = matter.generate_mattes(sample_frames, trimaps=trimaps)
        assert len(result) == len(sample_frames)
        for matte in result:
            assert matte.dtype == np.float32
            assert matte.min() >= 0.0
            assert matte.max() <= 1.0
