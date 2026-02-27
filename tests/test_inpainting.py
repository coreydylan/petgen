"""Tests for the ProPainter inpainting module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from petgen.config import PetGenConfig
from petgen.modules.inpainting import ProPainterInpainter


@pytest.fixture
def inpainter(tmp_config: PetGenConfig) -> ProPainterInpainter:
    return ProPainterInpainter(tmp_config)


@pytest.fixture
def sample_frames() -> list[np.ndarray]:
    """Create 10 test frames with some variation."""
    rng = np.random.RandomState(42)
    return [rng.randint(50, 200, (64, 64, 3), dtype=np.uint8) for _ in range(10)]


@pytest.fixture
def sample_mouth_masks() -> list[np.ndarray]:
    """Create sample mouth masks -- white circle in center."""
    masks = []
    for _ in range(10):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, (32, 40), 10, 255, -1)
        masks.append(mask)
    return masks


class TestProPainterAvailability:
    def test_not_available_without_weights(self, inpainter: ProPainterInpainter):
        """ProPainter should not be available when weights dir is empty."""
        assert inpainter.propainter_available is False

    def test_available_with_weights(self, inpainter: ProPainterInpainter):
        """ProPainter should be available when weights directory has files."""
        weights_dir = inpainter.config.weights_dir / "propainter"
        weights_dir.mkdir(parents=True, exist_ok=True)
        (weights_dir / "ProPainter.pth").touch()
        assert inpainter.propainter_available is True


class TestFallbackGaussianRepair:
    def test_fallback_returns_same_count(
        self,
        inpainter: ProPainterInpainter,
        sample_frames: list,
        sample_mouth_masks: list,
    ):
        """Fallback should return the same number of frames."""
        result = inpainter.repair_mouth_artifacts(sample_frames, sample_mouth_masks)
        assert len(result) == len(sample_frames)

    def test_fallback_preserves_shape(
        self,
        inpainter: ProPainterInpainter,
        sample_frames: list,
        sample_mouth_masks: list,
    ):
        """Output frames should have the same shape as input."""
        result = inpainter.repair_mouth_artifacts(sample_frames, sample_mouth_masks)
        for orig, out in zip(sample_frames, result):
            assert out.shape == orig.shape

    def test_fallback_empty_masks_passthrough(
        self,
        inpainter: ProPainterInpainter,
        sample_frames: list,
    ):
        """Empty masks should pass through frames unchanged."""
        empty_masks = [np.zeros((64, 64), dtype=np.uint8) for _ in range(10)]
        result = inpainter.repair_mouth_artifacts(sample_frames, empty_masks)
        for orig, out in zip(sample_frames, result):
            np.testing.assert_array_equal(orig, out)

    def test_fallback_empty_frames(self, inpainter: ProPainterInpainter):
        result = inpainter.repair_mouth_artifacts([], [])
        assert result == []

    def test_fallback_mismatched_mask_count(
        self,
        inpainter: ProPainterInpainter,
        sample_frames: list,
        sample_mouth_masks: list,
    ):
        """Fewer masks than frames -- extra frames copied as-is."""
        short_masks = sample_mouth_masks[:3]
        result = inpainter.repair_mouth_artifacts(sample_frames, short_masks)
        assert len(result) == len(sample_frames)
        for i in range(3, len(sample_frames)):
            np.testing.assert_array_equal(result[i], sample_frames[i])

    def test_fallback_modifies_boundary(
        self,
        inpainter: ProPainterInpainter,
        sample_mouth_masks: list,
    ):
        """Gaussian fallback should modify pixels in the mouth boundary region."""
        frames = [np.full((64, 64, 3), 100, dtype=np.uint8) for _ in range(10)]
        # Add a bright patch inside the mask area to create contrast
        for f in frames:
            f[30:50, 22:42] = 200
        result = inpainter.repair_mouth_artifacts(frames, sample_mouth_masks)
        # At least some frames should differ from the original
        any_diff = any(not np.array_equal(r, o) for r, o in zip(result, frames))
        assert any_diff


class TestOpticalFlow:
    def test_compute_optical_flows_two_frames(self, inpainter: ProPainterInpainter):
        rng = np.random.RandomState(0)
        f1 = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        f2 = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        fwd, bwd = inpainter._compute_optical_flows([f1, f2])
        assert len(fwd) == 1
        assert len(bwd) == 1
        assert fwd[0].shape == (64, 64, 2)
        assert bwd[0].shape == (64, 64, 2)

    def test_compute_optical_flows_single_frame(self, inpainter: ProPainterInpainter):
        f1 = np.zeros((64, 64, 3), dtype=np.uint8)
        fwd, bwd = inpainter._compute_optical_flows([f1])
        assert fwd == []
        assert bwd == []

    def test_compute_optical_flows_empty(self, inpainter: ProPainterInpainter):
        fwd, bwd = inpainter._compute_optical_flows([])
        assert fwd == []
        assert bwd == []


class TestNormalizeMask:
    def test_uint8_to_float(self):
        mask = np.full((64, 64), 255, dtype=np.uint8)
        result = ProPainterInpainter._normalize_mask(mask, 64, 64)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 1.0)

    def test_resize_mask(self):
        mask = np.full((32, 32), 128, dtype=np.uint8)
        result = ProPainterInpainter._normalize_mask(mask, 64, 64)
        assert result.shape == (64, 64)

    def test_3d_mask_squeezed(self):
        mask = np.full((64, 64, 1), 255, dtype=np.uint8)
        result = ProPainterInpainter._normalize_mask(mask, 64, 64)
        assert result.ndim == 2
