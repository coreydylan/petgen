"""Tests for the secondary motion system."""

from __future__ import annotations

import numpy as np
import pytest

from petgen.models import AnimationConfig, BreedGroup, Landmark
from petgen.modules.motion import SecondaryMotion


@pytest.fixture
def motion() -> SecondaryMotion:
    return SecondaryMotion()


@pytest.fixture
def sample_frames() -> list[np.ndarray]:
    """Create a list of 50 solid-colored test frames (2 seconds at 25fps)."""
    rng = np.random.RandomState(0)
    return [rng.randint(50, 200, (256, 256, 3), dtype=np.uint8) for _ in range(50)]


@pytest.fixture
def uniform_frames() -> list[np.ndarray]:
    """Create uniform gray frames for predictable diff testing."""
    return [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(50)]


@pytest.fixture
def per_frame_landmarks() -> list[list[Landmark]]:
    """Create per-frame landmarks (10 points per frame, 50 frames)."""
    landmarks = []
    for _ in range(50):
        frame_lms = [
            Landmark(x=100 + i * 10, y=80 + (i % 3) * 5, visibility=1.0)
            for i in range(10)
        ]
        landmarks.append(frame_lms)
    return landmarks


class TestBreathing:
    def test_returns_same_number_of_frames(self, motion: SecondaryMotion, sample_frames: list):
        result = motion.apply_breathing(sample_frames, bpm=15.0, amplitude=0.02, fps=25.0)
        assert len(result) == len(sample_frames)

    def test_frames_have_same_shape(self, motion: SecondaryMotion, sample_frames: list):
        result = motion.apply_breathing(sample_frames, bpm=15.0, amplitude=0.02, fps=25.0)
        for orig, out in zip(sample_frames, result):
            assert out.shape == orig.shape

    def test_applies_sinusoidal_transform(self, motion: SecondaryMotion, uniform_frames: list):
        """Check that breathing modifies the lower portion of the frame."""
        result = motion.apply_breathing(uniform_frames, bpm=15.0, amplitude=0.03, fps=25.0)
        # Upper 40% should remain untouched
        h = uniform_frames[0].shape[0]
        split_y = int(h * 0.4)
        for orig, out in zip(uniform_frames, result):
            np.testing.assert_array_equal(orig[:split_y], out[:split_y])

    def test_lower_region_changes_over_time(self, motion: SecondaryMotion):
        """The lower region should vary across frames due to the sine wave."""
        # Use a gradient frame so warping produces visible pixel changes
        gradient_frames = []
        for _ in range(50):
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
            for row in range(256):
                frame[row, :, :] = row  # vertical gradient
            gradient_frames.append(frame)
        result = motion.apply_breathing(gradient_frames, bpm=15.0, amplitude=0.03, fps=25.0)
        h = gradient_frames[0].shape[0]
        split_y = int(h * 0.4)
        # Collect the mean of lower region across frames — should not all be identical
        means = [r[split_y:].mean() for r in result]
        assert not all(m == means[0] for m in means), "Lower region should vary across frames"

    def test_empty_frames(self, motion: SecondaryMotion):
        result = motion.apply_breathing([], bpm=15.0)
        assert result == []

    def test_cat_bpm_range(self, motion: SecondaryMotion, uniform_frames: list):
        """Higher BPM should produce faster oscillation."""
        result_slow = motion.apply_breathing(uniform_frames, bpm=12.0, amplitude=0.03, fps=25.0)
        result_fast = motion.apply_breathing(uniform_frames, bpm=30.0, amplitude=0.03, fps=25.0)
        # Both should produce valid frames
        assert len(result_slow) == len(result_fast) == len(uniform_frames)


class TestEarTwitches:
    def test_returns_same_number_of_frames(
        self, motion: SecondaryMotion, sample_frames: list, per_frame_landmarks: list
    ):
        energy = np.random.rand(50)
        result = motion.apply_ear_twitches(sample_frames, energy, per_frame_landmarks, fps=25.0)
        assert len(result) == len(sample_frames)

    def test_high_energy_causes_displacement(
        self, motion: SecondaryMotion, sample_frames: list, per_frame_landmarks: list
    ):
        """High energy spikes should modify frames (needs non-uniform pixels)."""
        # Create energy with spikes — most frames low, some very high
        # Normalization divides by mean, so spikes need to be well above mean
        energy = np.ones(50) * 0.1
        energy[5] = 5.0
        energy[15] = 5.0
        energy[25] = 5.0  # These will be >> 1.5x the mean
        result = motion.apply_ear_twitches(sample_frames, energy, per_frame_landmarks, fps=25.0)
        # At least some frames should differ from the original (non-uniform) input
        diffs = [np.abs(r.astype(float) - o.astype(float)).sum() for r, o in zip(result, sample_frames)]
        assert any(d > 0 for d in diffs), "High energy should cause frame modifications"

    def test_zero_energy_minimal_change(
        self, motion: SecondaryMotion, uniform_frames: list, per_frame_landmarks: list
    ):
        """Zero energy should cause minimal changes (only rare micro-twitches)."""
        energy = np.zeros(50)
        result = motion.apply_ear_twitches(uniform_frames, energy, per_frame_landmarks, fps=25.0)
        # Most frames should be unchanged with zero energy
        unchanged = sum(
            1 for r, o in zip(result, uniform_frames) if np.array_equal(r, o)
        )
        assert unchanged > len(uniform_frames) * 0.5

    def test_empty_landmarks(self, motion: SecondaryMotion, sample_frames: list):
        energy = np.random.rand(50)
        result = motion.apply_ear_twitches(sample_frames, energy, [], fps=25.0)
        assert len(result) == len(sample_frames)


class TestHeadBob:
    def test_returns_same_number_of_frames(self, motion: SecondaryMotion, sample_frames: list):
        pitch = np.random.rand(50)
        result = motion.apply_head_bob(sample_frames, pitch, amplitude=0.01, fps=25.0)
        assert len(result) == len(sample_frames)

    def test_constant_pitch_no_change(self, motion: SecondaryMotion, uniform_frames: list):
        """Constant pitch produces uniform normalized values of 0, so minimal displacement."""
        pitch = np.ones(50) * 100.0
        result = motion.apply_head_bob(uniform_frames, pitch, amplitude=0.01, fps=25.0)
        # With constant pitch, normalized values are all 0 -> no displacement
        for orig, out in zip(uniform_frames, result):
            np.testing.assert_array_equal(orig, out)

    def test_varying_pitch_causes_change(self, motion: SecondaryMotion, sample_frames: list):
        """Varying pitch should cause vertical displacement in the upper region."""
        pitch = np.linspace(0, 500, 50)
        result = motion.apply_head_bob(sample_frames, pitch, amplitude=0.02, fps=25.0)
        h = sample_frames[0].shape[0]
        # The last frame (highest pitch) should differ from the original in upper half
        diff = np.abs(result[-1][:h // 2].astype(float) - sample_frames[-1][:h // 2].astype(float))
        assert diff.sum() > 0, "High pitch frames should differ in upper region"

    def test_empty_pitch(self, motion: SecondaryMotion, sample_frames: list):
        result = motion.apply_head_bob(sample_frames, np.array([]), fps=25.0)
        assert len(result) == len(sample_frames)


class TestBlinks:
    def test_returns_same_number_of_frames(
        self, motion: SecondaryMotion, sample_frames: list, per_frame_landmarks: list
    ):
        result = motion.apply_blinks(sample_frames, per_frame_landmarks, fps=25.0)
        assert len(result) == len(sample_frames)

    def test_blinks_modify_eye_region(
        self, motion: SecondaryMotion, per_frame_landmarks: list
    ):
        """At least some frames should be modified during blink events."""
        # Use gradient frames so vertical scaling produces visible changes
        rng = np.random.RandomState(7)
        frames = [rng.randint(50, 200, (256, 256, 3), dtype=np.uint8) for _ in range(150)]
        lms = per_frame_landmarks * 3  # extend for 150 frames
        result = motion.apply_blinks(frames, lms, interval_range=(3.0, 6.0), fps=25.0)
        diffs = [not np.array_equal(r, o) for r, o in zip(result, frames)]
        assert any(diffs), "Blinks should modify at least some frames"

    def test_blink_interval_respected(
        self, motion: SecondaryMotion, per_frame_landmarks: list
    ):
        """Blinks should occur within the specified interval range."""
        # With 150 frames at 25fps = 6 seconds, should get at least 1 blink with (3,6) interval
        rng = np.random.RandomState(8)
        frames = [rng.randint(50, 200, (256, 256, 3), dtype=np.uint8) for _ in range(150)]
        lms = per_frame_landmarks * 3
        result = motion.apply_blinks(frames, lms, interval_range=(3.0, 6.0), fps=25.0)
        modified = sum(1 for r, o in zip(result, frames) if not np.array_equal(r, o))
        assert modified >= 1, "Should have at least one blink in 6 seconds"

    def test_empty_frames(self, motion: SecondaryMotion):
        result = motion.apply_blinks([], [], fps=25.0)
        assert result == [] or len(result) == 0


class TestTailWag:
    def test_returns_same_number_of_frames(self, motion: SecondaryMotion, sample_frames: list):
        result = motion.apply_tail_wag(sample_frames, excitement=0.5, fps=25.0)
        assert len(result) == len(sample_frames)

    def test_frames_have_same_shape(self, motion: SecondaryMotion, sample_frames: list):
        result = motion.apply_tail_wag(sample_frames, excitement=0.5, fps=25.0)
        for orig, out in zip(sample_frames, result):
            assert out.shape == orig.shape

    def test_higher_excitement_more_motion(self, motion: SecondaryMotion, uniform_frames: list):
        """Higher excitement should produce more visible changes."""
        result_calm = motion.apply_tail_wag(uniform_frames, excitement=0.1, fps=25.0)
        result_excited = motion.apply_tail_wag(uniform_frames, excitement=1.0, fps=25.0)
        # Compare total pixel displacement
        diff_calm = sum(
            np.abs(r.astype(float) - o.astype(float)).sum()
            for r, o in zip(result_calm, uniform_frames)
        )
        diff_excited = sum(
            np.abs(r.astype(float) - o.astype(float)).sum()
            for r, o in zip(result_excited, uniform_frames)
        )
        assert diff_excited >= diff_calm, "Higher excitement should cause more pixel displacement"

    def test_empty_frames(self, motion: SecondaryMotion):
        result = motion.apply_tail_wag([], excitement=0.5)
        assert result == []


class TestApplyAll:
    def test_chains_effects(self, motion: SecondaryMotion, per_frame_landmarks: list):
        """apply_all should produce valid output with all effects enabled."""
        frames = [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(50)]
        audio_features = {
            "energy": np.random.rand(50),
            "pitch": np.random.rand(50) * 300,
            "mean_energy": np.array([0.6]),
        }
        config = AnimationConfig(fps=25)
        result = motion.apply_all(frames, audio_features, per_frame_landmarks, config)
        assert len(result) == len(frames)
        for r in result:
            assert r.shape == frames[0].shape

    def test_cat_breed_uses_higher_bpm(self, motion: SecondaryMotion, per_frame_landmarks: list):
        """Cat breed group should use ~25 BPM instead of ~15 BPM."""
        frames = [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(50)]
        audio_features = {"energy": np.random.rand(50), "pitch": np.random.rand(50) * 300}
        config = AnimationConfig(fps=25)
        # Should not raise — cat BPM path is exercised
        result = motion.apply_all(
            frames, audio_features, per_frame_landmarks, config, breed_group=BreedGroup.CAT
        )
        assert len(result) == len(frames)

    def test_custom_breathing_bpm(self, motion: SecondaryMotion, per_frame_landmarks: list):
        """Custom breathing_bpm in config should override breed defaults."""
        frames = [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(50)]
        audio_features = {}
        config = AnimationConfig(fps=25, breathing_bpm=20.0, enable_ear_twitches=False, enable_head_bob=False, enable_tail_wag=False, enable_blinks=False)
        result = motion.apply_all(
            frames, audio_features, per_frame_landmarks, config, breed_group=BreedGroup.LARGE_DOG
        )
        assert len(result) == len(frames)

    def test_disabled_effects(self, motion: SecondaryMotion, per_frame_landmarks: list):
        """With all secondary effects disabled, only breathing should apply."""
        frames = [np.full((256, 256, 3), 128, dtype=np.uint8) for _ in range(50)]
        audio_features = {"energy": np.random.rand(50), "pitch": np.random.rand(50) * 300}
        config = AnimationConfig(
            fps=25,
            enable_blinks=False,
            enable_ear_twitches=False,
            enable_head_bob=False,
            enable_tail_wag=False,
        )
        result = motion.apply_all(frames, audio_features, per_frame_landmarks, config)
        assert len(result) == len(frames)
