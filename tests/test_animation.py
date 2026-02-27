"""Tests for the lip sync animation module (JoyVASA + LivePortrait)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from petgen.models import AnimationConfig, MotionSequence
from petgen.modules.animation import (
    LipSyncAnimator,
    _EXPRESSION_DIM,
    _MOUTH_EXPRESSION_INDICES,
    _NUM_KEYPOINTS,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def animator(tmp_config):
    return LipSyncAnimator(tmp_config)


@pytest.fixture
def sample_motion() -> MotionSequence:
    """Create a sample motion sequence with realistic shapes."""
    T = 50  # 2 seconds at 25fps
    rng = np.random.default_rng(42)
    return MotionSequence(
        keypoints=rng.standard_normal((T, _NUM_KEYPOINTS, 2)).astype(np.float32),
        expressions=rng.standard_normal((T, _EXPRESSION_DIM)).astype(np.float32),
        fps=25.0,
    )


@pytest.fixture
def sample_source_image(tmp_path) -> Path:
    """Create a temporary source image file."""
    import cv2

    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    path = tmp_path / "source.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def sample_audio(tmp_path) -> Path:
    """Create a temporary WAV file with silence."""
    import struct
    import wave

    path = tmp_path / "audio.wav"
    sr = 16000
    duration = 2.0
    num_samples = int(sr * duration)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{num_samples}h", *([0] * num_samples)))
    return path


# ------------------------------------------------------------------
# _limit_mouth_amplitude
# ------------------------------------------------------------------


class TestLimitMouthAmplitude:
    def test_limit_scales_expression_coefficients(self, animator, sample_motion):
        limit = 0.5
        result = animator._limit_mouth_amplitude(sample_motion, limit)

        # Expression coefficients for mouth indices should be scaled toward mean
        for idx in _MOUTH_EXPRESSION_INDICES:
            if idx < _EXPRESSION_DIM:
                orig = sample_motion.expressions[:, idx]
                new = result.expressions[:, idx]
                mean_val = orig.mean()
                expected = mean_val + (orig - mean_val) * limit
                np.testing.assert_allclose(new, expected, rtol=1e-5)

    def test_limit_scales_mouth_keypoints(self, animator, sample_motion):
        limit = 0.5
        mouth_kp_start = _NUM_KEYPOINTS - 5
        result = animator._limit_mouth_amplitude(sample_motion, limit)

        orig_kp = sample_motion.keypoints[:, mouth_kp_start:, :]
        new_kp = result.keypoints[:, mouth_kp_start:, :]
        mean_kp = orig_kp.mean(axis=0, keepdims=True)
        expected_kp = mean_kp + (orig_kp - mean_kp) * limit
        np.testing.assert_allclose(new_kp, expected_kp, rtol=1e-5)

    def test_limit_does_not_mutate_original(self, animator, sample_motion):
        orig_kp = sample_motion.keypoints.copy()
        orig_expr = sample_motion.expressions.copy()

        animator._limit_mouth_amplitude(sample_motion, 0.5)

        np.testing.assert_array_equal(sample_motion.keypoints, orig_kp)
        np.testing.assert_array_equal(sample_motion.expressions, orig_expr)

    def test_limit_preserves_non_mouth_keypoints(self, animator, sample_motion):
        mouth_kp_start = _NUM_KEYPOINTS - 5
        result = animator._limit_mouth_amplitude(sample_motion, 0.3)

        # Non-mouth keypoints should be unchanged
        np.testing.assert_array_equal(
            result.keypoints[:, :mouth_kp_start, :],
            sample_motion.keypoints[:, :mouth_kp_start, :],
        )

    def test_limit_preserves_non_mouth_expressions(self, animator, sample_motion):
        result = animator._limit_mouth_amplitude(sample_motion, 0.3)

        non_mouth = [i for i in range(_EXPRESSION_DIM) if i not in _MOUTH_EXPRESSION_INDICES]
        for idx in non_mouth:
            np.testing.assert_array_equal(
                result.expressions[:, idx],
                sample_motion.expressions[:, idx],
            )

    def test_limit_of_one_is_identity(self, animator, sample_motion):
        result = animator._limit_mouth_amplitude(sample_motion, 1.0)

        np.testing.assert_allclose(result.keypoints, sample_motion.keypoints, rtol=1e-5)
        np.testing.assert_allclose(result.expressions, sample_motion.expressions, rtol=1e-5)

    def test_limit_of_zero_collapses_to_mean(self, animator, sample_motion):
        result = animator._limit_mouth_amplitude(sample_motion, 0.0)

        # Mouth expressions should all equal their temporal mean
        for idx in _MOUTH_EXPRESSION_INDICES:
            if idx < _EXPRESSION_DIM:
                vals = result.expressions[:, idx]
                assert np.allclose(vals, vals[0]), f"Expression idx {idx} not collapsed to mean"

    def test_preserves_fps(self, animator, sample_motion):
        result = animator._limit_mouth_amplitude(sample_motion, 0.5)
        assert result.fps == sample_motion.fps

    def test_invalid_limit_raises(self, animator, sample_motion):
        with pytest.raises(ValueError, match="mouth amplitude limit"):
            animator._limit_mouth_amplitude(sample_motion, -0.1)
        with pytest.raises(ValueError, match="mouth amplitude limit"):
            animator._limit_mouth_amplitude(sample_motion, 1.5)


# ------------------------------------------------------------------
# generate_motion (mocked)
# ------------------------------------------------------------------


class TestGenerateMotion:
    def test_raises_without_weights(self, animator, sample_audio, sample_source_image):
        """JoyVASA weights directory must exist."""
        with pytest.raises(FileNotFoundError, match="JoyVASA weights"):
            animator.generate_motion(sample_audio, sample_source_image)

    def test_generate_motion_output_shape(self, animator, tmp_path, sample_audio):
        """With mocked JoyVASA internals, output has correct structure."""
        import torch

        T = 50
        output_dim = _NUM_KEYPOINTS * 2 + _EXPRESSION_DIM
        fake_output = torch.randn(1, T, output_dim)

        fake_wav2vec = MagicMock()
        fake_wav2vec.return_value = MagicMock(last_hidden_state=torch.randn(1, 100, 768))

        fake_processor = MagicMock()
        fake_processor.return_value = MagicMock(
            input_values=torch.randn(1, 32000)
        )

        fake_diffusion = MagicMock()
        fake_diffusion.return_value = fake_output

        animator._joyvasa = {
            "processor": fake_processor,
            "wav2vec": fake_wav2vec,
            "diffusion": fake_diffusion,
            "device": "cpu",
            "dtype": torch.float32,
        }

        source_img = tmp_path / "source.png"
        source_img.touch()

        motion = animator.generate_motion(sample_audio, source_img)

        assert isinstance(motion, MotionSequence)
        assert motion.keypoints.shape == (T, _NUM_KEYPOINTS, 2)
        assert motion.expressions.shape == (T, _EXPRESSION_DIM)
        assert motion.fps == 25.0


# ------------------------------------------------------------------
# animate (mocked)
# ------------------------------------------------------------------


class TestAnimate:
    def test_raises_without_weights(self, animator, sample_source_image, sample_motion):
        with pytest.raises(FileNotFoundError, match="LivePortrait"):
            animator.animate(sample_source_image, sample_motion)

    def test_animate_produces_frames(self, animator, sample_source_image, sample_motion):
        """With mocked LivePortrait, animate returns correct number of frames."""
        import torch

        fake_frame = torch.rand(1, 3, 256, 256)

        animator._liveportrait = {
            "appearance_encoder": MagicMock(return_value=torch.randn(1, 256, 64, 64)),
            "motion_encoder": MagicMock(return_value=torch.randn(1, _NUM_KEYPOINTS, 2)),
            "warping_module": MagicMock(return_value=torch.randn(1, 256, 64, 64)),
            "spade_generator": MagicMock(return_value=fake_frame),
            "device": "cpu",
            "dtype": torch.float32,
        }

        frames = animator.animate(sample_source_image, sample_motion)

        assert len(frames) == sample_motion.num_frames
        for f in frames:
            assert isinstance(f, np.ndarray)
            assert f.dtype == np.uint8
            assert f.ndim == 3
            assert f.shape[2] == 3  # RGB

    def test_animate_raises_on_missing_image(self, animator, tmp_path, sample_motion):
        import torch

        animator._liveportrait = {
            "appearance_encoder": MagicMock(),
            "motion_encoder": MagicMock(),
            "warping_module": MagicMock(),
            "spade_generator": MagicMock(),
            "device": "cpu",
            "dtype": torch.float32,
        }
        missing = tmp_path / "nonexistent.png"

        with pytest.raises(FileNotFoundError, match="Cannot read source image"):
            animator.animate(missing, sample_motion)


# ------------------------------------------------------------------
# animate_from_audio (integration, mocked)
# ------------------------------------------------------------------


class TestAnimateFromAudio:
    def test_end_to_end_with_amplitude_limiting(self, animator, sample_source_image, sample_audio):
        """Full pipeline with mouth amplitude limiting applied."""
        import torch

        T = 25
        output_dim = _NUM_KEYPOINTS * 2 + _EXPRESSION_DIM

        # Mock JoyVASA
        animator._joyvasa = {
            "processor": MagicMock(return_value=MagicMock(input_values=torch.randn(1, 32000))),
            "wav2vec": MagicMock(
                return_value=MagicMock(last_hidden_state=torch.randn(1, 50, 768))
            ),
            "diffusion": MagicMock(return_value=torch.randn(1, T, output_dim)),
            "device": "cpu",
            "dtype": torch.float32,
        }

        # Mock LivePortrait
        fake_frame = torch.rand(1, 3, 256, 256)
        animator._liveportrait = {
            "appearance_encoder": MagicMock(return_value=torch.randn(1, 256, 64, 64)),
            "motion_encoder": MagicMock(return_value=torch.randn(1, _NUM_KEYPOINTS, 2)),
            "warping_module": MagicMock(return_value=torch.randn(1, 256, 64, 64)),
            "spade_generator": MagicMock(return_value=fake_frame),
            "device": "cpu",
            "dtype": torch.float32,
        }

        config = AnimationConfig(mouth_amplitude_limit=0.6)
        frames = animator.animate_from_audio(sample_source_image, sample_audio, config)

        assert len(frames) == T
        assert all(f.dtype == np.uint8 for f in frames)

    def test_no_limiting_when_limit_is_one(self, animator):
        """When mouth_amplitude_limit=1.0, _limit_mouth_amplitude should not be called."""
        motion = MotionSequence(
            keypoints=np.zeros((10, _NUM_KEYPOINTS, 2)),
            expressions=np.zeros((10, _EXPRESSION_DIM)),
        )
        config = AnimationConfig(mouth_amplitude_limit=1.0)

        animator.generate_motion = MagicMock(return_value=motion)
        animator.animate = MagicMock(return_value=[np.zeros((256, 256, 3), dtype=np.uint8)])
        animator._limit_mouth_amplitude = MagicMock()

        animator.animate_from_audio(Path("/fake/img.png"), Path("/fake/audio.wav"), config)

        animator._limit_mouth_amplitude.assert_not_called()

    def test_limiting_applied_when_below_one(self, animator):
        """When mouth_amplitude_limit < 1.0, _limit_mouth_amplitude is called."""
        motion = MotionSequence(
            keypoints=np.zeros((10, _NUM_KEYPOINTS, 2)),
            expressions=np.zeros((10, _EXPRESSION_DIM)),
        )
        config = AnimationConfig(mouth_amplitude_limit=0.7)

        animator.generate_motion = MagicMock(return_value=motion)
        animator.animate = MagicMock(return_value=[np.zeros((256, 256, 3), dtype=np.uint8)])
        original_limit = animator._limit_mouth_amplitude
        animator._limit_mouth_amplitude = MagicMock(side_effect=original_limit)

        animator.animate_from_audio(Path("/fake/img.png"), Path("/fake/audio.wav"), config)

        animator._limit_mouth_amplitude.assert_called_once()
        call_args = animator._limit_mouth_amplitude.call_args
        assert call_args[0][1] == pytest.approx(0.7)


# ------------------------------------------------------------------
# MotionSequence properties
# ------------------------------------------------------------------


class TestMotionSequence:
    def test_num_frames(self):
        ms = MotionSequence(
            keypoints=np.zeros((100, 21, 2)),
            expressions=np.zeros((100, 63)),
            fps=25.0,
        )
        assert ms.num_frames == 100

    def test_duration(self):
        ms = MotionSequence(
            keypoints=np.zeros((50, 21, 2)),
            expressions=np.zeros((50, 63)),
            fps=25.0,
        )
        assert ms.duration == pytest.approx(2.0)
