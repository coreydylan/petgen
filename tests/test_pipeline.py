"""Tests for the PetGen pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from petgen.config import PetGenConfig
from petgen.models import (
    AnimationConfig,
    BreedGroup,
    CharacterProfile,
    MotionSequence,
    PipelineResult,
    VoiceSettings,
)
from petgen.pipeline import PetGenPipeline


@pytest.fixture
def pipeline(tmp_config: PetGenConfig) -> PetGenPipeline:
    return PetGenPipeline(tmp_config)


@pytest.fixture
def character_with_files(tmp_config: PetGenConfig) -> CharacterProfile:
    """Create a character with actual image files on disk."""
    char_dir = tmp_config.characters_dir / "test-char"
    images_dir = char_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image file
    from PIL import Image

    dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    front_path = images_dir / "pose_front.png"
    ref_path = images_dir / "ref_0.png"
    dummy_img.save(front_path)
    dummy_img.save(ref_path)

    return CharacterProfile(
        id="test-char",
        name="Buddy",
        breed="Labrador",
        breed_group=BreedGroup.LARGE_DOG,
        reference_images=[ref_path],
        canonical_poses={"front": front_path},
        voice=VoiceSettings(exaggeration=0.6, speed=1.0),
    )


class TestPetGenPipeline:
    def test_lazy_module_loading(self, pipeline: PetGenPipeline) -> None:
        """Properties should lazily instantiate modules."""
        assert pipeline._character_creator is None
        _ = pipeline.character_creator
        assert pipeline._character_creator is not None

    def test_create_character_delegates(self, pipeline: PetGenPipeline) -> None:
        """create_character should delegate to CharacterCreator."""
        mock_creator = MagicMock()
        mock_creator.create_character.return_value = MagicMock(spec=CharacterProfile)
        pipeline._character_creator = mock_creator

        photos = [Path("/fake/photo.jpg")]
        pipeline.create_character(name="Rex", photos=photos, breed="Poodle")

        mock_creator.create_character.assert_called_once_with(
            reference_images=photos, name="Rex", breed="Poodle"
        )

    def test_generate_uses_canonical_pose_when_no_scene_prompt(
        self, pipeline: PetGenPipeline, character_with_files: CharacterProfile
    ) -> None:
        """When no scene_prompt given, use canonical front pose as source image."""
        # Mock all the modules
        mock_voice = MagicMock()
        mock_voice.generate.return_value = Path("/tmp/audio.wav")
        pipeline._voice_generator = mock_voice

        mock_detector = MagicMock()
        mock_detector.detect_face.return_value = MagicMock(
            x1=10, y1=10, x2=200, y2=200, width=190, height=190, center=(105, 105)
        )
        mock_detector.detect_landmarks.return_value = []
        mock_detector.track_mouth_video.return_value = []
        pipeline._face_detector = mock_detector

        mock_animator = MagicMock()
        dummy_frames = [np.zeros((256, 256, 3), dtype=np.uint8)] * 5
        mock_animator.animate_from_audio.return_value = dummy_frames
        pipeline._animator = mock_animator

        mock_motion = MagicMock()
        mock_motion.apply_all.return_value = dummy_frames
        pipeline._motion = mock_motion

        mock_comp = MagicMock()
        mock_comp.composite_full.return_value = Path("/tmp/output.mp4")
        pipeline._compositor = mock_comp

        result = pipeline.generate(
            character=character_with_files,
            script="Hello!",
        )

        assert isinstance(result, PipelineResult)
        assert result.video_path == Path("/tmp/output.mp4")

    def test_extract_audio_features_fallback(self, pipeline: PetGenPipeline) -> None:
        """Audio feature extraction should return defaults for missing audio."""
        features = pipeline._extract_audio_features(
            Path("/nonexistent/audio.wav"), num_frames=10, fps=25
        )
        assert "energy" in features
        assert "pitch" in features
        assert "mean_energy" in features
        assert len(features["energy"]) == 10
        assert len(features["pitch"]) == 10


class TestPipelineConfig:
    def test_aspect_ratio_resolution_mapping(self) -> None:
        from petgen.pipeline import _ASPECT_RESOLUTIONS
        from petgen.models import AspectRatio

        assert _ASPECT_RESOLUTIONS[AspectRatio.VERTICAL] == (1080, 1920)
        assert _ASPECT_RESOLUTIONS[AspectRatio.SQUARE] == (1080, 1080)
        assert _ASPECT_RESOLUTIONS[AspectRatio.LANDSCAPE] == (1920, 1080)
