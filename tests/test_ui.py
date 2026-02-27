"""Tests for the Gradio web interface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from petgen.models import BreedGroup, CharacterProfile, PipelineResult, VoiceSettings


@pytest.fixture
def mock_pipeline():
    """Create a mock PetGenPipeline with common behaviors."""
    pipeline = MagicMock()

    sample_char = CharacterProfile(
        id="test-001",
        name="Buddy",
        breed="Golden Retriever",
        breed_group=BreedGroup.LARGE_DOG,
        reference_images=[Path("/tmp/ref1.jpg")],
        canonical_poses={"front": Path("/tmp/front.png")},
        voice=VoiceSettings(exaggeration=0.7, speed=1.0),
    )

    pipeline.character_creator.list_characters.return_value = [sample_char]
    pipeline.character_creator.get_character.return_value = sample_char
    pipeline.create_character.return_value = sample_char

    return pipeline


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config."""
    config = MagicMock()
    config.output_dir = tmp_path / "output"
    config.output_dir.mkdir()
    config.default_fps = 25
    return config


@pytest.fixture
def _patch_pipeline(mock_config, mock_pipeline):
    """Patch _get_pipeline to return mocked objects."""
    with patch("petgen.ui.app._get_pipeline", return_value=(mock_config, mock_pipeline)):
        yield mock_config, mock_pipeline


class TestCreateUI:
    """Test that create_ui() returns a proper Gradio Blocks app."""

    def test_returns_blocks_instance(self):
        import gradio as gr

        with patch("petgen.ui.app.get_character_choices", return_value=[]):
            from petgen.ui.app import create_ui

            app = create_ui()

        assert isinstance(app, gr.Blocks)

    def test_has_five_tabs(self):
        import gradio as gr

        with patch("petgen.ui.app.get_character_choices", return_value=[]):
            from petgen.ui.app import create_ui

            app = create_ui()

        # Blocks stores children; count Tab instances
        tabs = [
            child
            for child in _walk_blocks(app)
            if isinstance(child, gr.Tab)
        ]
        assert len(tabs) == 5


class TestGetCharacterChoices:
    """Test the character dropdown helper."""

    def test_returns_tuples(self, _patch_pipeline):
        from petgen.ui.app import get_character_choices

        choices = get_character_choices()
        assert len(choices) == 1
        label, cid = choices[0]
        assert "Buddy" in label
        assert "Golden Retriever" in label
        assert cid == "test-001"

    def test_returns_empty_when_no_characters(self, _patch_pipeline):
        _, pipeline = _patch_pipeline
        pipeline.character_creator.list_characters.return_value = []

        from petgen.ui.app import get_character_choices

        choices = get_character_choices()
        assert choices == []

    def test_handles_exception(self, _patch_pipeline):
        _, pipeline = _patch_pipeline
        pipeline.character_creator.list_characters.side_effect = RuntimeError("fail")

        from petgen.ui.app import get_character_choices

        choices = get_character_choices()
        assert choices == []


class TestHandleCreateCharacter:
    """Test the character creation handler."""

    def test_no_files_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_create_character

        gallery, info, dropdown_update = handle_create_character(None, "Buddy", "")
        assert gallery is None
        assert "No photos" in str(info)

    def test_no_name_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_create_character

        gallery, info, dropdown_update = handle_create_character(
            ["/tmp/photo.jpg"], "", ""
        )
        assert gallery is None
        assert "No name" in str(info)

    def test_success(self, _patch_pipeline):
        from petgen.ui.app import handle_create_character

        _, pipeline = _patch_pipeline
        gallery, info, dropdown_update = handle_create_character(
            ["/tmp/photo.jpg"], "Buddy", "Golden Retriever"
        )
        pipeline.create_character.assert_called_once()
        assert isinstance(info, dict)
        assert info["name"] == "Buddy"


class TestHandleGenerateVideo:
    """Test the video generation handler."""

    def test_no_character_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_generate_video

        video, stats = handle_generate_video("", "Hello!", "", 0.7, 1.0, "9:16 (Vertical)")
        assert video is None

    def test_no_script_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_generate_video

        video, stats = handle_generate_video("test-001", "", "", 0.7, 1.0, "9:16 (Vertical)")
        assert video is None

    def test_success(self, _patch_pipeline):
        from petgen.ui.app import handle_generate_video

        mock_config, pipeline = _patch_pipeline
        pipeline.generate.return_value = PipelineResult(
            video_path=Path("/tmp/output.mp4"),
            audio_path=Path("/tmp/audio.wav"),
            character=pipeline.character_creator.get_character.return_value,
            duration=3.5,
            num_frames=87,
            metadata={"elapsed_seconds": 12.3},
        )

        video, stats = handle_generate_video(
            "test-001", "Hello world!", "", 0.7, 1.0, "9:16 (Vertical)"
        )
        assert video == "/tmp/output.mp4"
        assert "3.5" in stats
        assert "87" in stats
        assert "12.3" in stats


class TestHandleVoicePreview:
    """Test the voice preview handler."""

    def test_no_character_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_voice_preview

        result = handle_voice_preview("", "Hello", 0.7)
        assert result is None

    def test_no_text_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_voice_preview

        result = handle_voice_preview("test-001", "", 0.7)
        assert result is None

    def test_success(self, _patch_pipeline):
        from petgen.ui.app import handle_voice_preview

        _, pipeline = _patch_pipeline
        pipeline.voice_generator.generate.return_value = Path("/tmp/preview.wav")

        result = handle_voice_preview("test-001", "Hello!", 0.5)
        assert result == "/tmp/preview.wav"


class TestHandleSceneGenerate:
    """Test the scene generation handler."""

    def test_no_character_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_scene_generate

        result = handle_scene_generate("", "a park", "photorealistic")
        assert result is None

    def test_no_description_warns(self, _patch_pipeline):
        from petgen.ui.app import handle_scene_generate

        result = handle_scene_generate("test-001", "", "photorealistic")
        assert result is None

    def test_success(self, _patch_pipeline):
        from petgen.ui.app import handle_scene_generate

        _, pipeline = _patch_pipeline
        pipeline.character_creator.generate_scene.return_value = Path("/tmp/scene.png")

        result = handle_scene_generate("test-001", "a sunny park", "watercolor")
        assert result == "/tmp/scene.png"
        pipeline.character_creator.generate_scene.assert_called_once()


class TestHandleGalleryRefresh:
    """Test the gallery refresh handler."""

    def test_empty_output_dir(self, _patch_pipeline):
        from petgen.ui.app import handle_gallery_refresh

        result = handle_gallery_refresh()
        assert result is None  # empty dir, no videos

    def test_finds_videos(self, _patch_pipeline):
        from petgen.ui.app import handle_gallery_refresh

        mock_config, _ = _patch_pipeline
        (mock_config.output_dir / "test.mp4").touch()
        (mock_config.output_dir / "test2.webm").touch()
        (mock_config.output_dir / "not_a_video.txt").touch()

        result = handle_gallery_refresh()
        assert result is not None
        assert len(result) == 2
        extensions = {Path(p).suffix for p in result}
        assert extensions == {".mp4", ".webm"}


# ---------------------------------------------------------------------------
# Helper to walk Gradio component tree
# ---------------------------------------------------------------------------


def _walk_blocks(block):
    """Recursively yield all children of a Gradio Blocks/layout."""
    if hasattr(block, "children"):
        for child in block.children:
            yield child
            yield from _walk_blocks(child)
