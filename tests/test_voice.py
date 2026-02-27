"""Tests for the voice generation module."""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from petgen.config import PetGenConfig
from petgen.models import BreedGroup, VoiceSettings
from petgen.modules.voice import DEFAULT_PRESETS, VoiceGenerator, VoicePreset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: Path, duration_s: float = 0.5, sample_rate: int = 24000) -> Path:
    """Write a minimal valid WAV file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return path


def _mock_torch_tensor(n_samples: int = 24000):
    """Create a mock tensor mimicking ChatterboxTTS output."""
    mock = MagicMock()
    mock.dim.return_value = 1
    mock.unsqueeze.return_value = mock
    mock.abs.return_value = mock
    mock.max.return_value = 0.8
    mock.__truediv__ = lambda self, other: self
    mock.__mul__ = lambda self, other: self
    return mock


# ---------------------------------------------------------------------------
# ChatterboxTTS generate() flow
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_audio_libs():
    """Inject mock torch and torchaudio into sys.modules for local imports."""
    mock_torch = MagicMock()
    mock_torchaudio = MagicMock()
    with patch.dict(
        "sys.modules",
        {"torch": mock_torch, "torchaudio": mock_torchaudio, "torchaudio.functional": mock_torchaudio.functional},
    ):
        yield mock_torch, mock_torchaudio


class TestGenerateChatterbox:
    """Test generate() using the Chatterbox backend."""

    def test_generate_basic(self, _mock_audio_libs, tmp_config: PetGenConfig):
        """Generate audio from text with default settings."""
        _, mock_torchaudio = _mock_audio_libs
        gen = VoiceGenerator(tmp_config)
        mock_model = MagicMock()
        mock_model.sr = 24000
        wav_tensor = _mock_torch_tensor()
        mock_model.generate.return_value = wav_tensor
        gen._model = mock_model

        output = tmp_config.temp_dir / "test_basic.wav"
        result = gen.generate("Hello world!", output_path=output)

        assert result == output
        mock_model.generate.assert_called_once_with(text="Hello world!", exaggeration=0.7)
        mock_torchaudio.save.assert_called_once()

    def test_generate_with_voice_ref(self, _mock_audio_libs, tmp_config: PetGenConfig):
        """Generate audio with a voice reference clip for cloning."""
        gen = VoiceGenerator(tmp_config)
        mock_model = MagicMock()
        mock_model.sr = 24000
        mock_model.generate.return_value = _mock_torch_tensor()
        gen._model = mock_model

        ref_clip = _make_wav(tmp_config.voice_presets_dir / "ref.wav")
        output = tmp_config.temp_dir / "test_ref.wav"
        result = gen.generate("Woof!", voice_ref=ref_clip, output_path=output)

        assert result == output
        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["audio_prompt"] == str(ref_clip)

    def test_generate_custom_exaggeration(self, _mock_audio_libs, tmp_config: PetGenConfig):
        """Generate audio with custom exaggeration."""
        gen = VoiceGenerator(tmp_config)
        mock_model = MagicMock()
        mock_model.sr = 24000
        mock_model.generate.return_value = _mock_torch_tensor()
        gen._model = mock_model

        output = tmp_config.temp_dir / "test_exag.wav"
        gen.generate("Excited bark!", exaggeration=0.9, output_path=output)

        call_kwargs = mock_model.generate.call_args[1]
        assert call_kwargs["exaggeration"] == 0.9

    def test_generate_speed_adjustment(self, _mock_audio_libs, tmp_config: PetGenConfig):
        """Generate audio with speed adjustment triggers resampling."""
        _, mock_torchaudio = _mock_audio_libs
        gen = VoiceGenerator(tmp_config)
        mock_model = MagicMock()
        mock_model.sr = 24000
        mock_model.generate.return_value = _mock_torch_tensor()
        gen._model = mock_model

        output = tmp_config.temp_dir / "test_speed.wav"
        gen.generate("Fast talk!", speed=1.2, output_path=output)

        mock_torchaudio.functional.resample.assert_called_once()

    def test_generate_auto_output_path(self, _mock_audio_libs, tmp_config: PetGenConfig):
        """Generate audio without specifying output_path gets auto-generated path."""
        gen = VoiceGenerator(tmp_config)
        mock_model = MagicMock()
        mock_model.sr = 24000
        mock_model.generate.return_value = _mock_torch_tensor()
        gen._model = mock_model

        result = gen.generate("Auto path test")

        assert result.parent == tmp_config.temp_dir
        assert result.suffix == ".wav"
        assert result.stem.startswith("voice_")


# ---------------------------------------------------------------------------
# Fish Audio S1 fallback
# ---------------------------------------------------------------------------


class TestFishAudioFallback:
    """Test fallback to Fish Audio S1 API."""

    def test_fallback_on_import_error(self, tmp_config: PetGenConfig):
        """Falls back to Fish Audio when chatterbox import fails."""
        tmp_config.fish_audio_api_key = "test-fish-key"
        gen = VoiceGenerator(tmp_config)

        # Simulate Chatterbox being unavailable
        with patch(
            "petgen.modules.voice.VoiceGenerator.model",
            new_callable=lambda: property(lambda self: None),
        ):
            gen._use_fallback = True

            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_resp = MagicMock()
                mock_resp.__enter__ = MagicMock(return_value=mock_resp)
                mock_resp.__exit__ = MagicMock(return_value=False)
                mock_resp.read.return_value = b"fake-audio-bytes"
                mock_urlopen.return_value = mock_resp

                output = tmp_config.temp_dir / "fallback.wav"
                result = gen.generate("Fish audio test", output_path=output)

                assert result == output
                assert output.read_bytes() == b"fake-audio-bytes"

    def test_fallback_no_api_key_raises(self, tmp_config: PetGenConfig):
        """Raises RuntimeError if Fish Audio key is missing and Chatterbox unavailable."""
        tmp_config.fish_audio_api_key = ""
        gen = VoiceGenerator(tmp_config)
        gen._use_fallback = True

        with pytest.raises(RuntimeError, match="Fish Audio API key not configured"):
            gen.generate("No key test")

    def test_chatterbox_failure_triggers_fallback(self, tmp_config: PetGenConfig):
        """If Chatterbox generate() raises, falls back to Fish Audio."""
        tmp_config.fish_audio_api_key = "test-fish-key"
        gen = VoiceGenerator(tmp_config)
        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("GPU OOM")
        gen._model = mock_model

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = b"fallback-audio"
            mock_urlopen.return_value = mock_resp

            output = tmp_config.temp_dir / "fallback2.wav"
            result = gen.generate("Fallback test", output_path=output)

            assert result == output
            assert gen._use_fallback is True


# ---------------------------------------------------------------------------
# generate_with_preset()
# ---------------------------------------------------------------------------


class TestGenerateWithPreset:
    """Test breed-group preset-based generation."""

    @patch.object(VoiceGenerator, "generate")
    def test_each_breed_group_has_preset(self, mock_gen, tmp_config: PetGenConfig):
        """Every BreedGroup in DEFAULT_PRESETS can be used with generate_with_preset."""
        gen = VoiceGenerator(tmp_config)
        mock_gen.return_value = tmp_config.temp_dir / "preset.wav"

        for breed_group in BreedGroup:
            result = gen.generate_with_preset("Test text", breed_group)
            assert result is not None

            call_kwargs = mock_gen.call_args[1]
            expected = DEFAULT_PRESETS[breed_group]
            assert call_kwargs["exaggeration"] == expected.exaggeration
            assert call_kwargs["speed"] == expected.speed

    @patch.object(VoiceGenerator, "generate")
    def test_preset_finds_reference_clip(self, mock_gen, tmp_config: PetGenConfig):
        """Picks up a reference clip from the presets directory."""
        ref_clip = _make_wav(tmp_config.voice_presets_dir / "large_dog.wav")
        mock_gen.return_value = tmp_config.temp_dir / "preset.wav"

        gen = VoiceGenerator(tmp_config)
        gen.generate_with_preset("Deep bark", BreedGroup.LARGE_DOG)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["voice_ref"] == ref_clip

    @patch.object(VoiceGenerator, "generate")
    def test_preset_finds_clip_in_subdirectory(self, mock_gen, tmp_config: PetGenConfig):
        """Picks up a reference clip from a breed-group subdirectory."""
        subdir = tmp_config.voice_presets_dir / "cat"
        ref_clip = _make_wav(subdir / "sassy_cat.wav")
        mock_gen.return_value = tmp_config.temp_dir / "preset.wav"

        gen = VoiceGenerator(tmp_config)
        gen.generate_with_preset("Meow", BreedGroup.CAT)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["voice_ref"] == ref_clip

    @patch.object(VoiceGenerator, "generate")
    def test_preset_no_clip_passes_none(self, mock_gen, tmp_config: PetGenConfig):
        """When no reference clip exists, voice_ref is None."""
        mock_gen.return_value = tmp_config.temp_dir / "preset.wav"

        gen = VoiceGenerator(tmp_config)
        gen.generate_with_preset("No clip", BreedGroup.PUPPY)

        call_kwargs = mock_gen.call_args[1]
        assert call_kwargs["voice_ref"] is None


# ---------------------------------------------------------------------------
# list_presets()
# ---------------------------------------------------------------------------


class TestListPresets:
    """Test preset discovery and listing."""

    def test_empty_dir(self, tmp_config: PetGenConfig):
        """Returns empty list when no presets exist."""
        gen = VoiceGenerator(tmp_config)
        assert gen.list_presets() == []

    def test_lists_wav_files(self, tmp_config: PetGenConfig):
        """Discovers .wav files in the presets directory."""
        _make_wav(tmp_config.voice_presets_dir / "small_dog.wav")
        _make_wav(tmp_config.voice_presets_dir / "cat.wav")

        gen = VoiceGenerator(tmp_config)
        presets = gen.list_presets()

        assert len(presets) == 2
        names = {p.name for p in presets}
        assert names == {"small_dog", "cat"}

    def test_lists_presets_with_metadata(self, tmp_config: PetGenConfig):
        """Reads metadata JSON alongside .wav files."""
        _make_wav(tmp_config.voice_presets_dir / "custom.wav")
        meta = {
            "name": "custom",
            "breed_group": "kitten",
            "exaggeration": 0.8,
            "pitch_shift": 0.0,
            "speed": 1.15,
            "description": "A tiny squeaky kitten",
        }
        (tmp_config.voice_presets_dir / "custom.json").write_text(json.dumps(meta))

        gen = VoiceGenerator(tmp_config)
        presets = gen.list_presets()

        assert len(presets) == 1
        p = presets[0]
        assert p.name == "custom"
        assert p.breed_group == BreedGroup.KITTEN
        assert p.settings.exaggeration == 0.8
        assert p.settings.speed == 1.15
        assert p.description == "A tiny squeaky kitten"

    def test_infers_breed_from_filename(self, tmp_config: PetGenConfig):
        """Infers breed group from WAV filename matching BreedGroup value."""
        _make_wav(tmp_config.voice_presets_dir / "puppy.wav")

        gen = VoiceGenerator(tmp_config)
        presets = gen.list_presets()

        assert len(presets) == 1
        assert presets[0].breed_group == BreedGroup.PUPPY

    def test_infers_breed_from_parent_dir(self, tmp_config: PetGenConfig):
        """Infers breed group from parent directory name."""
        _make_wav(tmp_config.voice_presets_dir / "large_dog" / "baritone.wav")

        gen = VoiceGenerator(tmp_config)
        presets = gen.list_presets()

        assert len(presets) == 1
        assert presets[0].breed_group == BreedGroup.LARGE_DOG

    def test_unknown_name_defaults_medium_dog(self, tmp_config: PetGenConfig):
        """Unknown filenames default to MEDIUM_DOG."""
        _make_wav(tmp_config.voice_presets_dir / "mystery_voice.wav")

        gen = VoiceGenerator(tmp_config)
        presets = gen.list_presets()

        assert len(presets) == 1
        assert presets[0].breed_group == BreedGroup.MEDIUM_DOG


# ---------------------------------------------------------------------------
# create_preset()
# ---------------------------------------------------------------------------


class TestCreatePreset:
    """Test custom preset creation."""

    def test_create_copies_clip_and_metadata(self, tmp_config: PetGenConfig):
        """Creates a preset by copying the clip and writing metadata JSON."""
        source = _make_wav(tmp_config.temp_dir / "source.wav")

        gen = VoiceGenerator(tmp_config)
        preset = gen.create_preset(
            name="my_cat",
            breed_group=BreedGroup.CAT,
            reference_clip=source,
            description="Sassy cat voice",
        )

        assert preset.name == "my_cat"
        assert preset.breed_group == BreedGroup.CAT
        assert preset.reference_clip == tmp_config.voice_presets_dir / "my_cat.wav"
        assert preset.reference_clip.is_file()
        assert preset.description == "Sassy cat voice"

        # Verify metadata file
        meta_path = tmp_config.voice_presets_dir / "my_cat.json"
        assert meta_path.is_file()
        meta = json.loads(meta_path.read_text())
        assert meta["breed_group"] == "cat"
        assert meta["exaggeration"] == 0.4  # from DEFAULT_PRESETS[CAT]
        assert meta["speed"] == 0.95

    def test_create_with_custom_settings(self, tmp_config: PetGenConfig):
        """Creates a preset with custom VoiceSettings overrides."""
        source = _make_wav(tmp_config.temp_dir / "source2.wav")

        gen = VoiceGenerator(tmp_config)
        custom = VoiceSettings(exaggeration=0.3, speed=0.8, pitch_shift=-2.0)
        preset = gen.create_preset(
            name="deep_bark",
            breed_group=BreedGroup.LARGE_DOG,
            reference_clip=source,
            settings=custom,
        )

        assert preset.settings.exaggeration == 0.3
        assert preset.settings.speed == 0.8
        assert preset.settings.pitch_shift == -2.0
        assert preset.settings.preset_name == "deep_bark"

    def test_created_preset_appears_in_list(self, tmp_config: PetGenConfig):
        """A created preset is discoverable by list_presets()."""
        source = _make_wav(tmp_config.temp_dir / "source3.wav")

        gen = VoiceGenerator(tmp_config)
        gen.create_preset(
            name="test_list",
            breed_group=BreedGroup.KITTEN,
            reference_clip=source,
        )

        presets = gen.list_presets()
        assert any(p.name == "test_list" for p in presets)


# ---------------------------------------------------------------------------
# Model lazy loading
# ---------------------------------------------------------------------------


class TestModelLoading:
    """Test lazy model loading and fallback detection."""

    def test_model_starts_none(self, tmp_config: PetGenConfig):
        """Model is not loaded at init time."""
        gen = VoiceGenerator(tmp_config)
        assert gen._model is None
        assert gen._use_fallback is False

    def test_fallback_flag_set_on_import_failure(self, tmp_config: PetGenConfig):
        """Sets _use_fallback when chatterbox import fails."""
        gen = VoiceGenerator(tmp_config)

        with patch.dict("sys.modules", {"chatterbox": None, "chatterbox.tts": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'chatterbox'"),
            ):
                _ = gen.model

        assert gen._use_fallback is True
        assert gen._model is None
