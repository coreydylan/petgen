"""Tests for the PetGen CLI."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

import pytest

from petgen.cli import main
from petgen.models import BreedGroup, CharacterProfile


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestCLI:
    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "PetGen" in result.output

    def test_create_character_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["create-character", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output
        assert "--photos" in result.output

    def test_generate_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--character" in result.output
        assert "--script" in result.output
        assert "--aspect" in result.output

    def test_voice_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["voice", "--help"])
        assert result.exit_code == 0
        assert "--character" in result.output
        assert "--script" in result.output

    def test_scene_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["scene", "--help"])
        assert result.exit_code == 0
        assert "--prompt" in result.output

    def test_list_characters_empty(self) -> None:
        runner = CliRunner()
        with patch("petgen.cli.get_config") as mock_config:
            cfg = MagicMock()
            cfg.characters_dir = MagicMock()
            cfg.characters_dir.glob.return_value = []
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg
            result = runner.invoke(main, ["list-characters"])
            assert result.exit_code == 0
            assert "No characters found" in result.output

    def test_list_presets_shows_defaults(self) -> None:
        runner = CliRunner()
        with patch("petgen.cli.get_config") as mock_config:
            cfg = MagicMock()
            cfg.voice_presets_dir = MagicMock()
            cfg.voice_presets_dir.is_dir.return_value = False
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg
            result = runner.invoke(main, ["list-presets"])
            assert result.exit_code == 0
            assert "small_dog" in result.output
            assert "large_dog" in result.output
            assert "cat" in result.output

    def test_generate_missing_character(self) -> None:
        runner = CliRunner()
        with patch("petgen.cli.get_config") as mock_config, \
             patch("petgen.pipeline.PetGenPipeline") as mock_pipeline_cls:
            cfg = MagicMock()
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg

            mock_pipeline = MagicMock()
            mock_pipeline.character_creator.get_character.return_value = None
            mock_pipeline_cls.return_value = mock_pipeline

            result = runner.invoke(main, [
                "generate",
                "--character", "nonexistent",
                "--script", "Hello",
            ])
            assert result.exit_code != 0
            assert "not found" in result.output

    def test_multi_scene_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["multi-scene", "--help"])
        assert result.exit_code == 0
        assert "--characters" in result.output
        assert "--prompt" in result.output

    def test_export_character_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["export-character", "--help"])
        assert result.exit_code == 0
        assert "--character" in result.output
        assert "--output" in result.output

    def test_import_character_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["import-character", "--help"])
        assert result.exit_code == 0
        assert "--zip" in result.output

    def test_clone_voice_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["clone-voice", "--help"])
        assert result.exit_code == 0
        assert "--character" in result.output
        assert "--audio" in result.output

    def test_multi_scene_missing_character(self) -> None:
        runner = CliRunner()
        with patch("petgen.cli.get_config") as mock_config, \
             patch("petgen.pipeline.PetGenPipeline") as mock_pipeline_cls:
            cfg = MagicMock()
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg

            mock_pipeline = MagicMock()
            mock_pipeline.character_creator.get_character.return_value = None
            mock_pipeline_cls.return_value = mock_pipeline

            result = runner.invoke(main, [
                "multi-scene",
                "--characters", "a,b",
                "--prompt", "playing",
            ])
            assert result.exit_code != 0
            assert "not found" in result.output

    def test_multi_scene_needs_two_characters(self) -> None:
        runner = CliRunner()
        with patch("petgen.cli.get_config") as mock_config, \
             patch("petgen.pipeline.PetGenPipeline") as mock_pipeline_cls:
            cfg = MagicMock()
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg

            result = runner.invoke(main, [
                "multi-scene",
                "--characters", "only_one",
                "--prompt", "solo",
            ])
            assert result.exit_code != 0
            assert "At least 2" in result.output

    def test_export_character_missing(self) -> None:
        runner = CliRunner()
        with patch("petgen.cli.get_config") as mock_config, \
             patch("petgen.modules.character.CharacterCreator") as mock_cc_cls:
            cfg = MagicMock()
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg

            mock_cc = MagicMock()
            mock_cc.get_character.return_value = None
            mock_cc_cls.return_value = mock_cc

            result = runner.invoke(main, [
                "export-character",
                "--character", "ghost",
                "--output", "/tmp/ghost.zip",
            ])
            assert result.exit_code != 0
            assert "not found" in result.output

    def test_clone_voice_missing_character(self) -> None:
        runner = CliRunner()
        with patch("petgen.cli.get_config") as mock_config, \
             patch("petgen.modules.character.CharacterCreator") as mock_cc_cls:
            cfg = MagicMock()
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg

            mock_cc = MagicMock()
            mock_cc.get_character.return_value = None
            mock_cc_cls.return_value = mock_cc

            # Create a temp wav file for the --audio arg
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                wf = wave.open(tf.name, "w")
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(struct.pack("<100h", *([0] * 100)))
                wf.close()
                wav_path = tf.name

            result = runner.invoke(main, [
                "clone-voice",
                "--character", "ghost",
                "--audio", wav_path,
            ])
            assert result.exit_code != 0
            assert "not found" in result.output

            Path(wav_path).unlink(missing_ok=True)
