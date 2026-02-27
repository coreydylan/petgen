"""Tests for the PetGen CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from petgen.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


import pytest


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
        with patch("petgen.cli.get_config") as mock_config:
            cfg = MagicMock()
            cfg.characters_dir = MagicMock()
            cfg.characters_dir.glob.return_value = []
            cfg.ensure_dirs.return_value = None
            mock_config.return_value = cfg

            result = runner.invoke(main, [
                "generate",
                "--character", "nonexistent",
                "--script", "Hello",
            ])
            assert result.exit_code != 0
            assert "not found" in result.output
