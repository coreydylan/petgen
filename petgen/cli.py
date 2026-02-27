"""PetGen CLI — command-line interface for the talking pet video generator."""

from __future__ import annotations

from pathlib import Path

import click

from petgen.config import get_config


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """PetGen — AI-powered talking pet video generator."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = get_config()


@main.command()
@click.option("--name", required=True, help="Name for the pet character")
@click.option("--photos", required=True, type=click.Path(exists=True), help="Directory of pet photos")
@click.option("--breed", default=None, help="Breed (auto-detected if omitted)")
@click.pass_context
def create_character(ctx: click.Context, name: str, photos: str, breed: str | None) -> None:
    """Create a new pet character from reference photos."""
    raise NotImplementedError("TODO: implement in Task #6")


@main.command()
@click.option("--character", required=True, help="Character name or ID")
@click.option("--script", required=True, help="Text for the pet to say")
@click.option("--scene", default=None, help="Scene description prompt")
@click.option("--voice-preset", default=None, help="Voice preset name")
@click.option("--emotion", default=0.7, type=float, help="Emotion exaggeration (0-1)")
@click.option("--aspect", default="9:16", type=click.Choice(["9:16", "1:1", "16:9"]))
@click.option("--output", default=None, type=click.Path(), help="Output video path")
@click.pass_context
def generate(
    ctx: click.Context,
    character: str,
    script: str,
    scene: str | None,
    voice_preset: str | None,
    emotion: float,
    aspect: str,
    output: str | None,
) -> None:
    """Generate a talking pet video from a character and script."""
    raise NotImplementedError("TODO: implement in Task #6")


@main.command()
@click.option("--character", required=True, help="Character name or ID")
@click.option("--script", required=True, help="Text to speak")
@click.option("--emotion", default=0.7, type=float, help="Emotion exaggeration (0-1)")
@click.option("--output", default=None, type=click.Path(), help="Output WAV path")
@click.pass_context
def voice(ctx: click.Context, character: str, script: str, emotion: float, output: str | None) -> None:
    """Generate voice audio only."""
    raise NotImplementedError("TODO: implement in Task #6")


@main.command()
@click.option("--character", required=True, help="Character name or ID")
@click.option("--prompt", required=True, help="Scene description")
@click.option("--style", default="photorealistic", help="Image style")
@click.option("--output", default=None, type=click.Path(), help="Output image path")
@click.pass_context
def scene(ctx: click.Context, character: str, prompt: str, style: str, output: str | None) -> None:
    """Generate a scene image with the character."""
    raise NotImplementedError("TODO: implement in Task #6")


@main.command(name="list-characters")
@click.pass_context
def list_characters(ctx: click.Context) -> None:
    """List all saved pet characters."""
    raise NotImplementedError("TODO: implement in Task #6")


@main.command(name="list-presets")
@click.pass_context
def list_presets(ctx: click.Context) -> None:
    """List available voice presets."""
    raise NotImplementedError("TODO: implement in Task #6")


if __name__ == "__main__":
    main()
