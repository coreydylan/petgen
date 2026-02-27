"""PetGen CLI — command-line interface for the talking pet video generator."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from petgen.config import get_config

logger = logging.getLogger("petgen")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """PetGen — AI-powered talking pet video generator."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = get_config()


@main.command()
@click.option("--name", required=True, help="Name for the pet character")
@click.option("--photos", required=True, type=click.Path(exists=True), help="Directory of pet photos or single image")
@click.option("--breed", default=None, help="Breed (auto-detected if omitted)")
@click.pass_context
def create_character(ctx: click.Context, name: str, photos: str, breed: str | None) -> None:
    """Create a new pet character from reference photos."""
    from petgen.pipeline import PetGenPipeline

    config = ctx.obj["config"]
    pipeline = PetGenPipeline(config)

    photos_path = Path(photos)
    if photos_path.is_dir():
        image_files = sorted(
            p for p in photos_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )
        if not image_files:
            click.echo(f"Error: No image files found in {photos_path}", err=True)
            sys.exit(1)
    elif photos_path.is_file():
        image_files = [photos_path]
    else:
        click.echo(f"Error: {photos_path} is not a file or directory", err=True)
        sys.exit(1)

    click.echo(f"Creating character '{name}' from {len(image_files)} photo(s)...")
    profile = pipeline.create_character(name=name, photos=image_files, breed=breed)

    click.echo(f"Character created:")
    click.echo(f"  ID:    {profile.id}")
    click.echo(f"  Name:  {profile.name}")
    click.echo(f"  Breed: {profile.breed} ({profile.breed_group.value})")
    click.echo(f"  Poses: {', '.join(profile.canonical_poses.keys())}")


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
    from petgen.models import AnimationConfig, AspectRatio
    from petgen.pipeline import PetGenPipeline

    config = ctx.obj["config"]
    pipeline = PetGenPipeline(config)

    # Look up character
    profile = pipeline.character_creator.get_character(character)
    if profile is None:
        click.echo(f"Error: Character '{character}' not found.", err=True)
        click.echo("Use 'petgen list-characters' to see available characters.", err=True)
        sys.exit(1)

    # Override voice settings if provided
    profile.voice.exaggeration = emotion
    if voice_preset:
        profile.voice.preset_name = voice_preset

    # Map aspect ratio string to enum
    aspect_map = {"9:16": AspectRatio.VERTICAL, "1:1": AspectRatio.SQUARE, "16:9": AspectRatio.LANDSCAPE}
    aspect_ratio = aspect_map[aspect]

    # Resolution from aspect ratio
    res_map = {
        AspectRatio.VERTICAL: (1080, 1920),
        AspectRatio.SQUARE: (1080, 1080),
        AspectRatio.LANDSCAPE: (1920, 1080),
    }

    animation_config = AnimationConfig(
        fps=config.default_fps,
        resolution=res_map[aspect_ratio],
        aspect_ratio=aspect_ratio,
    )

    output_path = Path(output) if output else None

    click.echo(f"Generating talking video for '{profile.name}'...")
    click.echo(f"  Script: \"{script[:80]}{'...' if len(script) > 80 else ''}\"")
    click.echo(f"  Aspect: {aspect}")

    result = pipeline.generate(
        character=profile,
        script=script,
        scene_prompt=scene,
        animation_config=animation_config,
        output_path=output_path,
    )

    click.echo(f"Video saved: {result.video_path}")
    click.echo(f"  Duration: {result.duration:.1f}s ({result.num_frames} frames)")
    click.echo(f"  Time:     {result.metadata.get('elapsed_seconds', 0):.1f}s")


@main.command()
@click.option("--character", required=True, help="Character name or ID")
@click.option("--script", required=True, help="Text to speak")
@click.option("--emotion", default=0.7, type=float, help="Emotion exaggeration (0-1)")
@click.option("--output", default=None, type=click.Path(), help="Output WAV path")
@click.pass_context
def voice(ctx: click.Context, character: str, script: str, emotion: float, output: str | None) -> None:
    """Generate voice audio only."""
    from petgen.pipeline import PetGenPipeline

    config = ctx.obj["config"]
    pipeline = PetGenPipeline(config)

    profile = pipeline.character_creator.get_character(character)
    if profile is None:
        click.echo(f"Error: Character '{character}' not found.", err=True)
        sys.exit(1)

    output_path = Path(output) if output else None

    click.echo(f"Generating voice for '{profile.name}'...")
    audio_path = pipeline.voice_generator.generate(
        text=script,
        voice_ref=profile.voice.reference_clip,
        exaggeration=emotion,
        speed=profile.voice.speed,
        output_path=output_path,
    )
    click.echo(f"Audio saved: {audio_path}")


@main.command()
@click.option("--character", required=True, help="Character name or ID")
@click.option("--prompt", required=True, help="Scene description")
@click.option("--style", default="photorealistic", help="Image style")
@click.option("--output", default=None, type=click.Path(), help="Output image path")
@click.pass_context
def scene(ctx: click.Context, character: str, prompt: str, style: str, output: str | None) -> None:
    """Generate a scene image with the character."""
    from petgen.pipeline import PetGenPipeline

    config = ctx.obj["config"]
    pipeline = PetGenPipeline(config)

    profile = pipeline.character_creator.get_character(character)
    if profile is None:
        click.echo(f"Error: Character '{character}' not found.", err=True)
        sys.exit(1)

    click.echo(f"Generating scene for '{profile.name}'...")
    scene_path = pipeline.character_creator.generate_scene(
        profile=profile, prompt=prompt, style=style
    )

    if output:
        import shutil
        shutil.copy2(str(scene_path), output)
        click.echo(f"Scene saved: {output}")
    else:
        click.echo(f"Scene saved: {scene_path}")


@main.command(name="list-characters")
@click.pass_context
def list_characters(ctx: click.Context) -> None:
    """List all saved pet characters."""
    from petgen.modules.character import CharacterCreator

    config = ctx.obj["config"]
    creator = CharacterCreator(config)
    characters = creator.list_characters()

    if not characters:
        click.echo("No characters found. Create one with 'petgen create-character'.")
        return

    click.echo(f"Found {len(characters)} character(s):\n")
    for c in characters:
        poses = ", ".join(c.canonical_poses.keys()) if c.canonical_poses else "none"
        click.echo(f"  {c.name}")
        click.echo(f"    ID:    {c.id}")
        click.echo(f"    Breed: {c.breed} ({c.breed_group.value})")
        click.echo(f"    Refs:  {len(c.reference_images)} photos")
        click.echo(f"    Poses: {poses}")
        click.echo()


@main.command(name="list-presets")
@click.pass_context
def list_presets(ctx: click.Context) -> None:
    """List available voice presets."""
    from petgen.modules.voice import DEFAULT_PRESETS, VoiceGenerator

    config = ctx.obj["config"]
    generator = VoiceGenerator(config)

    click.echo("Default breed presets:")
    for breed_group, settings in DEFAULT_PRESETS.items():
        click.echo(f"  {breed_group.value:12s}  exaggeration={settings.exaggeration}  speed={settings.speed}")

    click.echo()
    custom = generator.list_presets()
    if custom:
        click.echo(f"Custom presets ({len(custom)}):")
        for p in custom:
            click.echo(f"  {p.name:16s}  breed={p.breed_group.value}  clip={p.reference_clip.name}")
    else:
        click.echo("No custom presets. Create one with a voice reference clip.")


if __name__ == "__main__":
    main()
