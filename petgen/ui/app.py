"""Gradio web interface for PetGen."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from petgen.config import PetGenConfig, get_config
from petgen.models import AnimationConfig, AspectRatio
from petgen.pipeline import PetGenPipeline

logger = logging.getLogger(__name__)

# Aspect ratio display label -> enum value
_ASPECT_LABELS = {
    "9:16 (Vertical)": AspectRatio.VERTICAL,
    "1:1 (Square)": AspectRatio.SQUARE,
    "16:9 (Landscape)": AspectRatio.LANDSCAPE,
}

# Aspect ratio -> (width, height)
_ASPECT_RESOLUTIONS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.VERTICAL: (1080, 1920),
    AspectRatio.SQUARE: (1080, 1080),
    AspectRatio.LANDSCAPE: (1920, 1080),
}


def _get_pipeline() -> tuple[PetGenConfig, PetGenPipeline]:
    """Return a shared (config, pipeline) pair. Lazily initialized."""
    if not hasattr(_get_pipeline, "_cache"):
        config = get_config()
        _get_pipeline._cache = (config, PetGenPipeline(config))
    return _get_pipeline._cache


def get_character_choices() -> list[tuple[str, str]]:
    """Return a list of (display_label, character_id) tuples."""
    _, pipeline = _get_pipeline()
    try:
        chars = pipeline.character_creator.list_characters()
    except Exception:
        return []
    return [(f"{c.name} ({c.breed})", c.id) for c in chars]


def _refresh_character_dropdown() -> dict[str, Any]:
    """Return an update dict for a character dropdown."""
    choices = get_character_choices()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


# ---------------------------------------------------------------------------
# Tab handler functions
# ---------------------------------------------------------------------------


def handle_create_character(
    files: list[str] | None,
    name: str,
    breed: str,
) -> tuple[list[str] | None, dict[str, Any] | str, dict[str, Any]]:
    """Create a character from uploaded photos and return gallery + profile info."""
    if not files:
        gr.Warning("Please upload at least one pet photo.")
        return None, "No photos provided", gr.update()

    if not name.strip():
        gr.Warning("Please enter a name for your pet.")
        return None, "No name provided", gr.update()

    _, pipeline = _get_pipeline()
    photo_paths = [Path(f) for f in files]
    breed_val = breed.strip() if breed and breed.strip() else None

    try:
        profile = pipeline.create_character(
            name=name.strip(),
            photos=photo_paths,
            breed=breed_val,
        )
    except Exception as exc:
        gr.Warning(f"Character creation failed: {exc}")
        logger.exception("Character creation failed")
        return None, f"Error: {exc}", gr.update()

    # Gather pose images for gallery
    pose_images = [str(p) for p in profile.canonical_poses.values() if Path(p).is_file()]

    profile_info = {
        "id": profile.id,
        "name": profile.name,
        "breed": profile.breed,
        "breed_group": profile.breed_group.value,
        "poses": list(profile.canonical_poses.keys()),
        "reference_images": len(profile.reference_images),
    }

    gr.Info(f"Character '{profile.name}' created successfully!")
    return pose_images or None, profile_info, _refresh_character_dropdown()


def handle_generate_video(
    char_id: str,
    script: str,
    scene_desc: str,
    emotion: float,
    speed: float,
    aspect_label: str,
) -> tuple[str | None, str]:
    """Generate a talking pet video and return the video path + stats."""
    if not char_id:
        gr.Warning("Please select a character first.")
        return None, ""

    if not script.strip():
        gr.Warning("Please enter a script for your pet to say.")
        return None, ""

    config, pipeline = _get_pipeline()

    profile = pipeline.character_creator.get_character(char_id)
    if profile is None:
        gr.Warning(f"Character '{char_id}' not found.")
        return None, ""

    profile.voice.exaggeration = emotion
    profile.voice.speed = speed

    aspect_ratio = _ASPECT_LABELS.get(aspect_label, AspectRatio.VERTICAL)
    resolution = _ASPECT_RESOLUTIONS[aspect_ratio]

    animation_config = AnimationConfig(
        fps=config.default_fps,
        resolution=resolution,
        aspect_ratio=aspect_ratio,
    )

    try:
        result = pipeline.generate(
            character=profile,
            script=script.strip(),
            scene_prompt=scene_desc.strip() or None,
            animation_config=animation_config,
        )
    except Exception as exc:
        gr.Warning(f"Video generation failed: {exc}")
        logger.exception("Video generation failed")
        return None, f"Error: {exc}"

    elapsed = result.metadata.get("elapsed_seconds", 0)
    stats = (
        f"Duration: {result.duration:.1f}s | "
        f"Frames: {result.num_frames} | "
        f"Time: {elapsed:.1f}s"
    )
    return str(result.video_path), stats


def handle_voice_preview(
    char_id: str,
    text: str,
    emotion: float,
) -> str | None:
    """Generate a voice preview and return the audio path."""
    if not char_id:
        gr.Warning("Please select a character first.")
        return None

    if not text.strip():
        gr.Warning("Please enter some text.")
        return None

    _, pipeline = _get_pipeline()

    profile = pipeline.character_creator.get_character(char_id)
    if profile is None:
        gr.Warning(f"Character '{char_id}' not found.")
        return None

    try:
        audio_path = pipeline.voice_generator.generate(
            text=text.strip(),
            voice_ref=profile.voice.reference_clip,
            exaggeration=emotion,
            speed=profile.voice.speed,
        )
    except Exception as exc:
        gr.Warning(f"Voice preview failed: {exc}")
        logger.exception("Voice preview failed")
        return None

    return str(audio_path)


def handle_scene_generate(
    char_id: str,
    scene_desc: str,
    style: str,
) -> str | None:
    """Generate a scene image and return the image path."""
    if not char_id:
        gr.Warning("Please select a character first.")
        return None

    if not scene_desc.strip():
        gr.Warning("Please enter a scene description.")
        return None

    _, pipeline = _get_pipeline()

    profile = pipeline.character_creator.get_character(char_id)
    if profile is None:
        gr.Warning(f"Character '{char_id}' not found.")
        return None

    try:
        scene_path = pipeline.character_creator.generate_scene(
            profile=profile,
            prompt=scene_desc.strip(),
            style=style,
        )
    except Exception as exc:
        gr.Warning(f"Scene generation failed: {exc}")
        logger.exception("Scene generation failed")
        return None

    return str(scene_path)


def handle_gallery_refresh() -> list[str] | None:
    """Return all video files in the output directory."""
    config, _ = _get_pipeline()
    output_dir = config.output_dir
    if not output_dir.is_dir():
        return None

    videos = sorted(
        (p for p in output_dir.iterdir() if p.suffix.lower() in (".mp4", ".webm", ".mov")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return [str(v) for v in videos] if videos else None


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------


def create_ui() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    with gr.Blocks(title="PetGen", theme=gr.themes.Soft()) as app:
        gr.Markdown("# PetGen\nAI-powered talking pet video generator")

        # ------------------------------------------------------------------
        # Tab 1: Create Character
        # ------------------------------------------------------------------
        with gr.Tab("Create Character"):
            with gr.Row():
                with gr.Column():
                    photo_upload = gr.File(
                        label="Pet Photos",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    pet_name = gr.Textbox(label="Pet Name", placeholder="e.g. Max")
                    pet_breed = gr.Textbox(
                        label="Breed (optional)",
                        placeholder="Auto-detect",
                    )
                    create_btn = gr.Button("Create Character", variant="primary")
                with gr.Column():
                    pose_gallery = gr.Gallery(label="Generated Canonical Poses")
                    profile_json = gr.JSON(label="Character Profile")

            # Hidden dropdown to refresh across tabs
            create_char_dropdown = gr.Dropdown(
                label="Existing Characters",
                choices=get_character_choices(),
                interactive=True,
            )

        # ------------------------------------------------------------------
        # Tab 2: Generate Video
        # ------------------------------------------------------------------
        with gr.Tab("Generate Video"):
            with gr.Row():
                with gr.Column():
                    vid_char = gr.Dropdown(
                        label="Character",
                        choices=get_character_choices(),
                        interactive=True,
                    )
                    vid_script = gr.Textbox(
                        label="Script",
                        lines=3,
                        placeholder="What should your pet say?",
                    )
                    vid_scene = gr.Textbox(
                        label="Scene Description (optional)",
                        placeholder="e.g. sitting in a sunny park",
                    )
                    vid_emotion = gr.Slider(
                        label="Emotion Exaggeration",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                    )
                    vid_speed = gr.Slider(
                        label="Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                    )
                    vid_aspect = gr.Radio(
                        label="Aspect Ratio",
                        choices=list(_ASPECT_LABELS.keys()),
                        value="9:16 (Vertical)",
                    )
                    vid_btn = gr.Button("Generate Video", variant="primary")
                with gr.Column():
                    vid_output = gr.Video(label="Generated Video")
                    vid_stats = gr.Textbox(label="Generation Stats", interactive=False)

        # ------------------------------------------------------------------
        # Tab 3: Voice Preview
        # ------------------------------------------------------------------
        with gr.Tab("Voice Preview"):
            with gr.Row():
                with gr.Column():
                    voice_char = gr.Dropdown(
                        label="Character",
                        choices=get_character_choices(),
                        interactive=True,
                    )
                    voice_text = gr.Textbox(
                        label="Text",
                        lines=2,
                        placeholder="Enter text to preview...",
                    )
                    voice_emotion = gr.Slider(
                        label="Emotion",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                    )
                    voice_btn = gr.Button("Preview Voice", variant="primary")
                with gr.Column():
                    voice_audio = gr.Audio(label="Voice Preview")

        # ------------------------------------------------------------------
        # Tab 4: Scene Generator
        # ------------------------------------------------------------------
        with gr.Tab("Scene Generator"):
            with gr.Row():
                with gr.Column():
                    scene_char = gr.Dropdown(
                        label="Character",
                        choices=get_character_choices(),
                        interactive=True,
                    )
                    scene_desc = gr.Textbox(
                        label="Scene Description",
                        lines=2,
                        placeholder="Describe the scene...",
                    )
                    scene_style = gr.Dropdown(
                        label="Style",
                        choices=[
                            "photorealistic",
                            "cartoon",
                            "watercolor",
                            "oil painting",
                            "pixel art",
                        ],
                        value="photorealistic",
                    )
                    scene_btn = gr.Button("Generate Scene", variant="primary")
                with gr.Column():
                    scene_image = gr.Image(label="Generated Scene")

        # ------------------------------------------------------------------
        # Tab 5: Gallery
        # ------------------------------------------------------------------
        with gr.Tab("Gallery"):
            gallery_display = gr.Gallery(label="Generated Videos")
            gallery_refresh_btn = gr.Button("Refresh")

        # ------------------------------------------------------------------
        # Wire up event handlers
        # ------------------------------------------------------------------

        # All character dropdowns that need refreshing
        char_dropdowns = [create_char_dropdown, vid_char, voice_char, scene_char]

        create_btn.click(
            fn=handle_create_character,
            inputs=[photo_upload, pet_name, pet_breed],
            outputs=[pose_gallery, profile_json, create_char_dropdown],
        ).then(
            fn=_refresh_character_dropdown,
            outputs=vid_char,
        ).then(
            fn=_refresh_character_dropdown,
            outputs=voice_char,
        ).then(
            fn=_refresh_character_dropdown,
            outputs=scene_char,
        )

        vid_btn.click(
            fn=handle_generate_video,
            inputs=[vid_char, vid_script, vid_scene, vid_emotion, vid_speed, vid_aspect],
            outputs=[vid_output, vid_stats],
        )

        voice_btn.click(
            fn=handle_voice_preview,
            inputs=[voice_char, voice_text, voice_emotion],
            outputs=[voice_audio],
        )

        scene_btn.click(
            fn=handle_scene_generate,
            inputs=[scene_char, scene_desc, scene_style],
            outputs=[scene_image],
        )

        gallery_refresh_btn.click(
            fn=handle_gallery_refresh,
            outputs=[gallery_display],
        )

    return app
