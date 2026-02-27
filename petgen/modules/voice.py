"""Voice generation module using Chatterbox Turbo TTS."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from petgen.config import PetGenConfig
from petgen.models import BreedGroup, VoiceSettings

logger = logging.getLogger(__name__)


@dataclass
class VoicePreset:
    """A pre-configured voice reference with settings."""

    name: str
    breed_group: BreedGroup
    reference_clip: Path
    settings: VoiceSettings
    description: str = ""


# Default voice presets mapped to breed groups
DEFAULT_PRESETS: dict[BreedGroup, VoiceSettings] = {
    BreedGroup.SMALL_DOG: VoiceSettings(preset_name="small_dog", exaggeration=0.8, speed=1.1),
    BreedGroup.MEDIUM_DOG: VoiceSettings(preset_name="medium_dog", exaggeration=0.6, speed=1.0),
    BreedGroup.LARGE_DOG: VoiceSettings(preset_name="large_dog", exaggeration=0.5, speed=0.9),
    BreedGroup.PUPPY: VoiceSettings(preset_name="puppy", exaggeration=0.9, speed=1.2),
    BreedGroup.CAT: VoiceSettings(preset_name="cat", exaggeration=0.4, speed=0.95),
    BreedGroup.KITTEN: VoiceSettings(preset_name="kitten", exaggeration=0.7, speed=1.1),
}

# Fish Audio S1 API endpoint
FISH_AUDIO_API_URL = "https://api.fish.audio/v1/tts"


class VoiceGenerator:
    """Generates character voice audio from text scripts."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._model = None
        self._use_fallback = False

    @property
    def model(self):
        """Lazy-load ChatterboxTTS model. Falls back to Fish Audio if unavailable."""
        if self._model is None and not self._use_fallback:
            try:
                from chatterbox.tts import ChatterboxTTS

                self._model = ChatterboxTTS.from_pretrained(device=self.config.tts_device)
                logger.info("Loaded ChatterboxTTS on %s", self.config.tts_device)
            except Exception as exc:
                logger.warning(
                    "ChatterboxTTS unavailable (%s), falling back to Fish Audio S1", exc
                )
                self._use_fallback = True
        return self._model

    def generate(
        self,
        text: str,
        voice_ref: Path | None = None,
        exaggeration: float = 0.7,
        speed: float = 1.0,
        output_path: Path | None = None,
    ) -> Path:
        """Generate speech audio from text using Chatterbox Turbo.

        Falls back to Fish Audio S1 API if Chatterbox is unavailable.
        """
        if output_path is None:
            self.config.temp_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.config.temp_dir / f"voice_{uuid.uuid4().hex[:8]}.wav"

        # Try Chatterbox first, fall back to Fish Audio
        if not self._use_fallback:
            try:
                return self._generate_chatterbox(
                    text, voice_ref, exaggeration, speed, output_path
                )
            except Exception as exc:
                logger.warning("Chatterbox generation failed (%s), trying Fish Audio", exc)
                self._use_fallback = True

        return self._generate_fish_audio(text, output_path)

    def _generate_chatterbox(
        self,
        text: str,
        voice_ref: Path | None,
        exaggeration: float,
        speed: float,
        output_path: Path,
    ) -> Path:
        """Generate audio using ChatterboxTTS."""
        import torch
        import torchaudio

        model = self.model

        kwargs: dict = {
            "text": text,
            "exaggeration": exaggeration,
        }
        if voice_ref is not None:
            kwargs["audio_prompt"] = str(voice_ref)

        wav = model.generate(**kwargs)

        # Ensure tensor is 2D: (channels, samples)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Normalize to prevent clipping
        peak = wav.abs().max()
        if peak > 0:
            wav = wav / peak * 0.95

        # Apply speed adjustment via resampling
        sample_rate = model.sr
        if speed != 1.0 and speed > 0:
            target_rate = int(sample_rate * speed)
            wav = torchaudio.functional.resample(wav, orig_freq=target_rate, new_freq=sample_rate)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), wav, sample_rate)
        logger.info("Saved Chatterbox audio to %s", output_path)
        return output_path

    def _generate_fish_audio(self, text: str, output_path: Path) -> Path:
        """Generate audio using Fish Audio S1 API as fallback."""
        import urllib.request

        api_key = self.config.fish_audio_api_key
        if not api_key:
            raise RuntimeError(
                "Fish Audio API key not configured (set PETGEN_FISH_AUDIO_API_KEY) "
                "and Chatterbox TTS is unavailable."
            )

        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            FISH_AUDIO_API_URL,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(resp.read())

        logger.info("Saved Fish Audio audio to %s", output_path)
        return output_path

    def generate_with_preset(
        self,
        text: str,
        breed_group: BreedGroup,
        output_path: Path | None = None,
    ) -> Path:
        """Generate speech using breed-aware voice preset."""
        settings = DEFAULT_PRESETS.get(breed_group)
        if settings is None:
            raise ValueError(f"No default preset for breed group: {breed_group}")

        # Look for a matching reference clip in the presets directory
        voice_ref = self._find_preset_clip(breed_group)

        return self.generate(
            text=text,
            voice_ref=voice_ref,
            exaggeration=settings.exaggeration,
            speed=settings.speed,
            output_path=output_path,
        )

    def _find_preset_clip(self, breed_group: BreedGroup) -> Path | None:
        """Find a voice reference clip for a breed group in the presets directory."""
        presets_dir = self.config.voice_presets_dir
        if not presets_dir.is_dir():
            return None

        # Look for <breed_group_value>.wav or <breed_group_value>/*.wav
        direct = presets_dir / f"{breed_group.value}.wav"
        if direct.is_file():
            return direct

        group_dir = presets_dir / breed_group.value
        if group_dir.is_dir():
            wavs = sorted(group_dir.glob("*.wav"))
            if wavs:
                return wavs[0]

        return None

    def list_presets(self) -> list[VoicePreset]:
        """List available voice presets from the presets directory."""
        presets: list[VoicePreset] = []
        presets_dir = self.config.voice_presets_dir

        if not presets_dir.is_dir():
            return presets

        for wav_file in sorted(presets_dir.rglob("*.wav")):
            # Try to load metadata JSON alongside the clip
            meta_path = wav_file.with_suffix(".json")
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                breed_group = BreedGroup(meta.get("breed_group", "small_dog"))
                settings = VoiceSettings(
                    preset_name=meta.get("name", wav_file.stem),
                    exaggeration=meta.get("exaggeration", 0.7),
                    pitch_shift=meta.get("pitch_shift", 0.0),
                    speed=meta.get("speed", 1.0),
                    reference_clip=wav_file,
                )
                presets.append(
                    VoicePreset(
                        name=meta.get("name", wav_file.stem),
                        breed_group=breed_group,
                        reference_clip=wav_file,
                        settings=settings,
                        description=meta.get("description", ""),
                    )
                )
            else:
                # Infer breed group from filename or parent directory
                breed_group = self._infer_breed_group(wav_file)
                default_settings = DEFAULT_PRESETS.get(breed_group, VoiceSettings())
                presets.append(
                    VoicePreset(
                        name=wav_file.stem,
                        breed_group=breed_group,
                        reference_clip=wav_file,
                        settings=VoiceSettings(
                            preset_name=wav_file.stem,
                            exaggeration=default_settings.exaggeration,
                            speed=default_settings.speed,
                            reference_clip=wav_file,
                        ),
                    )
                )

        return presets

    def _infer_breed_group(self, wav_file: Path) -> BreedGroup:
        """Infer breed group from a WAV file's name or parent directory."""
        candidates = [wav_file.stem, wav_file.parent.name]
        for candidate in candidates:
            for bg in BreedGroup:
                if bg.value == candidate:
                    return bg
        return BreedGroup.MEDIUM_DOG  # safe default

    def create_preset(
        self,
        name: str,
        breed_group: BreedGroup,
        reference_clip: Path,
        settings: VoiceSettings | None = None,
        description: str = "",
    ) -> VoicePreset:
        """Create a custom voice preset from a reference clip."""
        presets_dir = self.config.voice_presets_dir
        presets_dir.mkdir(parents=True, exist_ok=True)

        dest_clip = presets_dir / f"{name}.wav"
        shutil.copy2(str(reference_clip), str(dest_clip))

        if settings is None:
            settings = DEFAULT_PRESETS.get(
                breed_group,
                VoiceSettings(preset_name=name, exaggeration=0.7, speed=1.0),
            )
        # Ensure preset_name and reference_clip are set
        settings = VoiceSettings(
            preset_name=name,
            reference_clip=dest_clip,
            exaggeration=settings.exaggeration,
            pitch_shift=settings.pitch_shift,
            speed=settings.speed,
        )

        meta = {
            "name": name,
            "breed_group": breed_group.value,
            "exaggeration": settings.exaggeration,
            "pitch_shift": settings.pitch_shift,
            "speed": settings.speed,
            "description": description,
        }
        meta_path = presets_dir / f"{name}.json"
        meta_path.write_text(json.dumps(meta, indent=2))

        preset = VoicePreset(
            name=name,
            breed_group=breed_group,
            reference_clip=dest_clip,
            settings=settings,
            description=description,
        )
        logger.info("Created voice preset '%s' at %s", name, dest_clip)
        return preset
