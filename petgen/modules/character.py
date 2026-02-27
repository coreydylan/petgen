"""Character creation module using Google Gemini / Nano Banana API."""

from __future__ import annotations

import json
import logging
import shutil
import uuid
import zipfile
from pathlib import Path

from PIL import Image

from petgen.config import PetGenConfig
from petgen.models import BreedGroup, CharacterProfile, GeminiModel, MultiCharacterScene

logger = logging.getLogger(__name__)

# Mapping from Gemini breed-detection output to BreedGroup
_BREED_GROUP_KEYWORDS: dict[str, BreedGroup] = {
    "puppy": BreedGroup.PUPPY,
    "kitten": BreedGroup.KITTEN,
    "cat": BreedGroup.CAT,
}

# Approximate adult-weight thresholds for dog size classification (pounds)
_SMALL_DOG_MAX_LB = 25
_MEDIUM_DOG_MAX_LB = 55

# Canonical pose definitions: name -> prompt fragment
_CANONICAL_POSES: dict[str, str] = {
    "front": (
        "front-facing portrait, looking directly at camera, "
        "ears visible, symmetrical face, mouth closed, neutral expression"
    ),
    "side": (
        "perfect side profile view, facing left, full body visible, "
        "clean silhouette, mouth closed, neutral expression"
    ),
    "three_quarter": (
        "three-quarter angle portrait, slight head turn, "
        "one ear partially behind head, mouth closed, neutral expression"
    ),
}

# Dark-fur advisory appended to prompts for breeds/photos with dark coloring
_DARK_FUR_ADVISORY = (
    "Maintain fine detail in dark fur — preserve texture, "
    "individual hair strands, and subtle color variation in shadows."
)


class CharacterCreator:
    """Creates and manages persistent pet characters using Gemini image generation."""

    def __init__(self, config: PetGenConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.config.gemini_api_key)
        return self._client

    def create_character(
        self,
        reference_images: list[Path],
        name: str,
        breed: str | None = None,
        breed_group: BreedGroup | None = None,
        model: GeminiModel = GeminiModel.NANO_BANANA_FAST,
        personality_tags: list[str] | None = None,
    ) -> CharacterProfile:
        """Create a new pet character from reference photos.

        Args:
            reference_images: 1-5 reference photos of the pet.
            name: Display name for the character.
            breed: Breed string. Auto-detected from first image if None.
            breed_group: Size/species group. Auto-detected if None.
            model: Gemini model to use for generation.
            personality_tags: Optional personality descriptors.

        Returns:
            A fully initialized CharacterProfile with canonical poses generated.
        """
        if not reference_images:
            raise ValueError("At least one reference image is required.")
        for img_path in reference_images:
            if not img_path.exists():
                raise FileNotFoundError(f"Reference image not found: {img_path}")

        # Auto-detect breed if not provided
        if breed is None or breed_group is None:
            detected_breed, detected_group = self.detect_breed(reference_images[0])
            breed = breed or detected_breed
            breed_group = breed_group or detected_group

        character_id = str(uuid.uuid4())

        # Create directory structure
        char_dir = self.config.characters_dir / character_id
        images_dir = char_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Copy reference images into character directory
        saved_refs: list[Path] = []
        for i, src in enumerate(reference_images):
            dest = images_dir / f"ref_{i}{src.suffix}"
            img = Image.open(src)
            img.save(dest)
            saved_refs.append(dest)

        profile = CharacterProfile(
            id=character_id,
            name=name,
            breed=breed,
            breed_group=breed_group,
            reference_images=saved_refs,
            personality_tags=personality_tags or [],
        )

        # Generate canonical poses for animation source frames
        canonical_poses = self.generate_canonical_poses(profile, model=model)
        profile.canonical_poses = canonical_poses

        # Persist profile
        profile.save(self.config.characters_dir)

        logger.info("Created character '%s' (%s) — %s", name, breed, character_id)
        return profile

    def generate_canonical_poses(
        self,
        profile: CharacterProfile,
        model: GeminiModel = GeminiModel.NANO_BANANA_FAST,
    ) -> dict[str, Path]:
        """Generate front, side, and 3/4 canonical poses for animation source frames.

        Uses the character's reference images as identity anchors and generates
        each pose with identity-preserving prompts.

        Returns:
            Mapping of pose name to saved image path.
        """
        images_dir = self.config.characters_dir / profile.id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Load reference images — the Gemini SDK auto-converts PIL Images to Parts
        ref_parts = []
        for ref_path in profile.reference_images:
            ref_parts.append(Image.open(ref_path))

        is_dark = _is_likely_dark_fur(profile.breed)
        poses: dict[str, Path] = {}

        for pose_name, pose_prompt_fragment in _CANONICAL_POSES.items():
            prompt = (
                f"Generate this exact {profile.breed} in a {pose_prompt_fragment}, "
                f"photorealistic, studio lighting, white background, high detail."
            )
            if is_dark:
                prompt += f" {_DARK_FUR_ADVISORY}"

            contents = [*ref_parts, prompt]
            response = self.client.models.generate_content(
                model=model.value,
                contents=contents,
            )

            # Extract generated image from response
            out_path = images_dir / f"pose_{pose_name}.png"
            _save_image_from_response(response, out_path)
            poses[pose_name] = out_path
            logger.debug("Generated canonical pose '%s' for %s", pose_name, profile.id)

        return poses

    def generate_scene(
        self,
        profile: CharacterProfile,
        prompt: str,
        style: str = "photorealistic",
        model: GeminiModel = GeminiModel.NANO_BANANA_FAST,
    ) -> Path:
        """Generate a scene image with the character in a described setting.

        Args:
            profile: The character to place in the scene.
            prompt: Scene description (e.g. "wearing a pirate hat on a sailing ship").
            style: Visual style for the generation.
            model: Gemini model to use.

        Returns:
            Path to the saved scene image.
        """
        images_dir = self.config.characters_dir / profile.id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Build reference parts — prefer canonical front pose + original refs
        # The Gemini SDK auto-converts PIL Images to inline parts
        ref_parts = []
        if "front" in profile.canonical_poses:
            ref_parts.append(Image.open(profile.canonical_poses["front"]))
        for ref_path in profile.reference_images[:2]:
            ref_parts.append(Image.open(ref_path))

        is_dark = _is_likely_dark_fur(profile.breed)
        full_prompt = (
            f"Generate this exact {profile.breed} {prompt}, "
            f"{style}, front-facing, mouth closed, neutral expression"
        )
        if is_dark:
            full_prompt += f". {_DARK_FUR_ADVISORY}"

        contents = [*ref_parts, full_prompt]
        response = self.client.models.generate_content(
            model=model.value,
            contents=contents,
        )

        scene_id = uuid.uuid4().hex[:8]
        out_path = images_dir / f"scene_{scene_id}.png"
        _save_image_from_response(response, out_path)
        logger.info("Generated scene for %s: %s", profile.id, out_path.name)
        return out_path

    def generate_mouth_open_ref(
        self,
        profile: CharacterProfile,
        model: GeminiModel = GeminiModel.NANO_BANANA_FAST,
    ) -> Path:
        """Generate a reference image of the character with mouth open showing teeth/tongue.

        Uses front-facing canonical pose + refs as identity anchors.

        Args:
            profile: The character whose mouth-open variant to generate.
            model: Gemini model to use for generation.

        Returns:
            Path to the saved mouth-open reference image.
        """
        images_dir = self.config.characters_dir / profile.id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Build reference parts for identity anchoring
        ref_parts = []
        if "front" in profile.canonical_poses:
            ref_parts.append(Image.open(profile.canonical_poses["front"]))
        for ref_path in profile.reference_images[:2]:
            ref_parts.append(Image.open(ref_path))

        is_dark = _is_likely_dark_fur(profile.breed)
        prompt = (
            f"Generate this exact {profile.breed} with mouth wide open, "
            f"showing teeth and tongue, front-facing, photorealistic, "
            f"studio lighting, white background, high detail."
        )
        if is_dark:
            prompt += f" {_DARK_FUR_ADVISORY}"

        contents = [*ref_parts, prompt]
        response = self.client.models.generate_content(
            model=model.value,
            contents=contents,
        )

        out_path = images_dir / "mouth_open.png"
        _save_image_from_response(response, out_path)

        # Store path in canonical poses
        profile.canonical_poses["mouth_open"] = out_path
        profile.save(self.config.characters_dir)

        logger.info("Generated mouth-open ref for %s: %s", profile.id, out_path)
        return out_path

    def detect_breed(self, image_path: Path) -> tuple[str, BreedGroup]:
        """Auto-detect breed and breed group from a pet photo.

        Uses Gemini's vision capability to classify the animal.

        Returns:
            Tuple of (breed_name, breed_group).
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_img = Image.open(image_path)

        prompt = (
            "Analyze this pet photo and respond ONLY with JSON (no markdown):\n"
            '{"breed": "<breed name>", "species": "<dog|cat>", '
            '"age_category": "<puppy|kitten|adult>", '
            '"estimated_adult_weight_lbs": <number>}\n'
            "Be specific about the breed. If mixed, list the dominant breed."
        )

        response = self.client.models.generate_content(
            model=self.config.gemini_model,
            contents=[pil_img, prompt],
        )

        return _parse_breed_response(response.text)

    def list_characters(self) -> list[CharacterProfile]:
        """List all saved character profiles."""
        profiles = []
        for path in self.config.characters_dir.glob("*.json"):
            profiles.append(CharacterProfile.load(path))
        return profiles

    def get_character(self, character_id: str) -> CharacterProfile | None:
        """Load a character by ID."""
        path = self.config.characters_dir / f"{character_id}.json"
        if path.exists():
            return CharacterProfile.load(path)
        # Try matching by name
        for profile in self.list_characters():
            if profile.name.lower() == character_id.lower():
                return profile
        return None

    def generate_multi_character_scene(
        self,
        characters: list[CharacterProfile],
        prompt: str,
        style: str = "photorealistic",
        model: GeminiModel = GeminiModel.NANO_BANANA_FAST,
    ) -> Path:
        """Generate a scene image featuring multiple characters together.

        Args:
            characters: 2-5 CharacterProfile objects to include in the scene.
            prompt: Scene description.
            style: Visual style for the generation.
            model: Gemini model to use.

        Returns:
            Path to the saved scene image.

        Raises:
            ValueError: If fewer than 2 or more than 5 characters are provided.
        """
        if len(characters) > 5:
            raise ValueError(f"Maximum 5 characters per scene, got {len(characters)}.")
        if len(characters) < 2:
            raise ValueError("Multi-character scene requires at least 2 characters.")

        # Save to first character's image directory
        images_dir = self.config.characters_dir / characters[0].id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Collect reference images — prefer front pose for each character
        ref_parts = []
        for char in characters:
            if "front" in char.canonical_poses:
                ref_parts.append(Image.open(char.canonical_poses["front"]))
            elif char.reference_images:
                ref_parts.append(Image.open(char.reference_images[0]))

        # Build character description string
        char_descriptions = " and ".join(
            f"{c.name} the {c.breed}" for c in characters
        )

        full_prompt = (
            f"Generate a scene with {char_descriptions} {prompt}, {style}"
        )

        contents = [*ref_parts, full_prompt]
        response = self.client.models.generate_content(
            model=model.value,
            contents=contents,
        )

        scene_id = uuid.uuid4().hex[:8]
        out_path = images_dir / f"multi_scene_{scene_id}.png"
        _save_image_from_response(response, out_path)
        logger.info(
            "Generated multi-character scene with %d characters: %s",
            len(characters),
            out_path.name,
        )
        return out_path

    def export_character(self, profile: CharacterProfile, output_path: Path) -> Path:
        """Bundle a character into a .zip file with JSON profile and all images.

        Args:
            profile: The character to export.
            output_path: Destination path for the .zip file.

        Returns:
            Path to the created zip file.
        """
        output_path = output_path.with_suffix(".zip")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write profile JSON
            profile_data = {
                "id": profile.id,
                "name": profile.name,
                "breed": profile.breed,
                "breed_group": profile.breed_group.value,
                "reference_images": [p.name for p in profile.reference_images],
                "canonical_poses": {k: v.name for k, v in profile.canonical_poses.items()},
                "voice": {
                    "preset_name": profile.voice.preset_name,
                    "reference_clip": (
                        profile.voice.reference_clip.name
                        if profile.voice.reference_clip
                        else None
                    ),
                    "exaggeration": profile.voice.exaggeration,
                    "pitch_shift": profile.voice.pitch_shift,
                    "speed": profile.voice.speed,
                },
                "personality_tags": profile.personality_tags,
                "metadata": profile.metadata,
            }
            zf.writestr("profile.json", json.dumps(profile_data, indent=2))

            # Bundle all images, deduplicating by archive name
            written: set[str] = set()

            for img_path in profile.reference_images:
                arc_name = f"images/{img_path.name}"
                if img_path.exists() and arc_name not in written:
                    zf.write(img_path, arc_name)
                    written.add(arc_name)

            for pose_name, pose_path in profile.canonical_poses.items():
                arc_name = f"images/{pose_path.name}"
                if pose_path.exists() and arc_name not in written:
                    zf.write(pose_path, arc_name)
                    written.add(arc_name)

            # Bundle voice reference clip if present
            if profile.voice.reference_clip and profile.voice.reference_clip.exists():
                arc_name = f"images/{profile.voice.reference_clip.name}"
                if arc_name not in written:
                    zf.write(profile.voice.reference_clip, arc_name)
                    written.add(arc_name)

        logger.info("Exported character '%s' to %s", profile.name, output_path)
        return output_path

    def import_character(self, zip_path: Path) -> CharacterProfile:
        """Import a character from a .zip bundle.

        Unpacks the zip into a new character directory with a fresh UUID,
        then returns the loaded CharacterProfile.

        Args:
            zip_path: Path to the .zip file to import.

        Returns:
            The imported CharacterProfile.

        Raises:
            FileNotFoundError: If zip_path does not exist.
            ValueError: If the zip is missing the required profile.json.
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if "profile.json" not in names:
                raise ValueError("Invalid character zip: missing profile.json")

            profile_data = json.loads(zf.read("profile.json"))

        # Generate a new UUID for the imported character
        new_id = str(uuid.uuid4())
        char_dir = self.config.characters_dir / new_id
        images_dir = char_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Extract images
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if name.startswith("images/") and not name.endswith("/"):
                    filename = Path(name).name
                    dest = images_dir / filename
                    dest.write_bytes(zf.read(name))

        # Rebuild paths relative to new character directory
        ref_images = [
            images_dir / fname
            for fname in profile_data.get("reference_images", [])
            if (images_dir / fname).exists()
        ]

        canonical_poses = {
            k: images_dir / fname
            for k, fname in profile_data.get("canonical_poses", {}).items()
            if (images_dir / fname).exists()
        }

        voice_data = profile_data.get("voice", {})
        ref_clip_name = voice_data.get("reference_clip")
        ref_clip = None
        if ref_clip_name and (images_dir / ref_clip_name).exists():
            ref_clip = images_dir / ref_clip_name

        from petgen.models import VoiceSettings

        profile = CharacterProfile(
            id=new_id,
            name=profile_data["name"],
            breed=profile_data["breed"],
            breed_group=BreedGroup(profile_data["breed_group"]),
            reference_images=ref_images,
            canonical_poses=canonical_poses,
            voice=VoiceSettings(
                preset_name=voice_data.get("preset_name"),
                reference_clip=ref_clip,
                exaggeration=voice_data.get("exaggeration", 0.7),
                pitch_shift=voice_data.get("pitch_shift", 0.0),
                speed=voice_data.get("speed", 1.0),
            ),
            personality_tags=profile_data.get("personality_tags", []),
            metadata=profile_data.get("metadata", {}),
        )

        profile.save(self.config.characters_dir)
        logger.info("Imported character '%s' as %s", profile.name, new_id)
        return profile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_breed_response(text: str) -> tuple[str, BreedGroup]:
    """Parse the JSON breed-detection response from Gemini.

    Returns:
        Tuple of (breed_name, breed_group).
    """
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    breed = data.get("breed", "Unknown")
    species = data.get("species", "dog").lower()
    age_category = data.get("age_category", "adult").lower()
    weight = data.get("estimated_adult_weight_lbs", 30)

    # Determine breed group
    if species == "cat":
        if age_category == "kitten":
            return breed, BreedGroup.KITTEN
        return breed, BreedGroup.CAT

    # Dog classification
    if age_category == "puppy":
        return breed, BreedGroup.PUPPY

    if weight <= _SMALL_DOG_MAX_LB:
        return breed, BreedGroup.SMALL_DOG
    elif weight <= _MEDIUM_DOG_MAX_LB:
        return breed, BreedGroup.MEDIUM_DOG
    else:
        return breed, BreedGroup.LARGE_DOG


def _save_image_from_response(response, out_path: Path) -> None:
    """Extract the first image from a Gemini response and save it as PNG.

    The Gemini API returns response parts that may include both text and
    inline image data.  We iterate through parts looking for image content.
    """
    for part in response.candidates[0].content.parts:
        if hasattr(part, "inline_data") and part.inline_data is not None:
            image_bytes = part.inline_data.data
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(image_bytes)
            return

    raise RuntimeError(
        "Gemini response did not contain an image. "
        "The model may have returned only text."
    )


def _is_likely_dark_fur(breed: str) -> bool:
    """Heuristic check for breeds commonly having dark/black fur."""
    dark_keywords = [
        "black",
        "rottweiler",
        "doberman",
        "labrador",
        "scottish terrier",
        "schipperke",
        "pug",
        "bombay",
        "panther",
        "persian black",
        "tuxedo",
    ]
    breed_lower = breed.lower()
    return any(kw in breed_lower for kw in dark_keywords)
