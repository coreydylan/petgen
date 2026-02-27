"""Tests for the character creation module."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from petgen.config import PetGenConfig
from petgen.models import BreedGroup, CharacterProfile, VoiceSettings
from petgen.models import MultiCharacterScene
from petgen.modules.character import (
    CharacterCreator,
    _is_likely_dark_fur,
    _parse_breed_response,
    _save_image_from_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def creator(tmp_config: PetGenConfig) -> CharacterCreator:
    """CharacterCreator with a mocked Gemini client."""
    c = CharacterCreator(tmp_config)
    c._client = MagicMock()
    return c


@pytest.fixture
def ref_image(tmp_path: Path) -> Path:
    """Create a small test reference image on disk."""
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    path = tmp_path / "ref_dog.png"
    img.save(path)
    return path


@pytest.fixture
def ref_images(tmp_path: Path) -> list[Path]:
    """Create multiple test reference images."""
    paths = []
    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        p = tmp_path / f"ref_{i}.png"
        img.save(p)
        paths.append(p)
    return paths


def _make_png_bytes() -> bytes:
    """Create minimal valid PNG bytes."""
    img = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_gemini_image_response(image_bytes: bytes | None = None) -> SimpleNamespace:
    """Build a mock Gemini response containing an inline image."""
    if image_bytes is None:
        image_bytes = _make_png_bytes()

    inline_data = SimpleNamespace(data=image_bytes)
    image_part = SimpleNamespace(inline_data=inline_data)
    content = SimpleNamespace(parts=[image_part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def _make_gemini_text_response(text: str) -> SimpleNamespace:
    """Build a mock Gemini response containing only text (no image)."""
    text_part = SimpleNamespace(inline_data=None)
    content = SimpleNamespace(parts=[text_part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate], text=text)


def _make_breed_detect_response(
    breed: str = "Golden Retriever",
    species: str = "dog",
    age_category: str = "adult",
    weight: int = 65,
) -> SimpleNamespace:
    """Build a mock Gemini response for breed detection (text only)."""
    payload = json.dumps(
        {
            "breed": breed,
            "species": species,
            "age_category": age_category,
            "estimated_adult_weight_lbs": weight,
        }
    )
    return SimpleNamespace(text=payload)


# ---------------------------------------------------------------------------
# Tests: _parse_breed_response
# ---------------------------------------------------------------------------


class TestParseBreedResponse:
    def test_large_dog(self):
        text = json.dumps(
            {
                "breed": "Golden Retriever",
                "species": "dog",
                "age_category": "adult",
                "estimated_adult_weight_lbs": 65,
            }
        )
        breed, group = _parse_breed_response(text)
        assert breed == "Golden Retriever"
        assert group == BreedGroup.LARGE_DOG

    def test_medium_dog(self):
        text = json.dumps(
            {
                "breed": "Border Collie",
                "species": "dog",
                "age_category": "adult",
                "estimated_adult_weight_lbs": 40,
            }
        )
        breed, group = _parse_breed_response(text)
        assert breed == "Border Collie"
        assert group == BreedGroup.MEDIUM_DOG

    def test_small_dog(self):
        text = json.dumps(
            {
                "breed": "Chihuahua",
                "species": "dog",
                "age_category": "adult",
                "estimated_adult_weight_lbs": 5,
            }
        )
        breed, group = _parse_breed_response(text)
        assert breed == "Chihuahua"
        assert group == BreedGroup.SMALL_DOG

    def test_puppy(self):
        text = json.dumps(
            {
                "breed": "Labrador Retriever",
                "species": "dog",
                "age_category": "puppy",
                "estimated_adult_weight_lbs": 70,
            }
        )
        breed, group = _parse_breed_response(text)
        assert breed == "Labrador Retriever"
        assert group == BreedGroup.PUPPY

    def test_cat(self):
        text = json.dumps(
            {
                "breed": "Siamese",
                "species": "cat",
                "age_category": "adult",
                "estimated_adult_weight_lbs": 10,
            }
        )
        breed, group = _parse_breed_response(text)
        assert breed == "Siamese"
        assert group == BreedGroup.CAT

    def test_kitten(self):
        text = json.dumps(
            {
                "breed": "Tabby",
                "species": "cat",
                "age_category": "kitten",
                "estimated_adult_weight_lbs": 9,
            }
        )
        breed, group = _parse_breed_response(text)
        assert breed == "Tabby"
        assert group == BreedGroup.KITTEN

    def test_markdown_fenced_json(self):
        text = (
            '```json\n'
            '{"breed": "Poodle", "species": "dog", '
            '"age_category": "adult", "estimated_adult_weight_lbs": 50}\n'
            '```'
        )
        breed, group = _parse_breed_response(text)
        assert breed == "Poodle"
        assert group == BreedGroup.MEDIUM_DOG

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_breed_response("not json at all")

    def test_missing_fields_defaults(self):
        text = json.dumps({"breed": "Unknown Mix"})
        breed, group = _parse_breed_response(text)
        assert breed == "Unknown Mix"
        # default weight=30, species=dog, age=adult -> medium dog
        assert group == BreedGroup.MEDIUM_DOG

    def test_boundary_small_to_medium(self):
        text = json.dumps(
            {"breed": "Cocker Spaniel", "species": "dog",
             "age_category": "adult", "estimated_adult_weight_lbs": 25}
        )
        _, group = _parse_breed_response(text)
        assert group == BreedGroup.SMALL_DOG  # <= 25 is small

    def test_boundary_medium_to_large(self):
        text = json.dumps(
            {"breed": "Boxer", "species": "dog",
             "age_category": "adult", "estimated_adult_weight_lbs": 55}
        )
        _, group = _parse_breed_response(text)
        assert group == BreedGroup.MEDIUM_DOG  # <= 55 is medium


# ---------------------------------------------------------------------------
# Tests: _save_image_from_response
# ---------------------------------------------------------------------------


class TestSaveImageFromResponse:
    def test_saves_image(self, tmp_path: Path):
        response = _make_gemini_image_response()
        out = tmp_path / "test.png"
        _save_image_from_response(response, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_raises_on_text_only_response(self, tmp_path: Path):
        response = _make_gemini_text_response("No image here")
        with pytest.raises(RuntimeError, match="did not contain an image"):
            _save_image_from_response(response, tmp_path / "nope.png")

    def test_creates_parent_dirs(self, tmp_path: Path):
        response = _make_gemini_image_response()
        out = tmp_path / "deep" / "nested" / "dir" / "img.png"
        _save_image_from_response(response, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Tests: _is_likely_dark_fur
# ---------------------------------------------------------------------------


class TestIsDarkFur:
    @pytest.mark.parametrize(
        "breed",
        ["Black Labrador", "Rottweiler", "Doberman Pinscher", "Bombay", "Schipperke"],
    )
    def test_dark_breeds(self, breed: str):
        assert _is_likely_dark_fur(breed) is True

    @pytest.mark.parametrize(
        "breed",
        ["Golden Retriever", "Samoyed", "West Highland White Terrier", "Siamese"],
    )
    def test_light_breeds(self, breed: str):
        assert _is_likely_dark_fur(breed) is False


# ---------------------------------------------------------------------------
# Tests: CharacterProfile round-trip serialization
# ---------------------------------------------------------------------------


class TestProfileSerialization:
    def test_save_and_load_roundtrip(self, tmp_config: PetGenConfig):
        profile = CharacterProfile(
            id="rt-001",
            name="Buddy",
            breed="Beagle",
            breed_group=BreedGroup.MEDIUM_DOG,
            reference_images=[Path("/fake/ref1.png"), Path("/fake/ref2.png")],
            canonical_poses={"front": Path("/fake/front.png")},
            voice=VoiceSettings(preset_name="medium_dog", exaggeration=0.6, speed=1.1),
            personality_tags=["curious", "energetic"],
            metadata={"source": "test"},
        )

        saved_path = profile.save(tmp_config.characters_dir)
        assert saved_path.exists()

        loaded = CharacterProfile.load(saved_path)
        assert loaded.id == profile.id
        assert loaded.name == profile.name
        assert loaded.breed == profile.breed
        assert loaded.breed_group == profile.breed_group
        assert len(loaded.reference_images) == 2
        assert loaded.canonical_poses["front"] == Path("/fake/front.png")
        assert loaded.voice.preset_name == "medium_dog"
        assert loaded.voice.exaggeration == 0.6
        assert loaded.personality_tags == ["curious", "energetic"]
        assert loaded.metadata == {"source": "test"}

    def test_save_creates_directory(self, tmp_path: Path):
        profile = CharacterProfile(
            id="dir-test",
            name="Test",
            breed="Poodle",
            breed_group=BreedGroup.MEDIUM_DOG,
            reference_images=[],
        )
        new_dir = tmp_path / "new" / "chars"
        saved = profile.save(new_dir)
        assert saved.exists()
        assert json.loads(saved.read_text())["id"] == "dir-test"

    def test_default_voice_roundtrip(self, tmp_path: Path):
        profile = CharacterProfile(
            id="voice-default",
            name="Defvo",
            breed="Mutt",
            breed_group=BreedGroup.MEDIUM_DOG,
            reference_images=[],
        )
        saved = profile.save(tmp_path)
        loaded = CharacterProfile.load(saved)
        assert loaded.voice.preset_name is None
        assert loaded.voice.exaggeration == 0.7
        assert loaded.voice.speed == 1.0


# ---------------------------------------------------------------------------
# Tests: CharacterCreator.create_character
# ---------------------------------------------------------------------------


class TestCreateCharacter:
    def test_create_with_explicit_breed(self, creator: CharacterCreator, ref_images: list[Path]):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        profile = creator.create_character(
            reference_images=ref_images,
            name="Rex",
            breed="German Shepherd",
            breed_group=BreedGroup.LARGE_DOG,
        )

        assert profile.name == "Rex"
        assert profile.breed == "German Shepherd"
        assert profile.breed_group == BreedGroup.LARGE_DOG
        assert len(profile.reference_images) == 3
        assert set(profile.canonical_poses.keys()) == {"front", "side", "three_quarter"}
        for pose_path in profile.canonical_poses.values():
            assert pose_path.exists()
        # Profile JSON persisted
        json_path = creator.config.characters_dir / f"{profile.id}.json"
        assert json_path.exists()
        # 3 canonical pose generations
        assert creator._client.models.generate_content.call_count == 3

    def test_create_auto_detect_breed(self, creator: CharacterCreator, ref_image: Path):
        breed_resp = _make_breed_detect_response(
            breed="Pomeranian", species="dog", age_category="adult", weight=7
        )
        image_resp = _make_gemini_image_response()
        creator._client.models.generate_content.side_effect = [
            breed_resp, image_resp, image_resp, image_resp
        ]

        profile = creator.create_character(
            reference_images=[ref_image],
            name="Fluffy",
        )

        assert profile.breed == "Pomeranian"
        assert profile.breed_group == BreedGroup.SMALL_DOG
        # 1 breed detect + 3 canonical poses
        assert creator._client.models.generate_content.call_count == 4

    def test_create_with_personality_tags(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        profile = creator.create_character(
            reference_images=[ref_image],
            name="Happy",
            breed="Corgi",
            breed_group=BreedGroup.SMALL_DOG,
            personality_tags=["playful", "sassy"],
        )

        assert profile.personality_tags == ["playful", "sassy"]

    def test_create_no_images_raises(self, creator: CharacterCreator):
        with pytest.raises(ValueError, match="At least one reference image"):
            creator.create_character(reference_images=[], name="Nobody")

    def test_create_missing_image_raises(self, creator: CharacterCreator, tmp_path: Path):
        fake = tmp_path / "nonexistent.png"
        with pytest.raises(FileNotFoundError, match="Reference image not found"):
            creator.create_character(
                reference_images=[fake],
                name="Ghost",
                breed="Unknown",
                breed_group=BreedGroup.MEDIUM_DOG,
            )

    def test_create_copies_refs_into_char_dir(self, creator: CharacterCreator, ref_images: list[Path]):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        profile = creator.create_character(
            reference_images=ref_images,
            name="Copy",
            breed="Beagle",
            breed_group=BreedGroup.MEDIUM_DOG,
        )

        # Reference images should live inside the character's images dir
        char_images_dir = creator.config.characters_dir / profile.id / "images"
        for ref in profile.reference_images:
            assert ref.parent == char_images_dir
            assert ref.exists()


# ---------------------------------------------------------------------------
# Tests: CharacterCreator.generate_canonical_poses
# ---------------------------------------------------------------------------


class TestGenerateCanonicalPoses:
    def test_generates_three_poses(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        char_id = "pose-test-001"
        images_dir = creator.config.characters_dir / char_id / "images"
        images_dir.mkdir(parents=True)

        profile = CharacterProfile(
            id=char_id,
            name="Posie",
            breed="Corgi",
            breed_group=BreedGroup.SMALL_DOG,
            reference_images=[ref_image],
        )

        poses = creator.generate_canonical_poses(profile)

        assert set(poses.keys()) == {"front", "side", "three_quarter"}
        for pose_path in poses.values():
            assert pose_path.exists()
            assert pose_path.suffix == ".png"
        assert creator._client.models.generate_content.call_count == 3

    def test_dark_fur_advisory_in_prompt(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        char_id = "dark-test"
        images_dir = creator.config.characters_dir / char_id / "images"
        images_dir.mkdir(parents=True)

        profile = CharacterProfile(
            id=char_id,
            name="Shadow",
            breed="Black Labrador",
            breed_group=BreedGroup.LARGE_DOG,
            reference_images=[ref_image],
        )

        creator.generate_canonical_poses(profile)

        # Verify all prompt strings include dark fur advisory
        for call in creator._client.models.generate_content.call_args_list:
            contents = call[1]["contents"]
            prompt_text = contents[-1]  # last item is the text prompt
            assert "dark fur" in prompt_text.lower()

    def test_uses_specified_model(self, creator: CharacterCreator, ref_image: Path):
        from petgen.models import GeminiModel

        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        char_id = "model-test"
        (creator.config.characters_dir / char_id / "images").mkdir(parents=True)

        profile = CharacterProfile(
            id=char_id, name="M", breed="Pug", breed_group=BreedGroup.SMALL_DOG,
            reference_images=[ref_image],
        )

        creator.generate_canonical_poses(profile, model=GeminiModel.NANO_BANANA_PRO)

        for call in creator._client.models.generate_content.call_args_list:
            assert call[1]["model"] == "gemini-3-pro-image-preview"


# ---------------------------------------------------------------------------
# Tests: CharacterCreator.generate_scene
# ---------------------------------------------------------------------------


class TestGenerateScene:
    def test_generates_scene_image(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        char_id = "scene-test"
        (creator.config.characters_dir / char_id / "images").mkdir(parents=True)

        profile = CharacterProfile(
            id=char_id,
            name="Captain",
            breed="Golden Retriever",
            breed_group=BreedGroup.LARGE_DOG,
            reference_images=[ref_image],
            canonical_poses={"front": ref_image},
        )

        scene_path = creator.generate_scene(
            profile, prompt="wearing a pirate hat on a sailing ship"
        )

        assert scene_path.exists()
        assert "scene_" in scene_path.name
        assert scene_path.suffix == ".png"
        creator._client.models.generate_content.assert_called_once()

    def test_scene_includes_style(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        char_id = "style-test"
        (creator.config.characters_dir / char_id / "images").mkdir(parents=True)

        profile = CharacterProfile(
            id=char_id, name="Arty", breed="Poodle",
            breed_group=BreedGroup.MEDIUM_DOG,
            reference_images=[ref_image],
        )

        creator.generate_scene(profile, prompt="in a park", style="watercolor painting")

        call_contents = creator._client.models.generate_content.call_args[1]["contents"]
        prompt_text = call_contents[-1]
        assert "watercolor painting" in prompt_text

    def test_scene_api_failure_propagates(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.side_effect = RuntimeError("API quota exceeded")

        char_id = "fail-test"
        (creator.config.characters_dir / char_id / "images").mkdir(parents=True)

        profile = CharacterProfile(
            id=char_id, name="Fail", breed="Poodle",
            breed_group=BreedGroup.MEDIUM_DOG,
            reference_images=[ref_image],
        )

        with pytest.raises(RuntimeError, match="API quota exceeded"):
            creator.generate_scene(profile, prompt="in a park")


# ---------------------------------------------------------------------------
# Tests: CharacterCreator.detect_breed
# ---------------------------------------------------------------------------


class TestDetectBreed:
    def test_detect_breed_dog(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_breed_detect_response(
            breed="Shiba Inu", species="dog", age_category="adult", weight=22
        )

        breed, group = creator.detect_breed(ref_image)

        assert breed == "Shiba Inu"
        assert group == BreedGroup.SMALL_DOG
        creator._client.models.generate_content.assert_called_once()

    def test_detect_breed_cat(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_breed_detect_response(
            breed="Maine Coon", species="cat", age_category="adult", weight=18
        )

        breed, group = creator.detect_breed(ref_image)

        assert breed == "Maine Coon"
        assert group == BreedGroup.CAT

    def test_detect_breed_puppy(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_breed_detect_response(
            breed="Labrador Retriever", species="dog", age_category="puppy", weight=70
        )

        breed, group = creator.detect_breed(ref_image)

        assert breed == "Labrador Retriever"
        assert group == BreedGroup.PUPPY

    def test_detect_breed_missing_image(self, creator: CharacterCreator, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Image not found"):
            creator.detect_breed(tmp_path / "nope.jpg")


# ---------------------------------------------------------------------------
# Tests: list_characters / get_character
# ---------------------------------------------------------------------------


class TestListAndGetCharacters:
    def test_list_empty(self, creator: CharacterCreator):
        assert creator.list_characters() == []

    def test_list_and_get(self, creator: CharacterCreator, tmp_config: PetGenConfig):
        p1 = CharacterProfile(
            id="list-a", name="Alpha", breed="Poodle",
            breed_group=BreedGroup.MEDIUM_DOG, reference_images=[],
        )
        p2 = CharacterProfile(
            id="list-b", name="Beta", breed="Siamese",
            breed_group=BreedGroup.CAT, reference_images=[],
        )
        p1.save(tmp_config.characters_dir)
        p2.save(tmp_config.characters_dir)

        chars = creator.list_characters()
        assert len(chars) == 2
        assert {c.id for c in chars} == {"list-a", "list-b"}

    def test_get_by_id(self, creator: CharacterCreator, tmp_config: PetGenConfig):
        p = CharacterProfile(
            id="get-test", name="Gamma", breed="Corgi",
            breed_group=BreedGroup.SMALL_DOG, reference_images=[],
        )
        p.save(tmp_config.characters_dir)

        loaded = creator.get_character("get-test")
        assert loaded is not None
        assert loaded.name == "Gamma"

    def test_get_by_name(self, creator: CharacterCreator, tmp_config: PetGenConfig):
        p = CharacterProfile(
            id="name-lookup", name="Delta", breed="Husky",
            breed_group=BreedGroup.LARGE_DOG, reference_images=[],
        )
        p.save(tmp_config.characters_dir)

        loaded = creator.get_character("delta")  # case-insensitive
        assert loaded is not None
        assert loaded.id == "name-lookup"

    def test_get_nonexistent(self, creator: CharacterCreator):
        assert creator.get_character("no-such-id") is None


# ---------------------------------------------------------------------------
# Tests: CharacterCreator.generate_multi_character_scene
# ---------------------------------------------------------------------------


class TestMultiCharacterScene:
    def test_generates_scene_with_two_characters(
        self, creator: CharacterCreator, ref_image: Path
    ):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        char_ids = ["multi-a", "multi-b"]
        profiles = []
        for cid in char_ids:
            (creator.config.characters_dir / cid / "images").mkdir(parents=True)
            profiles.append(
                CharacterProfile(
                    id=cid,
                    name=f"Pet_{cid}",
                    breed="Corgi",
                    breed_group=BreedGroup.SMALL_DOG,
                    reference_images=[ref_image],
                    canonical_poses={"front": ref_image},
                )
            )

        scene_path = creator.generate_multi_character_scene(
            characters=profiles, prompt="playing in a park"
        )

        assert scene_path.exists()
        assert "multi_scene_" in scene_path.name
        assert scene_path.suffix == ".png"
        creator._client.models.generate_content.assert_called_once()

    def test_prompt_contains_all_character_names(
        self, creator: CharacterCreator, ref_image: Path
    ):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        profiles = []
        for i, name in enumerate(["Luna", "Max", "Bella"]):
            cid = f"name-{i}"
            (creator.config.characters_dir / cid / "images").mkdir(parents=True)
            profiles.append(
                CharacterProfile(
                    id=cid,
                    name=name,
                    breed="Poodle",
                    breed_group=BreedGroup.MEDIUM_DOG,
                    reference_images=[ref_image],
                )
            )

        creator.generate_multi_character_scene(
            characters=profiles, prompt="at the beach"
        )

        call_contents = creator._client.models.generate_content.call_args[1]["contents"]
        prompt_text = call_contents[-1]
        assert "Luna" in prompt_text
        assert "Max" in prompt_text
        assert "Bella" in prompt_text

    def test_raises_on_more_than_five(self, creator: CharacterCreator, ref_image: Path):
        profiles = [
            CharacterProfile(
                id=f"too-many-{i}",
                name=f"Pet{i}",
                breed="Mutt",
                breed_group=BreedGroup.MEDIUM_DOG,
                reference_images=[ref_image],
            )
            for i in range(6)
        ]

        with pytest.raises(ValueError, match="Maximum 5 characters"):
            creator.generate_multi_character_scene(
                characters=profiles, prompt="crowded park"
            )

    def test_raises_on_fewer_than_two(self, creator: CharacterCreator, ref_image: Path):
        profile = CharacterProfile(
            id="solo",
            name="Lonely",
            breed="Pug",
            breed_group=BreedGroup.SMALL_DOG,
            reference_images=[ref_image],
        )

        with pytest.raises(ValueError, match="at least 2 characters"):
            creator.generate_multi_character_scene(
                characters=[profile], prompt="alone"
            )

    def test_scene_saved_to_first_characters_dir(
        self, creator: CharacterCreator, ref_image: Path
    ):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        first_id = "first-char"
        second_id = "second-char"
        for cid in [first_id, second_id]:
            (creator.config.characters_dir / cid / "images").mkdir(parents=True)

        profiles = [
            CharacterProfile(
                id=first_id, name="First", breed="Corgi",
                breed_group=BreedGroup.SMALL_DOG,
                reference_images=[ref_image],
            ),
            CharacterProfile(
                id=second_id, name="Second", breed="Poodle",
                breed_group=BreedGroup.MEDIUM_DOG,
                reference_images=[ref_image],
            ),
        ]

        scene_path = creator.generate_multi_character_scene(
            characters=profiles, prompt="together"
        )

        expected_dir = creator.config.characters_dir / first_id / "images"
        assert scene_path.parent == expected_dir

    def test_style_in_prompt(self, creator: CharacterCreator, ref_image: Path):
        creator._client.models.generate_content.return_value = _make_gemini_image_response()

        profiles = []
        for i in range(2):
            cid = f"style-{i}"
            (creator.config.characters_dir / cid / "images").mkdir(parents=True)
            profiles.append(
                CharacterProfile(
                    id=cid, name=f"P{i}", breed="Corgi",
                    breed_group=BreedGroup.SMALL_DOG,
                    reference_images=[ref_image],
                )
            )

        creator.generate_multi_character_scene(
            characters=profiles, prompt="in space", style="watercolor"
        )

        call_contents = creator._client.models.generate_content.call_args[1]["contents"]
        prompt_text = call_contents[-1]
        assert "watercolor" in prompt_text


# ---------------------------------------------------------------------------
# Tests: CharacterCreator.export_character / import_character
# ---------------------------------------------------------------------------


class TestExportImport:
    def test_export_creates_zip(self, creator: CharacterCreator, ref_image: Path, tmp_path: Path):
        char_id = "export-test"
        images_dir = creator.config.characters_dir / char_id / "images"
        images_dir.mkdir(parents=True)

        # Copy ref image into character dir to simulate a real character
        dest_ref = images_dir / "ref_0.png"
        import shutil
        shutil.copy2(str(ref_image), str(dest_ref))

        profile = CharacterProfile(
            id=char_id,
            name="Exporty",
            breed="Corgi",
            breed_group=BreedGroup.SMALL_DOG,
            reference_images=[dest_ref],
            canonical_poses={"front": dest_ref},
            personality_tags=["brave"],
        )

        zip_path = creator.export_character(profile, tmp_path / "exporty.zip")

        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "profile.json" in names
            assert any(n.startswith("images/") for n in names)

    def test_import_creates_new_character(
        self, creator: CharacterCreator, ref_image: Path, tmp_path: Path
    ):
        # First export a character
        char_id = "roundtrip-export"
        images_dir = creator.config.characters_dir / char_id / "images"
        images_dir.mkdir(parents=True)

        import shutil
        dest_ref = images_dir / "ref_0.png"
        shutil.copy2(str(ref_image), str(dest_ref))

        original = CharacterProfile(
            id=char_id,
            name="Roundtrip",
            breed="Beagle",
            breed_group=BreedGroup.MEDIUM_DOG,
            reference_images=[dest_ref],
            canonical_poses={"front": dest_ref},
            personality_tags=["curious"],
            metadata={"origin": "test"},
        )

        zip_path = creator.export_character(original, tmp_path / "roundtrip.zip")

        # Import the exported character
        imported = creator.import_character(zip_path)

        assert imported.id != original.id  # new UUID
        assert imported.name == "Roundtrip"
        assert imported.breed == "Beagle"
        assert imported.breed_group == BreedGroup.MEDIUM_DOG
        assert imported.personality_tags == ["curious"]
        assert imported.metadata == {"origin": "test"}
        assert len(imported.reference_images) == 1
        assert imported.reference_images[0].exists()
        assert "front" in imported.canonical_poses
        assert imported.canonical_poses["front"].exists()

        # Profile should be persisted
        json_path = creator.config.characters_dir / f"{imported.id}.json"
        assert json_path.exists()

    def test_import_missing_zip_raises(self, creator: CharacterCreator, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Zip file not found"):
            creator.import_character(tmp_path / "nonexistent.zip")

    def test_import_invalid_zip_raises(self, creator: CharacterCreator, tmp_path: Path):
        import zipfile

        bad_zip = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("readme.txt", "no profile here")

        with pytest.raises(ValueError, match="missing profile.json"):
            creator.import_character(bad_zip)

    def test_export_import_preserves_voice_settings(
        self, creator: CharacterCreator, ref_image: Path, tmp_path: Path
    ):
        char_id = "voice-export"
        images_dir = creator.config.characters_dir / char_id / "images"
        images_dir.mkdir(parents=True)

        import shutil
        dest_ref = images_dir / "ref_0.png"
        shutil.copy2(str(ref_image), str(dest_ref))

        original = CharacterProfile(
            id=char_id,
            name="Voicy",
            breed="Husky",
            breed_group=BreedGroup.LARGE_DOG,
            reference_images=[dest_ref],
            voice=VoiceSettings(
                preset_name="deep",
                exaggeration=0.3,
                pitch_shift=-1.5,
                speed=0.85,
            ),
        )

        zip_path = creator.export_character(original, tmp_path / "voicy.zip")
        imported = creator.import_character(zip_path)

        assert imported.voice.preset_name == "deep"
        assert imported.voice.exaggeration == 0.3
        assert imported.voice.pitch_shift == -1.5
        assert imported.voice.speed == 0.85
