"""Character CRUD endpoints."""

import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from petgen.pipeline import PetGenPipeline
from petgen.server.deps import get_pipeline
from petgen.server.models import CharacterResponse

router = APIRouter(prefix="/api/characters", tags=["characters"])


def _profile_to_response(profile) -> CharacterResponse:
    return CharacterResponse(
        id=profile.id,
        name=profile.name,
        breed=profile.breed,
        breed_group=profile.breed_group.value,
        num_refs=len(profile.reference_images),
        poses=list(profile.canonical_poses.keys()),
    )


@router.post("", response_model=CharacterResponse, status_code=201)
async def create_character(
    name: str = Form(...),
    breed: Optional[str] = Form(default=None),
    photos: List[UploadFile] = File(...),
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> CharacterResponse:
    """Create a new character from uploaded photos."""
    if not photos:
        raise HTTPException(status_code=400, detail="At least one photo is required")

    # Save uploaded files to temp directory
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        saved_paths: List[Path] = []
        for i, photo in enumerate(photos):
            suffix = Path(photo.filename or "photo_{}.jpg".format(i)).suffix or ".jpg"
            dest = tmp_dir / "upload_{}{}".format(i, suffix)
            content = await photo.read()
            dest.write_bytes(content)
            saved_paths.append(dest)

        profile = pipeline.create_character(
            name=name,
            photos=saved_paths,
            breed=breed,
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return _profile_to_response(profile)


@router.get("", response_model=List[CharacterResponse])
async def list_characters(
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> List[CharacterResponse]:
    """List all saved characters."""
    profiles = pipeline.character_creator.list_characters()
    return [_profile_to_response(p) for p in profiles]


@router.get("/{character_id}", response_model=CharacterResponse)
async def get_character(
    character_id: str,
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> CharacterResponse:
    """Get a character by ID."""
    profile = pipeline.character_creator.get_character(character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return _profile_to_response(profile)


@router.delete("/{character_id}", status_code=204)
async def delete_character(
    character_id: str,
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> None:
    """Delete a character and its files."""
    profile = pipeline.character_creator.get_character(character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    # Remove character directory and profile JSON
    char_dir = pipeline.config.characters_dir / profile.id
    if char_dir.is_dir():
        shutil.rmtree(char_dir)
    profile_json = pipeline.config.characters_dir / "{}.json".format(profile.id)
    if profile_json.is_file():
        profile_json.unlink()


@router.get("/{character_id}/export")
async def export_character(
    character_id: str,
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> FileResponse:
    """Export a character as a zip file."""
    profile = pipeline.character_creator.get_character(character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    export_path = pipeline.config.temp_dir / "{}_{}.zip".format(profile.name, profile.id[:8])
    pipeline.character_creator.export_character(profile, export_path)

    return FileResponse(
        path=str(export_path),
        filename="{}.zip".format(profile.name),
        media_type="application/zip",
    )


@router.post("/import", response_model=CharacterResponse, status_code=201)
async def import_character(
    file: UploadFile = File(...),
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> CharacterResponse:
    """Import a character from a zip file."""
    tmp_path = Path(tempfile.mktemp(suffix=".zip"))
    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        profile = pipeline.character_creator.import_character(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return _profile_to_response(profile)
