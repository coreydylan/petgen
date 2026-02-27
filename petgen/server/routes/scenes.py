"""Scene generation endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from petgen.pipeline import PetGenPipeline
from petgen.server.deps import get_pipeline
from petgen.server.models import SceneRequest

router = APIRouter(prefix="/api/scenes", tags=["scenes"])


@router.post("")
async def generate_scene(
    request: SceneRequest,
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> FileResponse:
    """Generate a scene image with a character."""
    profile = pipeline.character_creator.get_character(request.character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    scene_path = pipeline.character_creator.generate_scene(
        profile=profile,
        prompt=request.prompt,
        style=request.style,
    )

    return FileResponse(
        path=str(scene_path),
        filename=scene_path.name,
        media_type="image/png",
    )
