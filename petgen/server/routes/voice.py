"""Voice generation endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from petgen.pipeline import PetGenPipeline
from petgen.server.deps import get_pipeline
from petgen.server.models import VoiceRequest

router = APIRouter(prefix="/api/voice", tags=["voice"])


@router.post("")
async def generate_voice(
    request: VoiceRequest,
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> FileResponse:
    """Generate voice audio for a character speaking a script."""
    profile = pipeline.character_creator.get_character(request.character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    profile.voice.exaggeration = request.emotion

    audio_path = pipeline.voice_generator.generate(
        text=request.script,
        voice_ref=profile.voice.reference_clip,
        exaggeration=profile.voice.exaggeration,
        speed=profile.voice.speed,
    )

    return FileResponse(
        path=str(audio_path),
        filename=audio_path.name,
        media_type="audio/wav",
    )


@router.post("/preview")
async def preview_voice(
    request: VoiceRequest,
    pipeline: PetGenPipeline = Depends(get_pipeline),
) -> FileResponse:
    """Generate a quick voice preview (same pipeline, lower exaggeration for speed)."""
    profile = pipeline.character_creator.get_character(request.character_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Character not found")

    # Preview uses lower exaggeration for faster generation
    audio_path = pipeline.voice_generator.generate(
        text=request.script,
        voice_ref=profile.voice.reference_clip,
        exaggeration=max(0.3, request.emotion * 0.5),
        speed=profile.voice.speed * 1.2,
    )

    return FileResponse(
        path=str(audio_path),
        filename=audio_path.name,
        media_type="audio/wav",
    )
