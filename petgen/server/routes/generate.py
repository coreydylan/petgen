"""Video generation endpoints with async job queue."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from petgen.server.deps import get_progress, get_queue
from petgen.server.models import GenerateJobResponse, GenerateRequest, JobStatus, JobStatusResponse
from petgen.server.queue import JobQueue
from petgen.server.ws import ProgressManager

router = APIRouter(tags=["generate"])


@router.post("/api/generate", response_model=GenerateJobResponse, status_code=202)
async def submit_generation(
    request: GenerateRequest,
    queue: JobQueue = Depends(get_queue),
) -> GenerateJobResponse:
    """Submit a video generation job. Returns a job ID for polling or WebSocket streaming."""
    job = await queue.submit(request.model_dump())
    return GenerateJobResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at,
    )


@router.get("/api/generate/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    queue: JobQueue = Depends(get_queue),
) -> JobStatusResponse:
    """Get the current status of a generation job."""
    job = await queue.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result_url = None
    if job.status == JobStatus.COMPLETED and job.result_path is not None:
        result_url = "/files/{}".format(job.result_path.name)

    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        stage=job.stage,
        result_url=result_url,
        error=job.error,
    )


@router.get("/api/generate/{job_id}/download")
async def download_result(
    job_id: str,
    queue: JobQueue = Depends(get_queue),
) -> FileResponse:
    """Download the completed video for a generation job."""
    job = await queue.get_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job is {}, not completed".format(job.status.value))
    if job.result_path is None or not job.result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=str(job.result_path),
        filename=job.result_path.name,
        media_type="video/mp4",
    )


@router.websocket("/ws/generate/{job_id}")
async def ws_job_progress(
    job_id: str,
    ws: WebSocket,
    queue: JobQueue = Depends(get_queue),
    progress: ProgressManager = Depends(get_progress),
) -> None:
    """WebSocket endpoint for streaming job progress updates."""
    job = await queue.get_status(job_id)
    if job is None:
        await ws.close(code=4004, reason="Job not found")
        return

    await progress.connect(job_id, ws)
    try:
        # Keep the connection open until the client disconnects
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await progress.disconnect(job_id, ws)
