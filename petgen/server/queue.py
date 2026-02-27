"""Async job queue for video generation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from petgen.server.models import JobStatus

logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    status: JobStatus
    request: dict
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    progress: float = 0.0
    stage: str = ""
    result_path: Optional[Path] = None
    error: Optional[str] = None


class JobQueue:
    """Async job queue for video generation.

    Uses asyncio.Queue for submission and processes jobs sequentially
    since the GPU is the bottleneck.
    """

    def __init__(self, max_concurrent: int = 1) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._max_concurrent = max_concurrent
        self._progress_callback = None

    def set_progress_callback(self, callback) -> None:
        """Set an async callback for progress updates: callback(job_id, stage, progress, message)."""
        self._progress_callback = callback

    async def submit(self, request: dict) -> Job:
        """Submit a new generation job to the queue."""
        job_id = uuid.uuid4().hex[:12]
        job = Job(id=job_id, status=JobStatus.PENDING, request=request)
        self._jobs[job_id] = job
        await self._queue.put(job_id)
        logger.info("Job %s submitted", job_id)
        return job

    async def get_status(self, job_id: str) -> Optional[Job]:
        """Get the current status of a job."""
        return self._jobs.get(job_id)

    async def process_jobs(self, pipeline) -> None:
        """Background worker that processes jobs from the queue sequentially."""
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue

            job.status = JobStatus.PROCESSING
            job.stage = "starting"
            logger.info("Processing job %s", job_id)

            try:
                result_path = await self._run_pipeline(pipeline, job)
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
                job.stage = "completed"
                job.result_path = result_path
                logger.info("Job %s completed: %s", job_id, result_path)

                if self._progress_callback:
                    await self._progress_callback(
                        job_id, "completed", 1.0, "Video generation complete"
                    )

            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.stage = "failed"
                logger.error("Job %s failed: %s", job_id, exc)

                if self._progress_callback:
                    await self._progress_callback(
                        job_id, "failed", job.progress, str(exc)
                    )

            finally:
                self._queue.task_done()

    async def _run_pipeline(self, pipeline, job: Job) -> Path:
        """Run the generation pipeline for a job in a thread executor."""
        from petgen.models import AnimationConfig, AspectRatio

        req = job.request
        character_id = req["character_id"]

        # Update progress: loading character
        job.stage = "character"
        job.progress = 0.05
        if self._progress_callback:
            await self._progress_callback(
                job.id, "character", 0.05, "Loading character..."
            )

        profile = pipeline.character_creator.get_character(character_id)
        if profile is None:
            raise ValueError("Character '{}' not found".format(character_id))

        profile.voice.exaggeration = req.get("emotion", 0.7)
        profile.voice.speed = req.get("speed", 1.0)

        aspect_map = {
            "9:16": AspectRatio.VERTICAL,
            "1:1": AspectRatio.SQUARE,
            "16:9": AspectRatio.LANDSCAPE,
        }
        res_map = {
            AspectRatio.VERTICAL: (1080, 1920),
            AspectRatio.SQUARE: (1080, 1080),
            AspectRatio.LANDSCAPE: (1920, 1080),
        }
        aspect_ratio = aspect_map.get(req.get("aspect_ratio", "9:16"), AspectRatio.VERTICAL)
        animation_config = AnimationConfig(
            resolution=res_map[aspect_ratio],
            aspect_ratio=aspect_ratio,
        )

        # Update progress: generating
        job.stage = "generating"
        job.progress = 0.1
        if self._progress_callback:
            await self._progress_callback(
                job.id, "generating", 0.1, "Starting generation pipeline..."
            )

        # Run the blocking pipeline in a thread
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: pipeline.generate(
                character=profile,
                script=req["script"],
                scene_prompt=req.get("scene_prompt"),
                animation_config=animation_config,
            ),
        )

        return result.video_path
