"""Background job worker — standalone entry point for running the worker outside the server."""

from __future__ import annotations

import asyncio
import logging

from petgen.config import PetGenConfig
from petgen.pipeline import PetGenPipeline
from petgen.server.queue import JobQueue

logger = logging.getLogger(__name__)


async def run_worker(config: PetGenConfig | None = None) -> None:
    """Run the job queue worker as a standalone process.

    Useful for separating the worker from the web server process.
    """
    if config is None:
        config = PetGenConfig()
        config.ensure_dirs()

    pipeline = PetGenPipeline(config)
    queue = JobQueue(max_concurrent=1)

    logger.info("Worker started, waiting for jobs...")
    await queue.process_jobs(pipeline)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_worker())
