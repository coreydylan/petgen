"""Dependency injection for the FastAPI server."""

from typing import Optional

from petgen.config import PetGenConfig
from petgen.pipeline import PetGenPipeline
from petgen.server.queue import JobQueue
from petgen.server.ws import ProgressManager

_config: Optional[PetGenConfig] = None
_pipeline: Optional[PetGenPipeline] = None
_queue: Optional[JobQueue] = None
_progress: Optional[ProgressManager] = None


def init_deps() -> None:
    """Initialize singleton dependencies. Called during app lifespan startup."""
    global _config, _pipeline, _queue, _progress
    _config = PetGenConfig()
    _config.ensure_dirs()
    _pipeline = PetGenPipeline(_config)
    _queue = JobQueue(max_concurrent=1)
    _progress = ProgressManager()
    _queue.set_progress_callback(_progress.emit_progress)


def get_config() -> PetGenConfig:
    assert _config is not None, "Dependencies not initialized -- call init_deps() first"
    return _config


def get_pipeline() -> PetGenPipeline:
    assert _pipeline is not None, "Dependencies not initialized -- call init_deps() first"
    return _pipeline


def get_queue() -> JobQueue:
    assert _queue is not None, "Dependencies not initialized -- call init_deps() first"
    return _queue


def get_progress() -> ProgressManager:
    assert _progress is not None, "Dependencies not initialized -- call init_deps() first"
    return _progress


def reset_deps() -> None:
    """Reset all singletons. Used in tests."""
    global _config, _pipeline, _queue, _progress
    _config = None
    _pipeline = None
    _queue = None
    _progress = None
