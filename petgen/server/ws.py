"""WebSocket progress streaming for job updates."""

import json
import logging
from typing import Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ProgressManager:
    """Manages WebSocket connections for job progress updates."""

    def __init__(self) -> None:
        self.connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket) -> None:
        """Register a WebSocket connection for a job."""
        await ws.accept()
        if job_id not in self.connections:
            self.connections[job_id] = []
        self.connections[job_id].append(ws)
        logger.debug("WebSocket connected for job %s", job_id)

    async def disconnect(self, job_id: str, ws: WebSocket) -> None:
        """Remove a WebSocket connection."""
        if job_id in self.connections:
            self.connections[job_id] = [
                c for c in self.connections[job_id] if c is not ws
            ]
            if not self.connections[job_id]:
                del self.connections[job_id]
        logger.debug("WebSocket disconnected for job %s", job_id)

    async def emit_progress(
        self, job_id: str, stage: str, progress: float, message: str
    ) -> None:
        """Send a progress event to all connected clients for a job."""
        if job_id not in self.connections:
            return

        if stage == "completed":
            data = {"type": "completed", "video_url": message}
        elif stage == "failed":
            data = {"type": "error", "message": message}
        else:
            data = {
                "type": "progress",
                "stage": stage,
                "progress": progress,
                "message": message,
            }

        payload = json.dumps(data)
        dead: List[WebSocket] = []
        for ws in self.connections[job_id]:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        # Clean up dead connections
        for ws in dead:
            await self.disconnect(job_id, ws)
