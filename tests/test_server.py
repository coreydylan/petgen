"""Tests for the PetGen FastAPI server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from petgen.config import PetGenConfig
from petgen.models import (
    BreedGroup,
    CharacterProfile,
    PipelineResult,
    VoiceSettings,
)
from petgen.server.models import JobStatus
from petgen.server.queue import Job, JobQueue
from petgen.server.ws import ProgressManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_character(char_id: str = "char-001", name: str = "Max") -> CharacterProfile:
    return CharacterProfile(
        id=char_id,
        name=name,
        breed="Golden Retriever",
        breed_group=BreedGroup.LARGE_DOG,
        reference_images=[Path("/fake/ref_0.png")],
        canonical_poses={"front": Path("/fake/pose_front.png")},
        voice=VoiceSettings(exaggeration=0.5, speed=0.9),
        personality_tags=["friendly"],
    )


@pytest.fixture
def mock_pipeline():
    """Create a mock PetGenPipeline with all sub-modules stubbed."""
    pipeline = MagicMock()
    pipeline.config = MagicMock(spec=PetGenConfig)
    pipeline.config.characters_dir = Path("/tmp/petgen_test/characters")
    pipeline.config.output_dir = Path("/tmp/petgen_test/output")
    pipeline.config.temp_dir = Path("/tmp/petgen_test/tmp")

    # Character creator stubs
    pipeline.character_creator.list_characters.return_value = [_make_character()]
    pipeline.character_creator.get_character.return_value = _make_character()
    pipeline.character_creator.create_character.return_value = _make_character()
    pipeline.create_character.return_value = _make_character()

    # Voice generator stub
    pipeline.voice_generator.generate.return_value = Path("/tmp/petgen_test/voice.wav")

    # Scene generator stub
    pipeline.character_creator.generate_scene.return_value = Path("/tmp/petgen_test/scene.png")

    # Full pipeline stub
    pipeline.generate.return_value = PipelineResult(
        video_path=Path("/tmp/petgen_test/output/video.mp4"),
        audio_path=Path("/tmp/petgen_test/voice.wav"),
        character=_make_character(),
        duration=3.0,
        num_frames=75,
    )

    return pipeline


@pytest.fixture
def app(mock_pipeline, tmp_path):
    """Create a test FastAPI app with mocked dependencies injected directly."""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from petgen.server import deps
    from petgen.server.routes import characters, generate, scenes, voice

    # Reset and manually set deps so we bypass the lifespan handler
    deps.reset_deps()

    config = PetGenConfig(
        base_dir=tmp_path / "petgen",
        characters_dir=tmp_path / "petgen" / "characters",
        output_dir=tmp_path / "petgen" / "output",
        temp_dir=tmp_path / "petgen" / "tmp",
        weights_dir=tmp_path / "petgen" / "weights",
        voice_presets_dir=tmp_path / "petgen" / "voice_presets",
        gemini_api_key="test-key",
        device="cpu",
        tts_device="cpu",
    )
    config.ensure_dirs()

    deps._config = config
    deps._pipeline = mock_pipeline
    deps._queue = JobQueue(max_concurrent=1)
    deps._progress = ProgressManager()
    deps._queue.set_progress_callback(deps._progress.emit_progress)

    # Override pipeline config to use test dirs
    mock_pipeline.config = config

    # Build app without lifespan so deps stay mocked
    application = FastAPI(title="PetGen API Test")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(characters.router)
    application.include_router(generate.router)
    application.include_router(voice.router)
    application.include_router(scenes.router)

    @application.get("/api/health")
    async def health():
        return {"status": "ok"}

    yield application

    deps.reset_deps()


@pytest.fixture
def client(app):
    """Create a test client that skips the lifespan (deps already initialized)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_endpoint(self, client: TestClient) -> None:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Characters CRUD
# ---------------------------------------------------------------------------


class TestCharacters:
    def test_list_characters(self, client: TestClient) -> None:
        resp = client.get("/api/characters")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "Max"
        assert data[0]["breed"] == "Golden Retriever"
        assert data[0]["breed_group"] == "large_dog"

    def test_get_character(self, client: TestClient) -> None:
        resp = client.get("/api/characters/char-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "char-001"
        assert data["name"] == "Max"
        assert "front" in data["poses"]

    def test_get_character_not_found(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.character_creator.get_character.return_value = None
        resp = client.get("/api/characters/nonexistent")
        assert resp.status_code == 404

    def test_create_character(self, client: TestClient, tmp_path: Path) -> None:
        # Create a dummy image to upload
        img_path = tmp_path / "test_photo.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        resp = client.post(
            "/api/characters",
            data={"name": "Buddy", "breed": "Labrador"},
            files=[("photos", ("test.jpg", img_path.read_bytes(), "image/jpeg"))],
        )
        assert resp.status_code == 201, resp.text
        data = resp.json()
        assert data["name"] == "Max"  # returned from mock

    def test_create_character_no_photos(self, client: TestClient) -> None:
        resp = client.post(
            "/api/characters",
            data={"name": "Buddy"},
        )
        # FastAPI returns 422 for missing required file field
        assert resp.status_code == 422

    def test_delete_character(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        resp = client.delete("/api/characters/char-001")
        assert resp.status_code == 204

    def test_delete_character_not_found(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.character_creator.get_character.return_value = None
        resp = client.delete("/api/characters/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Generation jobs
# ---------------------------------------------------------------------------


class TestGeneration:
    def test_submit_job(self, client: TestClient) -> None:
        resp = client.post(
            "/api/generate",
            json={
                "character_id": "char-001",
                "script": "Hello world!",
                "emotion": 0.7,
            },
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "created_at" in data

    def test_get_job_status(self, client: TestClient) -> None:
        # Submit first
        resp = client.post(
            "/api/generate",
            json={"character_id": "char-001", "script": "Test"},
        )
        job_id = resp.json()["job_id"]

        # Poll status
        resp = client.get(f"/api/generate/{job_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["pending", "processing", "completed", "failed"]

    def test_get_job_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/generate/nonexistent-id")
        assert resp.status_code == 404

    def test_download_not_completed(self, client: TestClient) -> None:
        # Submit a job
        resp = client.post(
            "/api/generate",
            json={"character_id": "char-001", "script": "Test"},
        )
        job_id = resp.json()["job_id"]

        # Try to download before completion
        resp = client.get(f"/api/generate/{job_id}/download")
        assert resp.status_code == 400

    def test_download_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/generate/nonexistent/download")
        assert resp.status_code == 404

    def test_submit_validation_error(self, client: TestClient) -> None:
        # Missing required script field
        resp = client.post(
            "/api/generate",
            json={"character_id": "char-001"},
        )
        assert resp.status_code == 422

    def test_submit_invalid_aspect_ratio(self, client: TestClient) -> None:
        resp = client.post(
            "/api/generate",
            json={
                "character_id": "char-001",
                "script": "Test",
                "aspect_ratio": "4:3",
            },
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Voice endpoints
# ---------------------------------------------------------------------------


class TestVoice:
    def test_generate_voice(
        self, client: TestClient, mock_pipeline: MagicMock, tmp_path: Path
    ) -> None:
        # Set up the mock to return a real file
        audio_file = tmp_path / "voice.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
        mock_pipeline.voice_generator.generate.return_value = audio_file

        resp = client.post(
            "/api/voice",
            json={
                "character_id": "char-001",
                "script": "Hello!",
                "emotion": 0.8,
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_voice_character_not_found(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.character_creator.get_character.return_value = None
        resp = client.post(
            "/api/voice",
            json={
                "character_id": "nonexistent",
                "script": "Hello!",
            },
        )
        assert resp.status_code == 404

    def test_preview_voice(
        self, client: TestClient, mock_pipeline: MagicMock, tmp_path: Path
    ) -> None:
        audio_file = tmp_path / "preview.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)
        mock_pipeline.voice_generator.generate.return_value = audio_file

        resp = client.post(
            "/api/voice/preview",
            json={
                "character_id": "char-001",
                "script": "Quick test",
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Scene endpoints
# ---------------------------------------------------------------------------


class TestScenes:
    def test_generate_scene(
        self, client: TestClient, mock_pipeline: MagicMock, tmp_path: Path
    ) -> None:
        scene_file = tmp_path / "scene.png"
        scene_file.write_bytes(b"\x89PNG" + b"\x00" * 100)
        mock_pipeline.character_creator.generate_scene.return_value = scene_file

        resp = client.post(
            "/api/scenes",
            json={
                "character_id": "char-001",
                "prompt": "wearing a hat on a beach",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    def test_scene_character_not_found(
        self, client: TestClient, mock_pipeline: MagicMock
    ) -> None:
        mock_pipeline.character_creator.get_character.return_value = None
        resp = client.post(
            "/api/scenes",
            json={
                "character_id": "nonexistent",
                "prompt": "on a beach",
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Job Queue unit tests
# ---------------------------------------------------------------------------


class TestJobQueue:
    @pytest.mark.asyncio
    async def test_submit_creates_job(self) -> None:
        queue = JobQueue()
        job = await queue.submit({"character_id": "c1", "script": "hi"})
        assert job.id is not None
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_status_returns_job(self) -> None:
        queue = JobQueue()
        job = await queue.submit({"character_id": "c1", "script": "hi"})
        result = await queue.get_status(job.id)
        assert result is not None
        assert result.id == job.id

    @pytest.mark.asyncio
    async def test_get_status_unknown_returns_none(self) -> None:
        queue = JobQueue()
        result = await queue.get_status("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_job_processing_success(self) -> None:
        queue = JobQueue()

        mock_pipeline = MagicMock()
        mock_pipeline.character_creator.get_character.return_value = _make_character()
        mock_pipeline.generate.return_value = PipelineResult(
            video_path=Path("/tmp/out.mp4"),
            audio_path=Path("/tmp/audio.wav"),
            character=_make_character(),
        )

        job = await queue.submit({
            "character_id": "char-001",
            "script": "Hello!",
        })

        # Run the worker briefly in the background
        worker = asyncio.create_task(queue.process_jobs(mock_pipeline))
        await asyncio.sleep(0.2)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        updated = await queue.get_status(job.id)
        assert updated is not None
        assert updated.status == JobStatus.COMPLETED
        assert updated.result_path == Path("/tmp/out.mp4")

    @pytest.mark.asyncio
    async def test_job_processing_failure(self) -> None:
        queue = JobQueue()

        mock_pipeline = MagicMock()
        mock_pipeline.character_creator.get_character.return_value = None

        job = await queue.submit({
            "character_id": "missing",
            "script": "Hello!",
        })

        worker = asyncio.create_task(queue.process_jobs(mock_pipeline))
        await asyncio.sleep(0.2)
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

        updated = await queue.get_status(job.id)
        assert updated is not None
        assert updated.status == JobStatus.FAILED
        assert updated.error is not None


# ---------------------------------------------------------------------------
# WebSocket progress manager unit tests
# ---------------------------------------------------------------------------


class TestProgressManager:
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self) -> None:
        mgr = ProgressManager()
        ws = AsyncMock()
        await mgr.connect("job-1", ws)
        ws.accept.assert_awaited_once()
        assert "job-1" in mgr.connections
        assert len(mgr.connections["job-1"]) == 1

        await mgr.disconnect("job-1", ws)
        assert "job-1" not in mgr.connections

    @pytest.mark.asyncio
    async def test_emit_progress(self) -> None:
        mgr = ProgressManager()
        ws = AsyncMock()
        await mgr.connect("job-1", ws)

        await mgr.emit_progress("job-1", "voice", 0.3, "Generating audio...")

        ws.send_text.assert_awaited_once()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == "progress"
        assert payload["stage"] == "voice"
        assert payload["progress"] == 0.3

    @pytest.mark.asyncio
    async def test_emit_completed(self) -> None:
        mgr = ProgressManager()
        ws = AsyncMock()
        await mgr.connect("job-1", ws)

        await mgr.emit_progress("job-1", "completed", 1.0, "/files/video.mp4")

        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == "completed"
        assert payload["video_url"] == "/files/video.mp4"

    @pytest.mark.asyncio
    async def test_emit_error(self) -> None:
        mgr = ProgressManager()
        ws = AsyncMock()
        await mgr.connect("job-1", ws)

        await mgr.emit_progress("job-1", "failed", 0.5, "GPU out of memory")

        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == "error"
        assert payload["message"] == "GPU out of memory"

    @pytest.mark.asyncio
    async def test_emit_to_no_connections(self) -> None:
        mgr = ProgressManager()
        # Should not raise
        await mgr.emit_progress("unknown-job", "voice", 0.5, "test")

    @pytest.mark.asyncio
    async def test_dead_connection_cleanup(self) -> None:
        mgr = ProgressManager()
        ws = AsyncMock()
        ws.send_text.side_effect = RuntimeError("connection closed")
        await mgr.connect("job-1", ws)

        await mgr.emit_progress("job-1", "voice", 0.3, "test")

        # Dead connection should be cleaned up
        assert "job-1" not in mgr.connections


# ---------------------------------------------------------------------------
# API models validation
# ---------------------------------------------------------------------------


class TestAPIModels:
    def test_generate_request_defaults(self) -> None:
        from petgen.server.models import GenerateRequest

        req = GenerateRequest(character_id="c1", script="Hello")
        assert req.emotion == 0.7
        assert req.speed == 1.0
        assert req.aspect_ratio == "9:16"

    def test_generate_request_invalid_emotion(self) -> None:
        from petgen.server.models import GenerateRequest

        with pytest.raises(Exception):
            GenerateRequest(character_id="c1", script="Hello", emotion=2.0)

    def test_generate_request_invalid_aspect(self) -> None:
        from petgen.server.models import GenerateRequest

        with pytest.raises(Exception):
            GenerateRequest(character_id="c1", script="Hello", aspect_ratio="4:3")

    def test_voice_request_defaults(self) -> None:
        from petgen.server.models import VoiceRequest

        req = VoiceRequest(character_id="c1", script="Hello")
        assert req.emotion == 0.7

    def test_scene_request_defaults(self) -> None:
        from petgen.server.models import SceneRequest

        req = SceneRequest(character_id="c1", prompt="on a beach")
        assert req.style == "photorealistic"

    def test_create_character_request(self) -> None:
        from petgen.server.models import CreateCharacterRequest

        req = CreateCharacterRequest(name="Buddy")
        assert req.name == "Buddy"
        assert req.breed is None


# ---------------------------------------------------------------------------
# Deps module tests
# ---------------------------------------------------------------------------


class TestDeps:
    def test_reset_deps(self) -> None:
        from petgen.server import deps

        deps.reset_deps()
        with pytest.raises(AssertionError):
            deps.get_config()
        with pytest.raises(AssertionError):
            deps.get_pipeline()
        with pytest.raises(AssertionError):
            deps.get_queue()
        with pytest.raises(AssertionError):
            deps.get_progress()

    def test_init_deps(self, tmp_path: Path) -> None:
        from petgen.server import deps

        deps.reset_deps()

        with patch.dict(
            "os.environ",
            {"PETGEN_BASE_DIR": str(tmp_path / "petgen")},
            clear=False,
        ):
            deps.init_deps()
            assert deps.get_config() is not None
            assert deps.get_pipeline() is not None
            assert deps.get_queue() is not None
            assert deps.get_progress() is not None

        deps.reset_deps()


# ---------------------------------------------------------------------------
# WebSocket integration test
# ---------------------------------------------------------------------------


class TestWebSocket:
    def test_ws_job_not_found(self, client: TestClient) -> None:
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/generate/nonexistent"):
                pass

    def test_ws_connect_to_valid_job(self, client: TestClient) -> None:
        # Submit a job first
        resp = client.post(
            "/api/generate",
            json={"character_id": "char-001", "script": "Test"},
        )
        job_id = resp.json()["job_id"]

        # Connect WebSocket — just verify connection is accepted
        with client.websocket_connect(f"/ws/generate/{job_id}") as ws:
            # Connection was accepted successfully
            pass
