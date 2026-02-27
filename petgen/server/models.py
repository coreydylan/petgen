"""Request and response Pydantic schemas for the PetGen API."""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# --- Character schemas ---


class CreateCharacterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    breed: Optional[str] = None


class CharacterResponse(BaseModel):
    id: str
    name: str
    breed: str
    breed_group: str
    num_refs: int
    poses: List[str]


# --- Generation schemas ---


class GenerateRequest(BaseModel):
    character_id: str
    script: str = Field(..., min_length=1)
    scene_prompt: Optional[str] = None
    emotion: float = Field(default=0.7, ge=0.0, le=1.0)
    speed: float = Field(default=1.0, gt=0.0, le=3.0)
    aspect_ratio: str = Field(default="9:16", pattern=r"^(9:16|1:1|16:9)$")


class GenerateJobResponse(BaseModel):
    job_id: str
    status: JobStatus
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = 0.0
    stage: str = ""
    result_url: Optional[str] = None
    error: Optional[str] = None


# --- Voice schemas ---


class VoiceRequest(BaseModel):
    character_id: str
    script: str = Field(..., min_length=1)
    emotion: float = Field(default=0.7, ge=0.0, le=1.0)


# --- Scene schemas ---


class SceneRequest(BaseModel):
    character_id: str
    prompt: str = Field(..., min_length=1)
    style: str = "photorealistic"
