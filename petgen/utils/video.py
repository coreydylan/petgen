"""Video processing utilities — FFmpeg wrappers and frame I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_frames(video_path: Path, fps: float | None = None) -> list[np.ndarray]:
    """Extract frames from a video file using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1 if fps is None else max(1, int(source_fps / fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    return frames


def frames_to_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int = 25,
    codec: str = "libx264",
    crf: int = 18,
) -> Path:
    """Write frames to a video file using FFmpeg via subprocess."""
    import subprocess
    import tempfile

    if not frames:
        raise ValueError("No frames to encode")

    h, w = frames[0].shape[:2]
    with tempfile.NamedTemporaryFile(suffix=".rgb", delete=False) as tmp:
        tmp_path = tmp.name
        for frame in frames:
            tmp.write(frame.tobytes())

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", tmp_path,
        "-c:v", codec,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    Path(tmp_path).unlink(missing_ok=True)
    return output_path


def mux_audio_video(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
) -> Path:
    """Combine video and audio into a single file using FFmpeg."""
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of an audio file in seconds."""
    import subprocess
    import json

    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def resize_frames(
    frames: list[np.ndarray],
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Resize all frames to target dimensions."""
    import cv2

    return [cv2.resize(f, (width, height), interpolation=cv2.INTER_LANCZOS4) for f in frames]
