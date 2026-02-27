"""Video download, frame extraction, and dataset preparation for animal mouth training."""

from __future__ import annotations

import json
import logging
import subprocess
import uuid
from pathlib import Path

import numpy as np

from petgen.config import PetGenConfig
from petgen.models import BoundingBox, Landmark

logger = logging.getLogger(__name__)


class AnimalMouthDataCollector:
    """Collect and process animal talking video data for training."""

    def __init__(self, config: PetGenConfig, output_dir: Path) -> None:
        self.config = config
        self.output_dir = output_dir
        self._face_detector = None

    def _get_face_detector(self):
        """Lazy-load PetFaceDetector."""
        if self._face_detector is None:
            from petgen.modules.face_detect import PetFaceDetector

            self._face_detector = PetFaceDetector(self.config)
        return self._face_detector

    def download_video(self, url: str, output_path: Path) -> Path:
        """Download video using yt-dlp (must be installed).

        Args:
            url: Video URL (YouTube, direct link, etc.).
            output_path: Where to save the downloaded file.

        Returns:
            Path to the downloaded video file.

        Raises:
            RuntimeError: If yt-dlp is not installed or download fails.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(output_path),
            url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp not found. Install with: pip install yt-dlp"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"yt-dlp failed: {e.stderr}")

        logger.info("Downloaded video to %s", output_path)
        return output_path

    def extract_frames(self, video_path: Path, fps: int = 25) -> list[Path]:
        """Extract frames at target FPS using FFmpeg.

        Args:
            video_path: Path to input video.
            fps: Target frames per second.

        Returns:
            Sorted list of paths to extracted frame images.
        """
        frames_dir = video_path.parent / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            str(frames_dir / "frame_%06d.jpg"),
        ]
        subprocess.run(cmd, check=True, capture_output=True)

        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
        logger.info("Extracted %d frames at %d FPS", len(frame_paths), fps)
        return frame_paths

    def detect_and_crop_faces(self, frame_paths: list[Path]) -> list[dict]:
        """Detect pet faces and crop each frame.

        Args:
            frame_paths: List of frame image paths.

        Returns:
            List of dicts with keys: frame_path, crop_path, bbox, species.
            Frames with no detected face are skipped.
        """
        import cv2

        detector = self._get_face_detector()
        results = []

        crops_dir = frame_paths[0].parent.parent / "crops" if frame_paths else Path()
        crops_dir.mkdir(parents=True, exist_ok=True)

        for fp in frame_paths:
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bbox = detector.detect_face(img_rgb)
            if bbox is None:
                continue

            species = detector.get_species(img_rgb)

            # Crop with padding
            h, w = img.shape[:2]
            pad = 0.15
            x1 = max(0, int(bbox.x1 - bbox.width * pad))
            y1 = max(0, int(bbox.y1 - bbox.height * pad))
            x2 = min(w, int(bbox.x2 + bbox.width * pad))
            y2 = min(h, int(bbox.y2 + bbox.height * pad))
            crop = img[y1:y2, x1:x2]

            crop_path = crops_dir / fp.name
            cv2.imwrite(str(crop_path), crop)

            results.append({
                "frame_path": str(fp),
                "crop_path": str(crop_path),
                "bbox": {"x1": bbox.x1, "y1": bbox.y1, "x2": bbox.x2, "y2": bbox.y2},
                "species": species,
            })

        logger.info("Detected faces in %d / %d frames", len(results), len(frame_paths))
        return results

    def generate_mouth_masks(
        self, frame_paths: list[Path], bboxes: list[BoundingBox]
    ) -> list[Path]:
        """Generate mouth segmentation masks using SAM2.

        Uses face landmarks to seed SAM2 mouth prompts.

        Args:
            frame_paths: Paths to frame images.
            bboxes: Corresponding bounding boxes for each frame.

        Returns:
            List of paths to saved mask images.
        """
        import cv2

        detector = self._get_face_detector()
        masks_dir = frame_paths[0].parent.parent / "masks" if frame_paths else Path()
        masks_dir.mkdir(parents=True, exist_ok=True)

        mask_paths = []
        for fp, bbox in zip(frame_paths, bboxes):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            landmarks = detector.detect_landmarks(img_rgb, bbox)
            mask = detector.segment_mouth(img_rgb, landmarks)

            mask_path = masks_dir / f"mask_{fp.stem}.png"
            cv2.imwrite(str(mask_path), mask)
            mask_paths.append(mask_path)

        logger.info("Generated %d mouth masks", len(mask_paths))
        return mask_paths

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio track as WAV.

        Args:
            video_path: Path to input video.

        Returns:
            Path to extracted WAV file.
        """
        audio_path = video_path.parent / "audio.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info("Extracted audio to %s", audio_path)
        return audio_path

    def build_dataset(self, video_url: str, fps: int = 25) -> dict:
        """Full pipeline: download -> extract -> detect -> mask -> align.

        Args:
            video_url: URL of the video to process.
            fps: Target frames per second.

        Returns:
            Metadata dict with all paths and dataset info.
        """
        dataset_id = str(uuid.uuid4())[:8]
        dataset_dir = self.output_dir / f"dataset_{dataset_id}"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Download
        video_path = dataset_dir / "source.mp4"
        self.download_video(video_url, video_path)

        # Extract frames
        frame_paths = self.extract_frames(video_path, fps=fps)

        # Detect and crop faces
        face_data = self.detect_and_crop_faces(frame_paths)

        # Generate mouth masks for frames with detected faces
        detected_paths = [Path(d["frame_path"]) for d in face_data]
        detected_bboxes = [
            BoundingBox(
                x1=d["bbox"]["x1"], y1=d["bbox"]["y1"],
                x2=d["bbox"]["x2"], y2=d["bbox"]["y2"],
            )
            for d in face_data
        ]
        mask_paths = self.generate_mouth_masks(detected_paths, detected_bboxes)

        # Extract audio
        audio_path = self.extract_audio(video_path)

        metadata = {
            "dataset_id": dataset_id,
            "video_url": video_url,
            "video_path": str(video_path),
            "audio_path": str(audio_path),
            "fps": fps,
            "num_frames": len(frame_paths),
            "num_detected": len(face_data),
            "face_data": face_data,
            "mask_paths": [str(p) for p in mask_paths],
        }

        self.save_metadata(metadata, dataset_dir / "metadata.json")
        logger.info("Dataset %s built: %d frames, %d faces detected",
                     dataset_id, len(frame_paths), len(face_data))
        return metadata

    def save_metadata(self, metadata: dict, output_path: Path) -> None:
        """Save dataset metadata as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metadata, indent=2))
