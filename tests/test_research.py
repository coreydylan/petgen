"""Tests for the MuseTalk animal adaptation R&D research package."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from petgen.models import BoundingBox, Landmark


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def research_output_dir(tmp_path):
    d = tmp_path / "research_output"
    d.mkdir()
    return d


@pytest.fixture
def sample_frames():
    """Create a list of random RGB frames."""
    return [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def sample_landmarks_48():
    """Full CatFLW-style 48-landmark set with spread-out positions."""
    landmarks = []
    # Distribute landmarks in a face-like pattern across a 256x256 image
    # Ears (0-3): top of face
    for i in range(4):
        landmarks.append(Landmark(x=float(80 + i * 30), y=float(40 + i * 10), visibility=1.0))
    # Eyes (4-11): upper-middle region
    for i in range(8):
        landmarks.append(Landmark(x=float(70 + i * 18), y=float(80 + (i % 2) * 20), visibility=1.0))
    # Mid-face (12-25): cheeks and bridge
    for i in range(14):
        landmarks.append(Landmark(x=float(60 + i * 10), y=float(110 + (i % 3) * 15), visibility=1.0))
    # Nose (26-32): center
    for i in range(7):
        landmarks.append(Landmark(x=float(100 + i * 10), y=float(140 + (i % 3) * 10), visibility=1.0))
    # Mouth (33-47): lower face, well-spread
    for i in range(15):
        angle = np.pi * i / 14
        landmarks.append(Landmark(
            x=float(128 + 30 * np.cos(angle)),
            y=float(190 + 15 * np.sin(angle)),
            visibility=1.0,
        ))
    return landmarks


@pytest.fixture
def sample_dataset_dir(tmp_path):
    """Create a mock dataset directory with pairs.json and frame/mask files."""
    ds_dir = tmp_path / "dataset"
    ds_dir.mkdir()
    frames_dir = ds_dir / "frames"
    masks_dir = ds_dir / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()

    import cv2

    pairs = []
    for i in range(3):
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255

        frame_path = frames_dir / f"frame_{i:04d}.jpg"
        mask_path = masks_dir / f"mask_frame_{i:04d}.png"
        cv2.imwrite(str(frame_path), frame)
        cv2.imwrite(str(mask_path), mask)

        pairs.append({"frame": str(frame_path), "mask": str(mask_path), "index": i})

    (ds_dir / "pairs.json").write_text(json.dumps(pairs))
    return ds_dir


# ------------------------------------------------------------------
# AnimalMouthDataCollector
# ------------------------------------------------------------------


class TestCollectorExtractFrames:
    def test_extract_frames_calls_ffmpeg(self, tmp_config, research_output_dir, tmp_path):
        from petgen.research.data.collector import AnimalMouthDataCollector

        collector = AnimalMouthDataCollector(tmp_config, research_output_dir)
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        frames_dir = video_path.parent / "frames"

        with patch("petgen.research.data.collector.subprocess.run") as mock_run:
            # Simulate that ffmpeg creates some frame files
            def create_frames(*args, **kwargs):
                frames_dir.mkdir(parents=True, exist_ok=True)
                for i in range(3):
                    (frames_dir / f"frame_{i + 1:06d}.jpg").touch()

            mock_run.side_effect = create_frames
            result = collector.extract_frames(video_path, fps=10)

        assert len(result) == 3
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd
        assert "fps=10" in " ".join(cmd)


class TestCollectorBuildDataset:
    def test_build_dataset_orchestrates_pipeline(self, tmp_config, research_output_dir):
        from petgen.research.data.collector import AnimalMouthDataCollector

        collector = AnimalMouthDataCollector(tmp_config, research_output_dir)

        with (
            patch.object(collector, "download_video", return_value=Path("/fake/video.mp4")) as mock_dl,
            patch.object(collector, "extract_frames", return_value=[Path("/fake/f1.jpg")]) as mock_ef,
            patch.object(collector, "detect_and_crop_faces", return_value=[{
                "frame_path": "/fake/f1.jpg",
                "crop_path": "/fake/c1.jpg",
                "bbox": {"x1": 10, "y1": 10, "x2": 100, "y2": 100},
                "species": "cat",
            }]) as mock_dc,
            patch.object(collector, "generate_mouth_masks", return_value=[Path("/fake/m1.png")]) as mock_gm,
            patch.object(collector, "extract_audio", return_value=Path("/fake/audio.wav")) as mock_ea,
        ):
            metadata = collector.build_dataset("https://example.com/video", fps=15)

        mock_dl.assert_called_once()
        mock_ef.assert_called_once()
        mock_dc.assert_called_once()
        mock_gm.assert_called_once()
        mock_ea.assert_called_once()

        assert "dataset_id" in metadata
        assert metadata["fps"] == 15
        assert metadata["num_frames"] == 1
        assert metadata["num_detected"] == 1


# ------------------------------------------------------------------
# AnimalFaceParser
# ------------------------------------------------------------------


class TestFaceParserRegions:
    def test_parse_with_landmarks_returns_correct_shape(self, tmp_config, sample_landmarks_48):
        from petgen.research.face_parser import AnimalFaceParser

        parser = AnimalFaceParser(tmp_config)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        seg_map = parser.parse(image, landmarks=sample_landmarks_48)

        assert seg_map.shape == (256, 256)
        assert seg_map.dtype == np.uint8

    def test_parse_contains_mouth_region(self, tmp_config, sample_landmarks_48):
        from petgen.research.face_parser import AnimalFaceParser

        parser = AnimalFaceParser(tmp_config)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        seg_map = parser.parse(image, landmarks=sample_landmarks_48)

        # Region 6 = mouth, should appear given landmarks 33-47
        assert 6 in seg_map, "Mouth region (6) should be present in segmentation map"

    def test_parse_empty_landmarks_returns_zeros(self, tmp_config):
        from petgen.research.face_parser import AnimalFaceParser

        parser = AnimalFaceParser(tmp_config)
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # Mock detector to return None (no face)
        parser._face_detector = MagicMock()
        parser._face_detector.detect_face.return_value = None

        seg_map = parser.parse(image, landmarks=[])

        assert seg_map.shape == (128, 128)
        assert seg_map.max() == 0

    def test_parse_short_landmarks(self, tmp_config):
        from petgen.research.face_parser import AnimalFaceParser

        parser = AnimalFaceParser(tmp_config)
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        # Short landmark list triggers approximate groups
        short_lms = [
            Landmark(x=float(40 + i * 5), y=float(40 + i * 3), visibility=1.0)
            for i in range(20)
        ]

        seg_map = parser.parse(image, landmarks=short_lms)

        assert seg_map.shape == (128, 128)


class TestFaceParserMouthRegion:
    def test_get_mouth_region_extracts_mask(self, tmp_config):
        from petgen.research.face_parser import AnimalFaceParser

        parser = AnimalFaceParser(tmp_config)
        seg_map = np.zeros((64, 64), dtype=np.uint8)
        seg_map[20:30, 25:35] = 6  # mouth region

        mouth_mask = parser.get_mouth_region(seg_map)

        assert mouth_mask.shape == (64, 64)
        assert mouth_mask.dtype == np.uint8
        assert mouth_mask[25, 30] == 255  # inside mouth
        assert mouth_mask[0, 0] == 0      # outside mouth

    def test_get_mouth_region_empty(self, tmp_config):
        from petgen.research.face_parser import AnimalFaceParser

        parser = AnimalFaceParser(tmp_config)
        seg_map = np.zeros((64, 64), dtype=np.uint8)

        mouth_mask = parser.get_mouth_region(seg_map)

        assert mouth_mask.max() == 0

    def test_visualize_produces_overlay(self, tmp_config, sample_landmarks_48):
        from petgen.research.face_parser import AnimalFaceParser

        parser = AnimalFaceParser(tmp_config)
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        seg_map = parser.parse(image, landmarks=sample_landmarks_48)
        vis = parser.visualize(image, seg_map)

        assert vis.shape == image.shape
        assert vis.dtype == np.uint8


# ------------------------------------------------------------------
# MuseTalkAnimalAdapter
# ------------------------------------------------------------------


class TestAdapterPreprocess:
    def test_preprocess_with_mocked_detector(self, tmp_config, sample_landmarks_48):
        from petgen.research.musetalk_adapter import MuseTalkAnimalAdapter

        adapter = MuseTalkAnimalAdapter(tmp_config)

        mock_detector = MagicMock()
        mock_detector.detect_face.return_value = BoundingBox(
            x1=50, y1=50, x2=200, y2=200, confidence=0.9
        )
        mock_detector.detect_landmarks.return_value = sample_landmarks_48
        adapter.face_detector = mock_detector

        mock_parser = MagicMock()
        seg_map = np.zeros((256, 256), dtype=np.uint8)
        seg_map[100:150, 100:150] = 6
        mock_parser.parse.return_value = seg_map
        mock_parser.get_mouth_region.return_value = (seg_map == 6).astype(np.uint8) * 255
        adapter.face_parser = mock_parser

        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        result = adapter.preprocess_frame(image)

        assert "face_crop" in result
        assert "mouth_mask" in result
        assert "landmarks" in result
        assert "bbox" in result
        assert result["face_crop"].ndim == 3
        mock_detector.detect_face.assert_called_once()

    def test_preprocess_no_face_returns_empty(self, tmp_config):
        from petgen.research.musetalk_adapter import MuseTalkAnimalAdapter

        adapter = MuseTalkAnimalAdapter(tmp_config)
        mock_detector = MagicMock()
        mock_detector.detect_face.return_value = None
        adapter.face_detector = mock_detector

        image = np.zeros((128, 128, 3), dtype=np.uint8)
        result = adapter.preprocess_frame(image)

        assert result == {}


class TestAdapterPassthrough:
    def test_inpaint_mouth_passthrough_without_weights(self, tmp_config):
        from petgen.research.musetalk_adapter import MuseTalkAnimalAdapter

        adapter = MuseTalkAnimalAdapter(tmp_config)
        face_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mouth_mask = np.zeros((64, 64), dtype=np.uint8)
        audio_features = np.random.randn(768).astype(np.float32)

        result = adapter.inpaint_mouth(face_crop, mouth_mask, audio_features, 0)

        np.testing.assert_array_equal(result, face_crop)

    def test_generate_frames_passthrough(self, tmp_config, tmp_path, sample_landmarks_48):
        from petgen.research.musetalk_adapter import MuseTalkAnimalAdapter

        adapter = MuseTalkAnimalAdapter(tmp_config)

        # Mock face detection
        mock_detector = MagicMock()
        mock_detector.detect_face.return_value = BoundingBox(
            x1=10, y1=10, x2=50, y2=50, confidence=0.9
        )
        mock_detector.detect_landmarks.return_value = sample_landmarks_48
        adapter.face_detector = mock_detector

        # Mock parser
        mock_parser = MagicMock()
        seg_map = np.zeros((64, 64), dtype=np.uint8)
        mock_parser.parse.return_value = seg_map
        mock_parser.get_mouth_region.return_value = np.zeros((64, 64), dtype=np.uint8)
        adapter.face_parser = mock_parser

        # Mock audio encoder to return empty features (no model)
        adapter._audio_encoder = "stub"  # mark as loaded
        with patch.object(adapter, "encode_audio", return_value=np.zeros((10, 768), dtype=np.float32)):
            source = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            audio_path = tmp_path / "test.wav"
            audio_path.touch()

            frames = adapter.generate_frames(source, audio_path, fps=25)

        assert len(frames) == 10
        assert frames[0].shape == (64, 64, 3)


# ------------------------------------------------------------------
# PetGenBenchmark
# ------------------------------------------------------------------


class TestBenchmarkTemporalConsistency:
    def test_identical_frames_have_low_error(self, research_output_dir):
        from petgen.research.benchmark import PetGenBenchmark

        bench = PetGenBenchmark(research_output_dir)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        frames = [frame.copy() for _ in range(5)]

        score = bench.compute_temporal_consistency(frames)

        # Identical frames should have very low warping error
        assert score < 50.0  # very low MSE

    def test_single_frame_returns_zero(self, research_output_dir):
        from petgen.research.benchmark import PetGenBenchmark

        bench = PetGenBenchmark(research_output_dir)
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        score = bench.compute_temporal_consistency([frame])

        assert score == 0.0

    def test_empty_frames_returns_zero(self, research_output_dir):
        from petgen.research.benchmark import PetGenBenchmark

        bench = PetGenBenchmark(research_output_dir)

        assert bench.compute_temporal_consistency([]) == 0.0


class TestBenchmarkComparison:
    def test_compare_approaches_returns_metrics(self, research_output_dir, tmp_path):
        from petgen.research.benchmark import PetGenBenchmark

        bench = PetGenBenchmark(research_output_dir)
        source = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        audio = tmp_path / "audio.wav"
        audio.touch()

        warp_frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        inpaint_frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

        results = bench.compare_approaches(
            source, audio, warp_frames, inpaint_frames
        )

        assert "warping" in results
        assert "inpainting" in results
        assert "temporal_consistency" in results["warping"]
        assert "temporal_consistency" in results["inpainting"]
        assert results["warping"]["num_frames"] == 5

    def test_compare_with_ground_truth(self, research_output_dir, tmp_path):
        from petgen.research.benchmark import PetGenBenchmark

        bench = PetGenBenchmark(research_output_dir)
        source = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        audio = tmp_path / "audio.wav"
        audio.touch()

        gt_frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        warp_frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
        inpaint_frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(5)]

        results = bench.compare_approaches(
            source, audio, warp_frames, inpaint_frames, ground_truth=gt_frames
        )

        assert "fid" in results["warping"]
        assert "fid" in results["inpainting"]

    def test_mouth_ssim(self, research_output_dir):
        from petgen.research.benchmark import PetGenBenchmark

        bench = PetGenBenchmark(research_output_dir)

        # Identical frames should have high SSIM
        frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255

        ssim = bench.compute_mouth_ssim([frame], [frame.copy()], [mask])

        assert ssim > 0.9  # near-perfect for identical frames

    def test_fid_identical_returns_low(self, research_output_dir):
        from petgen.research.benchmark import PetGenBenchmark

        bench = PetGenBenchmark(research_output_dir)
        frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(10)]

        fid = bench.compute_fid(frames, frames)

        assert fid >= 0.0
        assert fid < 1.0  # should be near zero for identical sets


# ------------------------------------------------------------------
# AnimalMouthDataset (training)
# ------------------------------------------------------------------


class TestTrainingDataset:
    def test_dataset_length(self, sample_dataset_dir):
        from petgen.research.train_musetalk import AnimalMouthDataset

        ds = AnimalMouthDataset(sample_dataset_dir)

        assert len(ds) == 3

    def test_dataset_getitem(self, sample_dataset_dir):
        from petgen.research.train_musetalk import AnimalMouthDataset

        ds = AnimalMouthDataset(sample_dataset_dir)
        sample = ds[0]

        assert "frame" in sample
        assert "mouth_mask" in sample
        assert "audio_features" in sample
        assert "target_frame" in sample
        assert sample["frame"].dtype == np.float32
        assert sample["mouth_mask"].dtype == np.float32
        assert sample["audio_features"].shape == (768,)

    def test_dataset_empty_dir(self, tmp_path):
        from petgen.research.train_musetalk import AnimalMouthDataset

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        ds = AnimalMouthDataset(empty_dir)

        assert len(ds) == 0


# ------------------------------------------------------------------
# MouthInpaintingLoss (training)
# ------------------------------------------------------------------


class TestTrainingLoss:
    def test_loss_computation_with_random_tensors(self):
        from petgen.research.train_musetalk import MouthInpaintingLoss

        loss_fn = MouthInpaintingLoss(l1_weight=1.0, perceptual_weight=0.5, adversarial_weight=0.1)

        predicted = np.random.rand(64, 64, 3).astype(np.float32)
        target = np.random.rand(64, 64, 3).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:40, 20:40] = 1.0

        losses = loss_fn(predicted, target, mask)

        assert "total" in losses
        assert "l1" in losses
        assert "perceptual" in losses
        assert "adversarial" in losses
        assert losses["total"] >= 0.0
        assert losses["l1"] >= 0.0
        assert losses["perceptual"] >= 0.0
        assert losses["adversarial"] == 0.0  # stub returns 0

    def test_loss_identical_inputs(self):
        from petgen.research.train_musetalk import MouthInpaintingLoss

        loss_fn = MouthInpaintingLoss()
        frame = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.ones((32, 32), dtype=np.float32)

        losses = loss_fn(frame, frame, mask)

        assert losses["l1"] < 1e-6
        assert losses["total"] < 0.01  # near zero

    def test_loss_empty_mask(self):
        from petgen.research.train_musetalk import MouthInpaintingLoss

        loss_fn = MouthInpaintingLoss()
        predicted = np.random.rand(32, 32, 3).astype(np.float32)
        target = np.random.rand(32, 32, 3).astype(np.float32)
        mask = np.zeros((32, 32), dtype=np.float32)

        losses = loss_fn(predicted, target, mask)

        # With empty mask, should still compute l1 (falls back to full-image mean)
        assert losses["total"] >= 0.0
