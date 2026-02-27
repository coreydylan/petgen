"""Tests for the pet face detection and landmark extraction module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from petgen.models import BoundingBox, Landmark
from petgen.modules.face_detect import (
    PetFaceDetector,
    _CAT_MOUTH_START,
    _CAT_NUM_LANDMARKS,
    _DOG_MOUTH_START,
    _DOG_NUM_LANDMARKS,
    _YOLO_CAT_CLASS,
    _YOLO_DOG_CLASS,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def detector(tmp_config):
    return PetFaceDetector(tmp_config)


def _make_yolo_result(cls_ids: list[int], confs: list[float], xyxys: list[list[float]]):
    """Build a fake ultralytics result object."""
    import torch

    boxes = MagicMock()
    boxes.cls = torch.tensor(cls_ids, dtype=torch.float32)
    boxes.conf = torch.tensor(confs, dtype=torch.float32)
    boxes.xyxy = torch.tensor(xyxys, dtype=torch.float32)
    boxes.__len__ = lambda self: len(cls_ids)
    result = MagicMock()
    result.boxes = boxes
    return [result]


# ------------------------------------------------------------------
# detect_face
# ------------------------------------------------------------------


class TestDetectFace:
    def test_detects_cat(self, detector, sample_image):
        fake_results = _make_yolo_result(
            cls_ids=[_YOLO_CAT_CLASS],
            confs=[0.92],
            xyxys=[[50.0, 40.0, 300.0, 280.0]],
        )
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = fake_results

        bbox = detector.detect_face(sample_image)

        assert bbox is not None
        assert isinstance(bbox, BoundingBox)
        assert bbox.confidence == pytest.approx(0.92)
        assert bbox.x1 == pytest.approx(50.0)
        assert bbox.y2 == pytest.approx(280.0)

    def test_detects_dog(self, detector, sample_image):
        fake_results = _make_yolo_result(
            cls_ids=[_YOLO_DOG_CLASS],
            confs=[0.88],
            xyxys=[[10.0, 20.0, 200.0, 250.0]],
        )
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = fake_results

        bbox = detector.detect_face(sample_image)

        assert bbox is not None
        assert bbox.confidence == pytest.approx(0.88)

    def test_picks_highest_confidence_pet(self, detector, sample_image):
        fake_results = _make_yolo_result(
            cls_ids=[_YOLO_DOG_CLASS, _YOLO_CAT_CLASS, 0],  # 0 = person
            confs=[0.6, 0.9, 0.99],
            xyxys=[
                [10.0, 10.0, 100.0, 100.0],
                [50.0, 50.0, 200.0, 200.0],
                [0.0, 0.0, 400.0, 400.0],
            ],
        )
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = fake_results

        bbox = detector.detect_face(sample_image)

        # Should pick the cat (0.9) not the person (0.99)
        assert bbox is not None
        assert bbox.confidence == pytest.approx(0.9)
        assert bbox.x1 == pytest.approx(50.0)

    def test_returns_none_when_no_pets(self, detector, sample_image):
        # Only a person detected (class 0)
        fake_results = _make_yolo_result(
            cls_ids=[0],
            confs=[0.95],
            xyxys=[[0.0, 0.0, 400.0, 400.0]],
        )
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = fake_results

        bbox = detector.detect_face(sample_image)

        assert bbox is None

    def test_returns_none_on_empty_results(self, detector, sample_image):
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = []

        assert detector.detect_face(sample_image) is None

    def test_returns_none_when_no_boxes(self, detector, sample_image):
        result = MagicMock()
        result.boxes = None
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = [result]

        assert detector.detect_face(sample_image) is None


# ------------------------------------------------------------------
# detect_landmarks
# ------------------------------------------------------------------


class TestDetectLandmarks:
    def test_heuristic_fallback_produces_correct_count(self, detector, sample_image, sample_bbox):
        """When no trained model is available, heuristic generates the right number."""
        # _landmark_model stays None -> heuristic fallback
        detector._landmark_model = None

        landmarks = detector.detect_landmarks(sample_image, sample_bbox)

        # Default species is cat -> 48 landmarks
        assert len(landmarks) == _CAT_NUM_LANDMARKS
        assert all(isinstance(lm, Landmark) for lm in landmarks)

    def test_heuristic_landmarks_within_image_bounds(self, detector, sample_image, sample_bbox):
        detector._landmark_model = None

        landmarks = detector.detect_landmarks(sample_image, sample_bbox)
        h, w = sample_image.shape[:2]

        for lm in landmarks:
            # Heuristic landmarks should be roughly near the bounding box
            assert -w < lm.x < 2 * w
            assert -h < lm.y < 2 * h

    def test_heuristic_landmarks_have_low_visibility(self, detector, sample_image, sample_bbox):
        detector._landmark_model = None

        landmarks = detector.detect_landmarks(sample_image, sample_bbox)

        # Heuristic landmarks are marked with 0.5 visibility
        for lm in landmarks:
            assert lm.visibility == pytest.approx(0.5)


# ------------------------------------------------------------------
# extract_mouth_landmarks
# ------------------------------------------------------------------


class TestExtractMouthLandmarks:
    def test_cat_mouth_landmarks(self, detector):
        landmarks = [Landmark(x=float(i), y=float(i), visibility=1.0) for i in range(48)]

        mouth = detector.extract_mouth_landmarks(landmarks)

        assert len(mouth) == 48 - _CAT_MOUTH_START  # 15 points
        assert mouth[0].x == pytest.approx(float(_CAT_MOUTH_START))

    def test_dog_mouth_landmarks(self, detector):
        landmarks = [Landmark(x=float(i), y=float(i), visibility=1.0) for i in range(46)]

        mouth = detector.extract_mouth_landmarks(landmarks)

        assert len(mouth) == 46 - _DOG_MOUTH_START  # 13 points
        assert mouth[0].x == pytest.approx(float(_DOG_MOUTH_START))

    def test_short_list_fallback(self, detector):
        landmarks = [Landmark(x=float(i), y=float(i)) for i in range(12)]

        mouth = detector.extract_mouth_landmarks(landmarks)

        # Should return last third -> 4 landmarks
        assert len(mouth) == 4
        assert mouth[0].x == pytest.approx(8.0)

    def test_empty_landmarks(self, detector):
        mouth = detector.extract_mouth_landmarks([])
        assert mouth == []


# ------------------------------------------------------------------
# segment_mouth
# ------------------------------------------------------------------


class TestSegmentMouth:
    def test_returns_empty_mask_without_mouth_landmarks(self, detector, sample_image):
        mask = detector.segment_mouth(sample_image, [])

        assert mask.shape == sample_image.shape[:2]
        assert mask.max() == 0

    def test_calls_sam2_with_correct_box(self, detector, sample_image):
        landmarks = [Landmark(x=float(i), y=float(i)) for i in range(48)]

        mock_predictor = MagicMock()
        fake_mask = np.ones((1, 512, 512), dtype=np.float32)
        mock_predictor.predict.return_value = (fake_mask, np.array([0.9]), None)
        detector._sam2_predictor = mock_predictor

        mask = detector.segment_mouth(sample_image, landmarks)

        mock_predictor.set_image.assert_called_once_with(sample_image)
        mock_predictor.predict.assert_called_once()
        call_kwargs = mock_predictor.predict.call_args
        assert "box" in call_kwargs.kwargs or len(call_kwargs.args) == 0
        assert mask.dtype == np.uint8
        assert mask.shape == (512, 512)


# ------------------------------------------------------------------
# get_species
# ------------------------------------------------------------------


class TestGetSpecies:
    def test_returns_cat_for_cat_detection(self, detector, sample_image):
        fake = _make_yolo_result([_YOLO_CAT_CLASS], [0.9], [[10, 10, 200, 200]])
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = fake

        assert detector.get_species(sample_image) == "cat"

    def test_returns_dog_for_dog_detection(self, detector, sample_image):
        fake = _make_yolo_result([_YOLO_DOG_CLASS], [0.85], [[10, 10, 200, 200]])
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = fake

        assert detector.get_species(sample_image) == "dog"

    def test_defaults_to_cat_when_empty(self, detector, sample_image):
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = []

        assert detector.get_species(sample_image) == "cat"

    def test_picks_highest_confidence_species(self, detector, sample_image):
        fake = _make_yolo_result(
            [_YOLO_CAT_CLASS, _YOLO_DOG_CLASS],
            [0.5, 0.95],
            [[10, 10, 100, 100], [20, 20, 200, 200]],
        )
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = fake

        assert detector.get_species(sample_image) == "dog"


# ------------------------------------------------------------------
# track_mouth_video
# ------------------------------------------------------------------


class TestTrackMouthVideo:
    def test_empty_frames(self, detector):
        assert detector.track_mouth_video([]) == []

    def test_returns_zero_masks_when_no_face(self, detector):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)] * 5
        detector._yolo_model = MagicMock()
        detector._yolo_model.predict.return_value = []

        masks = detector.track_mouth_video(frames)

        assert len(masks) == 5
        for m in masks:
            assert m.shape == (100, 100)
            assert m.max() == 0
