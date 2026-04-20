import cv2
import numpy as np
import pytest

from src.flow_estimator import (
    FarnebackFlow,
    LucasKanadeFlow,
    create_flow_estimator,
)

H, W = 120, 160


def _make_gray():
    """Synthetic grayscale frame with clear rectangular edges for corner detection."""
    frame = np.zeros((H, W), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (70, 70), 200, -1)
    cv2.rectangle(frame, (90, 40), (150, 100), 150, -1)
    return frame


@pytest.fixture
def structured_frame():
    return _make_gray()


@pytest.fixture
def frame_pair():
    prev = _make_gray()
    # Shift right by 3px so LK has real motion to track
    curr = np.roll(prev, 3, axis=1)
    return prev, curr


class TestLucasKanadeFlow:
    def test_detect_features_returns_points(self, structured_frame):
        lk = LucasKanadeFlow(max_corners=100, quality_level=0.01, min_distance=5)
        pts = lk.detect_features(structured_frame)
        assert pts is not None
        assert pts.ndim == 3            # (N, 1, 2)
        assert pts.shape[1] == 1
        assert pts.shape[2] == 2

    def test_compute_shapes_match(self, frame_pair):
        lk = LucasKanadeFlow(max_corners=100, quality_level=0.01, min_distance=5)
        prev, curr = frame_pair
        pts = lk.detect_features(prev)
        good_prev, good_curr = lk.compute(prev, curr, pts)
        assert len(good_prev) == len(good_curr)

    def test_compute_none_points_returns_empty(self, frame_pair):
        lk = LucasKanadeFlow()
        prev, curr = frame_pair
        gp, gc = lk.compute(prev, curr, None)
        assert len(gp) == 0
        assert len(gc) == 0

    def test_compute_empty_points_returns_empty(self, frame_pair):
        lk = LucasKanadeFlow()
        prev, curr = frame_pair
        gp, gc = lk.compute(prev, curr, np.array([]))
        assert len(gp) == 0
        assert len(gc) == 0


class TestFarnebackFlow:
    def test_output_shape(self, frame_pair):
        fb = FarnebackFlow()
        prev, curr = frame_pair
        flow = fb.compute(prev, curr)
        assert flow.shape == (H, W, 2)

    def test_output_dtype(self, frame_pair):
        fb = FarnebackFlow()
        prev, curr = frame_pair
        flow = fb.compute(prev, curr)
        assert flow.dtype == np.float32

    def test_identical_frames_near_zero_flow(self, structured_frame):
        fb = FarnebackFlow()
        flow = fb.compute(structured_frame, structured_frame)
        assert np.abs(flow).max() < 0.1

    def test_shifted_frame_has_nonzero_flow(self, frame_pair):
        fb = FarnebackFlow()
        prev, curr = frame_pair
        flow = fb.compute(prev, curr)
        assert np.abs(flow).max() > 0


class TestCreateFlowEstimator:
    def test_creates_lucas_kanade(self):
        cfg = {"flow": {"max_corners": 100, "quality_level": 0.3,
                        "min_distance": 7, "block_size": 7}}
        est = create_flow_estimator("lucas_kanade", cfg)
        assert isinstance(est, LucasKanadeFlow)

    def test_creates_farneback(self):
        est = create_flow_estimator("farneback", {})
        assert isinstance(est, FarnebackFlow)

    def test_unknown_method_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown flow method"):
            create_flow_estimator("unknown_method", {})
