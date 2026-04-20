import numpy as np
import pytest

from src.segmenter import MotionSegmenter

H, W = 120, 160


@pytest.fixture
def zero_flow():
    return np.zeros((H, W, 2), dtype=np.float32)


@pytest.fixture
def high_flow():
    """Uniform 10 px/frame horizontal motion — well above any reasonable threshold."""
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[:, :, 0] = 10.0
    return flow


@pytest.fixture
def bgr_frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


class TestComputeMask:
    def test_output_shape(self, zero_flow):
        seg = MotionSegmenter()
        mask = seg.compute_mask(zero_flow)
        assert mask.shape == (H, W)

    def test_output_dtype(self, zero_flow):
        seg = MotionSegmenter()
        mask = seg.compute_mask(zero_flow)
        assert mask.dtype == np.uint8

    def test_output_values_binary(self, high_flow):
        seg = MotionSegmenter(magnitude_threshold=0.5)
        mask = seg.compute_mask(high_flow)
        assert set(np.unique(mask)).issubset({0, 255})

    def test_zero_flow_produces_empty_mask(self, zero_flow):
        seg = MotionSegmenter(magnitude_threshold=0.5)
        mask = seg.compute_mask(zero_flow)
        assert np.count_nonzero(mask) == 0

    def test_high_flow_produces_nonempty_mask(self, high_flow):
        seg = MotionSegmenter(magnitude_threshold=0.5)
        mask = seg.compute_mask(high_flow)
        assert np.count_nonzero(mask) > 0

    def test_higher_threshold_reduces_moving_pixels(self, high_flow):
        seg_low = MotionSegmenter(magnitude_threshold=1.0)
        seg_high = MotionSegmenter(magnitude_threshold=8.0)
        moving_low = np.count_nonzero(seg_low.compute_mask(high_flow))
        moving_high = np.count_nonzero(seg_high.compute_mask(high_flow))
        assert moving_low >= moving_high


class TestApplyMask:
    def test_output_shape_matches_frame(self, zero_flow, bgr_frame):
        seg = MotionSegmenter()
        mask = seg.compute_mask(zero_flow)
        out = seg.apply_mask(bgr_frame, mask)
        assert out.shape == bgr_frame.shape

    def test_output_dtype(self, zero_flow, bgr_frame):
        seg = MotionSegmenter()
        mask = seg.compute_mask(zero_flow)
        out = seg.apply_mask(bgr_frame, mask)
        assert out.dtype == np.uint8

    def test_zero_mask_frame_unchanged(self, zero_flow, bgr_frame):
        """All-zero mask → no overlay applied → output equals input."""
        seg = MotionSegmenter(magnitude_threshold=0.5)
        mask = seg.compute_mask(zero_flow)
        out = seg.apply_mask(bgr_frame, mask)
        np.testing.assert_array_equal(out, bgr_frame)


class TestGetMovingStats:
    def test_required_keys_present(self, zero_flow):
        seg = MotionSegmenter()
        stats = seg.get_moving_stats(seg.compute_mask(zero_flow))
        assert {"moving_pixels", "moving_ratio", "num_moving_blobs"} <= stats.keys()

    def test_zero_mask_gives_zero_stats(self, zero_flow):
        seg = MotionSegmenter(magnitude_threshold=0.5)
        stats = seg.get_moving_stats(seg.compute_mask(zero_flow))
        assert stats["moving_pixels"] == 0
        assert stats["moving_ratio"] == 0.0
        assert stats["num_moving_blobs"] == 0

    def test_ratio_bounded(self, high_flow):
        seg = MotionSegmenter(magnitude_threshold=0.5)
        stats = seg.get_moving_stats(seg.compute_mask(high_flow))
        assert 0.0 <= stats["moving_ratio"] <= 1.0

    def test_moving_pixels_consistent_with_ratio(self, high_flow):
        seg = MotionSegmenter(magnitude_threshold=0.5)
        mask = seg.compute_mask(high_flow)
        stats = seg.get_moving_stats(mask)
        expected_ratio = stats["moving_pixels"] / mask.size
        assert abs(stats["moving_ratio"] - expected_ratio) < 1e-4
