import numpy as np
import pytest

from src.visualizer import (
    draw_flow_magnitude_overlay,
    visualize_dense_flow,
    visualize_sparse_flow,
)

H, W = 120, 160


@pytest.fixture
def bgr_frame():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (H, W, 3), dtype=np.uint8)


@pytest.fixture
def zero_flow():
    return np.zeros((H, W, 2), dtype=np.float32)


@pytest.fixture
def motion_flow():
    """Flow with magnitude ~3.6 px/frame (within the 0-5 visualizer clamp)."""
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[:, :, 0] = 3.0   # dx
    flow[:, :, 1] = 2.0   # dy
    return flow


# ---------------------------------------------------------------------------
# visualize_sparse_flow
# ---------------------------------------------------------------------------

class TestVisualizeSparseFlow:
    def test_empty_points_returns_same_shape(self, bgr_frame):
        out = visualize_sparse_flow(bgr_frame, np.array([]), np.array([]))
        assert out.shape == bgr_frame.shape

    def test_empty_points_output_equals_input(self, bgr_frame):
        out = visualize_sparse_flow(bgr_frame, np.array([]), np.array([]))
        np.testing.assert_array_equal(out, bgr_frame)

    def test_with_points_output_shape(self, bgr_frame):
        prev = np.array([[[10.0, 20.0]], [[50.0, 60.0]], [[100.0, 80.0]]], dtype=np.float32)
        curr = np.array([[[13.0, 22.0]], [[53.0, 63.0]], [[103.0, 82.0]]], dtype=np.float32)
        out = visualize_sparse_flow(bgr_frame, prev, curr)
        assert out.shape == bgr_frame.shape

    def test_with_points_output_dtype(self, bgr_frame):
        prev = np.array([[[10.0, 20.0]], [[50.0, 60.0]]], dtype=np.float32)
        curr = np.array([[[13.0, 22.0]], [[53.0, 63.0]]], dtype=np.float32)
        out = visualize_sparse_flow(bgr_frame, prev, curr)
        assert out.dtype == np.uint8

    def test_with_points_does_not_modify_input(self, bgr_frame):
        original = bgr_frame.copy()
        prev = np.array([[[10.0, 20.0]]], dtype=np.float32)
        curr = np.array([[[15.0, 25.0]]], dtype=np.float32)
        visualize_sparse_flow(bgr_frame, prev, curr)
        np.testing.assert_array_equal(bgr_frame, original)


# ---------------------------------------------------------------------------
# visualize_dense_flow
# ---------------------------------------------------------------------------

class TestVisualizeDenseFlow:
    def test_output_shape(self, zero_flow):
        vis = visualize_dense_flow(zero_flow)
        assert vis.shape == (H, W, 3)

    def test_output_dtype(self, zero_flow):
        vis = visualize_dense_flow(zero_flow)
        assert vis.dtype == np.uint8

    def test_zero_flow_is_black(self, zero_flow):
        """Zero magnitude → value channel = 0 → black in HSV → black in BGR."""
        vis = visualize_dense_flow(zero_flow)
        assert vis.max() == 0

    def test_motion_flow_is_not_black(self, motion_flow):
        vis = visualize_dense_flow(motion_flow)
        assert vis.max() > 0

    def test_pixel_values_in_valid_range(self, motion_flow):
        vis = visualize_dense_flow(motion_flow)
        assert vis.min() >= 0
        assert vis.max() <= 255


# ---------------------------------------------------------------------------
# draw_flow_magnitude_overlay
# ---------------------------------------------------------------------------

class TestDrawFlowMagnitudeOverlay:
    def test_output_shape(self, bgr_frame, zero_flow):
        out = draw_flow_magnitude_overlay(bgr_frame, zero_flow)
        assert out.shape == bgr_frame.shape

    def test_output_dtype(self, bgr_frame, zero_flow):
        out = draw_flow_magnitude_overlay(bgr_frame, zero_flow)
        assert out.dtype == np.uint8

    def test_alpha_zero_returns_original(self, bgr_frame, motion_flow):
        """alpha=0 means 100% original frame, 0% flow overlay."""
        out = draw_flow_magnitude_overlay(bgr_frame, motion_flow, alpha=0.0)
        np.testing.assert_array_equal(out, bgr_frame)

    def test_alpha_one_returns_flow_vis(self, bgr_frame, motion_flow):
        """alpha=1 means 100% flow visualization, 0% original frame."""
        out = draw_flow_magnitude_overlay(bgr_frame, motion_flow, alpha=1.0)
        expected = visualize_dense_flow(motion_flow)
        np.testing.assert_array_equal(out, expected)

    def test_does_not_modify_input_frame(self, bgr_frame, motion_flow):
        original = bgr_frame.copy()
        draw_flow_magnitude_overlay(bgr_frame, motion_flow)
        np.testing.assert_array_equal(bgr_frame, original)
