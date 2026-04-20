import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def visualize_sparse_flow(
    frame: np.ndarray,
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """
    Draw motion vectors on frame for sparse (LK) flow.
    Each tracked point gets a circle and an arrow showing
    where it moved.
    """
    output = frame.copy()

    if len(prev_points) == 0:
        return output

    for prev, curr in zip(prev_points, curr_points):
        px, py = int(prev[0][0]), int(prev[0][1])
        cx, cy = int(curr[0][0]), int(curr[0][1])

        # Scale the vector up so tiny motions are visible
        scale = 5
        ex = int(px + (cx - px) * scale)
        ey = int(py + (cy - py) * scale)

        cv2.arrowedLine(output, (px, py), (ex, ey), color, 2, tipLength=0.4)
        cv2.circle(output, (cx, cy), 4, color, -1)

    return output


def visualize_dense_flow(flow: np.ndarray) -> np.ndarray:
    """
    Convert dense flow array to HSV colorwheel visualization.
    Direction encodes Hue, Magnitude encodes Value.
    Static regions appear black, fast regions appear bright and colored.
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255

    # Boost magnitude so small motions are visible
    # Clip at 10 pixels max motion then scale to 0-255
    # Was clipping at 10 — change to 5 to match your video's actual range
    magnitude_clipped = np.clip(magnitude, 0, 5)
    hsv[..., 2] = (magnitude_clipped / 5 * 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def draw_flow_magnitude_overlay(
    frame: np.ndarray,
    flow: np.ndarray,
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Blend dense flow colorwheel on top of original frame.
    """
    flow_vis = visualize_dense_flow(flow)
    blended = cv2.addWeighted(frame, 1 - alpha, flow_vis, alpha, 0)
    return blended
