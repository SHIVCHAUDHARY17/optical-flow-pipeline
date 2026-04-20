import cv2
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LucasKanadeFlow:
    """
    Sparse optical flow using Lucas-Kanade method.

    Tracks a set of 'good features to track' across frames.
    Good features are corners — pixels with strong gradients in
    multiple directions, making them uniquely trackable.

    LK solves: for each feature point, find where it moved
    by assuming brightness constancy and small motion.

    Output: sparse set of (point, motion vector) pairs —
    not every pixel, only tracked keypoints.
    """

    def __init__(
        self,
        max_corners: int = 300,
        quality_level: float = 0.3,
        min_distance: float = 7,
        block_size: int = 7,
    ):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size

        # LK optical flow parameters
        # winSize: search window size per pyramid level
        # maxLevel: pyramid levels (handles large motion)
        # criteria: stop when error < 0.03 or 30 iterations
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.03,
            ),
        )

        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

        logger.info(f"LucasKanadeFlow initialized — max_corners={max_corners}")

    def detect_features(self, frame_gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect good features to track in a grayscale frame.
        Uses Shi-Tomasi corner detection (improved Harris corners).
        Returns array of (x, y) points or None if none found.
        """
        points = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)
        if points is not None:
            logger.debug(f"Detected {len(points)} feature points")
        return points

    def compute(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_points: np.ndarray,
    ) -> tuple:
        """
        Compute sparse optical flow between two grayscale frames.

        Args:
            prev_gray  : previous frame (grayscale)
            curr_gray  : current frame (grayscale)
            prev_points: feature points detected in prev_gray

        Returns:
            good_prev: matched points in previous frame
            good_curr: matched points in current frame
            (pair gives you the motion vectors)
        """
        if prev_points is None or len(prev_points) == 0:
            return np.array([]), np.array([])

        # Track features from prev to curr
        # status[i] = 1 if point i was found, 0 if lost
        curr_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_points, None, **self.lk_params
        )

        if curr_points is None:
            return np.array([]), np.array([])

        # Keep only successfully tracked points
        status = status.flatten()
        good_prev = prev_points[status == 1]
        good_curr = curr_points[status == 1]

        logger.debug(f"Tracked {len(good_curr)}/{len(prev_points)} points")
        return good_prev, good_curr


class FarnebackFlow:
    """
    Dense optical flow using Farneback method.

    Computes a motion vector for EVERY pixel in the frame.
    Uses polynomial expansion to approximate image regions,
    then estimates flow by comparing polynomial coefficients
    between frames.

    Output: flow array of shape (H, W, 2) where
    flow[y, x, 0] = horizontal motion of pixel (x,y)
    flow[y, x, 1] = vertical motion of pixel (x,y)

    Visualized using HSV colorspace:
    - Hue encodes direction (0-360 degrees)
    - Value encodes magnitude (how fast)
    """

    def __init__(self):
        # Standard Farneback parameters
        # pyr_scale: image pyramid scale (0.5 = classic)
        # levels: number of pyramid levels
        # winsize: averaging window size
        # iterations: per pyramid level
        # poly_n: pixel neighborhood for polynomial expansion
        # poly_sigma: Gaussian for polynomial expansion
        self.params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        logger.info("FarnebackFlow initialized")

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two grayscale frames.
        Returns flow array (H, W, 2).
        """
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **self.params)
        logger.debug(f"Dense flow computed — shape: {flow.shape}")
        return flow


class RAFTFlow:
    """
    Dense optical flow using RAFT-Small (Recurrent All-Pairs Field Transforms).

    Loads pretrained RAFT-Small via torch.hub (princeton-vl/RAFT).
    Accepts grayscale frames at any resolution — same interface as
    FarnebackFlow — and internally resizes to (infer_h, infer_w) before
    inference to keep the 4D correlation pyramid within 4GB VRAM.

    Memory breakdown at default 640x360 inference resolution:
      correlation pyramid ~69MB vs ~1,102MB at full 1280x720.

    After inference the flow field is bilinearly upsampled back to the
    original (H, W) and each vector component is rescaled so that pixel
    displacements are expressed in the original frame's coordinate system.

    Output: flow array (H, W, 2) — identical shape and semantics to
    FarnebackFlow, compatible with MotionSegmenter and all visualizers.
    """

    def __init__(
        self,
        device: str = "cuda",
        iters: int = 12,
        infer_width: int = 640,
        infer_height: int = 360,
    ):
        import torch
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

        self.iters = iters
        self.infer_w = infer_width
        self.infer_h = infer_height
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        logger.info(
            f"Loading RAFT-Small from torchvision — "
            f"device={self.device}, inference_res={infer_width}x{infer_height}"
        )
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("RAFTFlow initialized")

    def _to_tensor(self, gray: np.ndarray) -> "torch.Tensor":
        """Grayscale (H, W) → (1, 3, H, W) float32 tensor normalized to [-1, 1]."""
        import torch

        rgb = np.stack([gray, gray, gray], axis=2)                   # (H, W, 3)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # [0, 1]
        t = (t - 0.5) / 0.5                                          # [-1, 1]
        return t.unsqueeze(0).to(self.device)                        # (1, 3, H, W)

    def compute(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """
        Compute dense optical flow between two grayscale frames.
        Returns flow array (H, W, 2) — same interface as FarnebackFlow.
        """
        import torch
        import torch.nn.functional as F

        orig_h, orig_w = prev_gray.shape[:2]

        # Downscale to inference resolution before building correlation pyramid
        prev_small = cv2.resize(prev_gray, (self.infer_w, self.infer_h))
        curr_small = cv2.resize(curr_gray, (self.infer_w, self.infer_h))

        # Pad to multiple of 8 (RAFT requirement; 640x360 already satisfies this)
        pad_h = (8 - self.infer_h % 8) % 8
        pad_w = (8 - self.infer_w % 8) % 8

        img1 = self._to_tensor(prev_small)
        img2 = self._to_tensor(curr_small)

        if pad_h or pad_w:
            img1 = F.pad(img1, [0, pad_w, 0, pad_h])
            img2 = F.pad(img2, [0, pad_w, 0, pad_h])

        with torch.no_grad():
            # torchvision RAFT returns a list of flow predictions (one per update);
            # the last element is the most refined estimate
            flow_predictions = self.model(img1, img2, num_flow_updates=self.iters)

        # Crop padding: (1, 2, infer_h+pad, infer_w+pad) → (1, 2, infer_h, infer_w)
        flow_small = flow_predictions[-1][:, :, : self.infer_h, : self.infer_w]

        # Upsample to original resolution
        flow_orig = F.interpolate(
            flow_small,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )  # (1, 2, orig_h, orig_w)

        # Rescale vector magnitudes: motion was measured in inference-res pixels
        flow_orig[:, 0] *= orig_w / self.infer_w   # x component
        flow_orig[:, 1] *= orig_h / self.infer_h   # y component

        # (1, 2, H, W) → (H, W, 2) numpy — matches FarnebackFlow exactly
        flow_np = flow_orig[0].permute(1, 2, 0).cpu().numpy()
        logger.debug(
            f"RAFT flow — infer={self.infer_w}x{self.infer_h}, "
            f"output={orig_w}x{orig_h}, shape={flow_np.shape}"
        )
        return flow_np


def create_flow_estimator(method: str, config: dict):
    """
    Factory function — returns the right estimator based on config.
    """
    if method == "lucas_kanade":
        fc = config["flow"]
        return LucasKanadeFlow(
            max_corners=fc["max_corners"],
            quality_level=fc["quality_level"],
            min_distance=fc["min_distance"],
            block_size=fc["block_size"],
        )
    elif method == "farneback":
        return FarnebackFlow()
    elif method == "raft":
        rc = config.get("raft", {})
        return RAFTFlow(
            device=rc.get("device", "cuda"),
            iters=rc.get("iters", 20),
            infer_width=rc.get("infer_width", 640),
            infer_height=rc.get("infer_height", 360),
        )
    else:
        raise ValueError(
            f"Unknown flow method: {method}. Choose lucas_kanade, farneback, or raft"
        )
