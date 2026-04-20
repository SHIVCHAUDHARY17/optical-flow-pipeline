import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MotionSegmenter:
    """
    Segments moving objects from static background using
    optical flow magnitude thresholding.

    Core idea:
    - Static pixels have flow magnitude ≈ 0
    - Moving pixels have flow magnitude > threshold
    - No training required — pure geometric reasoning

    This is the fundamental principle behind background
    subtraction in surveillance and ADAS systems.
    """

    def __init__(
        self,
        magnitude_threshold: float = 2.0,
        blur_kernel: int = 5,
        morph_kernel: int = 7,
    ):
        """
        magnitude_threshold: min pixels/frame to count as moving
        blur_kernel: Gaussian blur before thresholding (removes noise)
        morph_kernel: morphological operations kernel size
                      (fills holes, removes tiny blobs)
        """
        self.magnitude_threshold = magnitude_threshold
        self.blur_kernel = blur_kernel
        self.morph_kernel = morph_kernel

        # Morphological kernel for cleaning up the mask
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)
        )

        logger.info(
            f"MotionSegmenter initialized — "
            f"threshold={magnitude_threshold}, "
            f"morph_kernel={morph_kernel}"
        )

    def compute_mask(self, flow: np.ndarray) -> np.ndarray:
        """
        Compute binary motion mask from dense flow field.

        Args:
            flow: dense flow array (H, W, 2) from Farneback

        Returns:
            mask: binary uint8 array (H, W)
                  255 = moving, 0 = static
        """
        # Step 1: compute per-pixel flow magnitude
        # magnitude[y,x] = sqrt(dx^2 + dy^2) in pixels/frame
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)

        # Step 2: Gaussian blur to reduce noise before thresholding
        # Small isolated moving pixels are usually noise not objects
        magnitude_blurred = cv2.GaussianBlur(
            magnitude, (self.blur_kernel, self.blur_kernel), 0
        )

        # Step 3: threshold — pixels above threshold are moving
        _, mask = cv2.threshold(
            magnitude_blurred, self.magnitude_threshold, 255, cv2.THRESH_BINARY
        )
        mask = mask.astype(np.uint8)

        # Step 4: morphological operations to clean up mask
        # Opening: removes small noise blobs (erode then dilate)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        # Closing: fills holes inside moving objects (dilate then erode)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        moving_pixels = np.count_nonzero(mask)
        logger.debug(f"Moving pixels: {moving_pixels}")

        return mask

    def apply_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        color: tuple = (0, 255, 255),
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay motion mask on original frame.

        Moving regions are tinted with color.
        Static regions remain unchanged.

        Args:
            frame : original BGR frame
            mask  : binary motion mask (255=moving, 0=static)
            color : BGR tint color for moving regions
            alpha : blend factor (0=no tint, 1=full color)

        Returns:
            Annotated frame with moving regions highlighted
        """
        output = frame.copy()

        # Create colored overlay for moving regions
        overlay = np.zeros_like(frame)
        overlay[mask == 255] = color

        # Blend overlay with original only where mask is active
        moving_region = mask == 255
        output[moving_region] = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)[
            moving_region
        ]

        # Draw mask contours for clean edge visualization
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Only draw contours around significant moving blobs
        # Filter out tiny blobs — likely noise
        min_area = 500
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(output, [contour], -1, color, 2)

        return output

    def get_moving_stats(self, mask: np.ndarray) -> dict:
        """
        Compute statistics about detected motion.
        Useful for MLflow logging and analytics.
        """
        total_pixels = mask.size
        moving_pixels = int(np.count_nonzero(mask))
        moving_ratio = moving_pixels / total_pixels

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_blobs = [c for c in contours if cv2.contourArea(c) > 500]

        return {
            "moving_pixels": moving_pixels,
            "moving_ratio": round(moving_ratio, 4),
            "num_moving_blobs": len(significant_blobs),
        }
