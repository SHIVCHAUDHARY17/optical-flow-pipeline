import cv2
import yaml
import argparse
import logging
import os
import numpy as np

from src.flow_estimator import create_flow_estimator, FarnebackFlow, LucasKanadeFlow
from src.visualizer import (
    visualize_sparse_flow,
    visualize_dense_flow,
    draw_flow_magnitude_overlay,
)

# Structured logging — no print statements anywhere
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_flow(config: dict):
    """
    Single method flow pipeline.
    lucas_kanade → sparse flow with scaled motion vectors
    farneback    → dense flow with HSV colorwheel overlay
    """
    method = config["flow"]["method"]
    video_path = config["video"]["path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]
    out_path = config["output"]["video_out"]

    logger.info(f"Starting flow pipeline — method: {method}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {total} frames @ {fps:.1f} FPS")

    estimator = create_flow_estimator(method, config)

    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Could not read first frame")
        return

    prev_frame = cv2.resize(prev_frame, (resize_w, resize_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    if method == "lucas_kanade":
        prev_points = estimator.detect_features(prev_gray)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if method == "lucas_kanade":
            good_prev, good_curr = estimator.compute(prev_gray, curr_gray, prev_points)
            output = visualize_sparse_flow(frame, good_prev, good_curr)
            if frame_idx % 30 == 0:
                prev_points = estimator.detect_features(curr_gray)
            else:
                prev_points = (
                    good_curr.reshape(-1, 1, 2) if len(good_curr) > 0 else prev_points
                )

        elif method in ("farneback", "raft"):
            flow = estimator.compute(prev_gray, curr_gray)
            output = draw_flow_magnitude_overlay(frame, flow, alpha=0.7)

        if writer is None:
            h, w = output.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(output)
        prev_gray = curr_gray
        frame_idx += 1

        if frame_idx % 50 == 0:
            logger.info(f"Processed {frame_idx}/{total} frames")

    cap.release()
    if writer:
        writer.release()
    logger.info(f"Done. Output saved to: {out_path}")


def run_flow_comparison(config: dict):
    """
    Runs LK and Farneback simultaneously on the same video.
    Left  → Lucas-Kanade sparse flow
    Right → Farneback dense HSV flow
    """
    video_path = config["video"]["path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]
    out_path = config["output"]["video_out"].replace(".mp4", "_comparison.mp4")

    logger.info("Running side-by-side comparison: LK vs Farneback")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fc = config["flow"]
    lk = LucasKanadeFlow(
        max_corners=fc["max_corners"],
        quality_level=fc["quality_level"],
        min_distance=fc["min_distance"],
        block_size=fc["block_size"],
    )
    fb = FarnebackFlow()

    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Could not read first frame")
        return

    prev_frame = cv2.resize(prev_frame, (resize_w, resize_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = lk.detect_features(prev_gray)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        good_prev, good_curr = lk.compute(prev_gray, curr_gray, prev_points)
        lk_out = visualize_sparse_flow(frame, good_prev, good_curr)
        if frame_idx % 30 == 0:
            prev_points = lk.detect_features(curr_gray)
        else:
            prev_points = (
                good_curr.reshape(-1, 1, 2) if len(good_curr) > 0 else prev_points
            )

        flow = fb.compute(prev_gray, curr_gray)
        fb_out = draw_flow_magnitude_overlay(frame, flow, alpha=0.7)

        cv2.putText(
            lk_out,
            "Lucas-Kanade (Sparse)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            fb_out,
            "Farneback (Dense HSV)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        combined = np.hstack([lk_out, fb_out])

        if writer is None:
            h, w = combined.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(combined)
        prev_gray = curr_gray
        frame_idx += 1

        if frame_idx % 50 == 0:
            logger.info(f"Processed {frame_idx}/{total} frames")

    cap.release()
    if writer:
        writer.release()
    logger.info(f"Comparison video saved to: {out_path}")


def run_segmentation(config: dict):
    """
    Farneback dense flow + motion segmentation.
    Thresholds flow magnitude to separate moving vs static pixels.
    Applies morphological cleanup and overlays cyan tint on moving regions.
    Logs per-frame blob count and motion percentage.
    """
    from src.segmenter import MotionSegmenter

    video_path = config["video"]["path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]
    out_path = config["output"]["segmentation_out"]
    seg_cfg = config["segmentation"]

    logger.info("Running motion segmentation pipeline")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fb = FarnebackFlow()
    segmenter = MotionSegmenter(
        magnitude_threshold=seg_cfg["magnitude_threshold"],
        blur_kernel=seg_cfg["blur_kernel"],
        morph_kernel=seg_cfg["morph_kernel"],
    )

    os.makedirs("outputs", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Could not read first frame")
        return

    prev_frame = cv2.resize(prev_frame, (resize_w, resize_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_idx = 0
    total_moving_ratio = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = fb.compute(prev_gray, curr_gray)
        mask = segmenter.compute_mask(flow)
        stats = segmenter.get_moving_stats(mask)
        total_moving_ratio += stats["moving_ratio"]
        output = segmenter.apply_mask(frame, mask)

        cv2.putText(
            output,
            f"Moving blobs: {stats['num_moving_blobs']}  "
            f"Motion: {stats['moving_ratio'] * 100:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        if writer is None:
            h, w = output.shape[:2]
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

        writer.write(output)
        prev_gray = curr_gray
        frame_idx += 1

        if frame_idx % 50 == 0:
            avg_motion = total_moving_ratio / frame_idx * 100
            logger.info(f"Frame {frame_idx}/{total} — avg motion: {avg_motion:.1f}%")

    cap.release()
    if writer:
        writer.release()

    avg_motion = total_moving_ratio / max(frame_idx, 1) * 100
    logger.info(f"Done. Avg scene motion: {avg_motion:.1f}%")
    logger.info(f"Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--mode",
        choices=["single", "comparison", "segmentation"],
        default="single",
        help=(
            "single      : run one method (set in config)\n"
            "comparison  : LK vs Farneback side by side\n"
            "segmentation: motion mask with blob stats"
        ),
    )
    parser.add_argument(
        "--method",
        choices=["lucas_kanade", "farneback", "raft"],
        default=None,
        help="Override flow.method from config (single mode only)",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    if args.method:
        config["flow"]["method"] = args.method

    if args.mode == "comparison":
        run_flow_comparison(config)
    elif args.mode == "segmentation":
        run_segmentation(config)
    else:
        run_flow(config)
