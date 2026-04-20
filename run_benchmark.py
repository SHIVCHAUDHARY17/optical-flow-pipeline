import cv2
import yaml
import time
import logging
import argparse
import numpy as np

from src.flow_estimator import LucasKanadeFlow, FarnebackFlow
from src.visualizer import visualize_sparse_flow, draw_flow_magnitude_overlay
from src.segmenter import MotionSegmenter
from src.experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def benchmark_lucas_kanade(config: dict, tracker: ExperimentTracker):
    """Benchmark LK sparse flow and log to MLflow."""
    fc = config["flow"]
    video_path = config["video"]["path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]

    params = {
        "method": "lucas_kanade",
        "max_corners": fc["max_corners"],
        "quality_level": fc["quality_level"],
        "min_distance": fc["min_distance"],
        "resize": f"{resize_w}x{resize_h}",
    }

    tracker.start_run(run_name="lucas_kanade")
    tracker.log_params(params)

    lk = LucasKanadeFlow(
        max_corners=fc["max_corners"],
        quality_level=fc["quality_level"],
        min_distance=fc["min_distance"],
        block_size=fc["block_size"],
    )

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (resize_w, resize_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = lk.detect_features(prev_gray)

    times = []
    tracked_counts = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= 100:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = time.perf_counter()
        good_prev, good_curr = lk.compute(prev_gray, curr_gray, prev_points)
        elapsed = time.perf_counter() - t

        times.append(elapsed)
        tracked_counts.append(len(good_curr))

        # Log per-frame metrics
        tracker.log_metric("fps", 1.0 / elapsed, step=frame_idx)
        tracker.log_metric("tracked_points", len(good_curr), step=frame_idx)

        if frame_idx % 30 == 0:
            prev_points = lk.detect_features(curr_gray)
        else:
            prev_points = (
                good_curr.reshape(-1, 1, 2) if len(good_curr) > 0 else prev_points
            )

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()

    avg_fps = 1.0 / (sum(times) / len(times))
    avg_points = sum(tracked_counts) / len(tracked_counts)

    summary = {
        "avg_fps": round(avg_fps, 2),
        "avg_inference_ms": round(sum(times) / len(times) * 1000, 2),
        "avg_tracked_points": round(avg_points, 1),
    }

    tracker.end_run(summary)
    logger.info(f"LK benchmark done — {summary}")
    return summary


def benchmark_farneback(config: dict, tracker: ExperimentTracker):
    """Benchmark Farneback dense flow and log to MLflow."""
    video_path = config["video"]["path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]

    params = {
        "method": "farneback",
        "resize": f"{resize_w}x{resize_h}",
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
    }

    tracker.start_run(run_name="farneback")
    tracker.log_params(params)

    fb = FarnebackFlow()

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (resize_w, resize_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    times = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= 100:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = time.perf_counter()
        flow = fb.compute(prev_gray, curr_gray)
        elapsed = time.perf_counter() - t

        times.append(elapsed)
        tracker.log_metric("fps", 1.0 / elapsed, step=frame_idx)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()

    avg_fps = 1.0 / (sum(times) / len(times))

    summary = {
        "avg_fps": round(avg_fps, 2),
        "avg_inference_ms": round(sum(times) / len(times) * 1000, 2),
    }

    tracker.end_run(summary)
    logger.info(f"Farneback benchmark done — {summary}")
    return summary


def benchmark_segmentation(config: dict, tracker: ExperimentTracker, threshold: float):
    """Benchmark segmentation at a given threshold and log to MLflow."""
    video_path = config["video"]["path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]
    seg_cfg = config["segmentation"]

    params = {
        "method": "segmentation",
        "magnitude_threshold": threshold,
        "blur_kernel": seg_cfg["blur_kernel"],
        "morph_kernel": seg_cfg["morph_kernel"],
        "resize": f"{resize_w}x{resize_h}",
    }

    tracker.start_run(run_name=f"segmentation_thresh_{threshold}")
    tracker.log_params(params)

    fb = FarnebackFlow()
    segmenter = MotionSegmenter(
        magnitude_threshold=threshold,
        blur_kernel=seg_cfg["blur_kernel"],
        morph_kernel=seg_cfg["morph_kernel"],
    )

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (resize_w, resize_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    motion_ratios = []
    blob_counts = []
    times = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= 100:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = time.perf_counter()
        flow = fb.compute(prev_gray, curr_gray)
        mask = segmenter.compute_mask(flow)
        elapsed = time.perf_counter() - t

        stats = segmenter.get_moving_stats(mask)
        times.append(elapsed)
        motion_ratios.append(stats["moving_ratio"])
        blob_counts.append(stats["num_moving_blobs"])

        tracker.log_metric("motion_ratio", stats["moving_ratio"], step=frame_idx)
        tracker.log_metric("num_blobs", stats["num_moving_blobs"], step=frame_idx)
        tracker.log_metric("fps", 1.0 / elapsed, step=frame_idx)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()

    summary = {
        "avg_fps": round(1.0 / (sum(times) / len(times)), 2),
        "avg_motion_pct": round(sum(motion_ratios) / len(motion_ratios) * 100, 2),
        "avg_blob_count": round(sum(blob_counts) / len(blob_counts), 1),
    }

    tracker.end_run(summary)
    logger.info(f"Segmentation benchmark done — threshold={threshold} — {summary}")
    return summary


def benchmark_raft(config: dict, tracker: ExperimentTracker):
    """Benchmark RAFT dense flow and log to MLflow."""
    from src.flow_estimator import RAFTFlow

    video_path = config["video"]["path"]
    resize_w = config["video"]["resize_width"]
    resize_h = config["video"]["resize_height"]
    rc = config.get("raft", {})
    device = rc.get("device", "cuda")
    iters = rc.get("iters", 20)

    params = {
        "method": "raft",
        "resize": f"{resize_w}x{resize_h}",
        "iters": iters,
        "device": device,
    }

    tracker.start_run(run_name="raft")
    tracker.log_params(params)

    raft = RAFTFlow(device=device, iters=iters)

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (resize_w, resize_h))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    times = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= 100:
            break

        frame = cv2.resize(frame, (resize_w, resize_h))
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        t = time.perf_counter()
        flow = raft.compute(prev_gray, curr_gray)
        elapsed = time.perf_counter() - t

        times.append(elapsed)
        tracker.log_metric("fps", 1.0 / elapsed, step=frame_idx)

        prev_gray = curr_gray
        frame_idx += 1

    cap.release()

    avg_fps = 1.0 / (sum(times) / len(times))
    summary = {
        "avg_fps": round(avg_fps, 2),
        "avg_inference_ms": round(sum(times) / len(times) * 1000, 2),
    }

    tracker.end_run(summary)
    logger.info(f"RAFT benchmark done — {summary}")
    return summary


def print_table(results: list):
    print("\n" + "=" * 65)
    print(f"{'Run':<35} {'Avg FPS':>10} {'Avg ms':>10}")
    print("=" * 65)
    for r in results:
        ms = r.get("avg_inference_ms", "—")
        print(f"{r['name']:<35} {r['avg_fps']:>10} {str(ms):>10}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    mlflow_cfg = config["mlflow"]

    tracker = ExperimentTracker(
        experiment_name=mlflow_cfg["experiment_name"],
        tracking_uri=mlflow_cfg["tracking_uri"],
    )

    results = []

    # Benchmark LK
    r = benchmark_lucas_kanade(config, tracker)
    r["name"] = "lucas_kanade"
    results.append(r)

    # Benchmark Farneback
    r = benchmark_farneback(config, tracker)
    r["name"] = "farneback"
    results.append(r)

    # Benchmark RAFT
    r = benchmark_raft(config, tracker)
    r["name"] = "raft"
    results.append(r)

    # Benchmark segmentation at multiple thresholds
    for thresh in [0.8, 1.2, 2.0]:
        r = benchmark_segmentation(config, tracker, threshold=thresh)
        r["name"] = f"segmentation_thresh_{thresh}"
        results.append(r)

    print_table(results)

    print(f"\nView results: mlflow ui --backend-store-uri mlruns")
