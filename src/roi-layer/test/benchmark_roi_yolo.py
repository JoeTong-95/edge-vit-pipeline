#!/usr/bin/env python3
"""
benchmark_roi_yolo.py

Headless direct-vs-ROI-cropped YOLO benchmark.

Comparison modes:
- direct video input -> YOLO
- ROI calibration pass -> fixed ROI crop -> cropped input -> YOLO

This script:
- runs sample1.mp4 through sample4.mp4
- evaluates 30 seconds per trial by default
- logs per-run and per-frame metrics into SQLite
- creates a styled summary chart

It does not render windows, videos, or GIFs while benchmarking.

Run from project root:
    python src/roi-layer/test/benchmark_roi_yolo.py
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_ROI_DIR = _THIS_DIR.parent
_SRC_DIR = _ROI_DIR.parent
_CONFIG_DIR = _SRC_DIR / "configuration-layer"
_INPUT_DIR = _SRC_DIR / "input-layer"
_YOLO_DIR = _SRC_DIR / "yolo-layer"
_REPO_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_ROI_DIR))
sys.path.insert(0, str(_CONFIG_DIR))
sys.path.insert(0, str(_INPUT_DIR))
sys.path.insert(0, str(_YOLO_DIR))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from config_node import get_config_value, load_config, validate_config
from detector import build_yolo_layer_package, filter_yolo_detections, initialize_yolo_layer, run_yolo_detection
from input_layer import InputLayer
from roi_layer import crop_frame_to_roi, initialize_roi_layer, update_roi_state

BACKGROUND = "#171717"
PANEL = "#242424"
GRID = "#4a4a4a"
TEXT = "#f3f3f3"

DIRECT_COLOR = "#ffb347"
ROI_CROP_COLOR = "#ff6a00"
SCATTER_ALPHA = 0.20

DEFAULT_OUTPUT_DIR = Path(r"E:\OneDrive\desktop\roi-optimization")
DEFAULT_MAX_SECONDS = 30.0


@dataclass(frozen=True)
class TrialConfig:
    sample_name: str
    video_path: Path
    mode: str


def load_runtime_settings() -> dict[str, object]:
    config_path = _CONFIG_DIR / "config.yaml"
    config = load_config(config_path)
    validate_config(config)
    return {
        "frame_resolution": tuple(get_config_value(config, "config_frame_resolution")),
        "model": get_config_value(config, "config_yolo_model"),
        "conf": get_config_value(config, "config_yolo_confidence_threshold"),
        "device": get_config_value(config, "config_device"),
        "roi_threshold": get_config_value(config, "config_roi_vehicle_count_threshold"),
    }


def probe_video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not probe video source: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def build_trials() -> list[TrialConfig]:
    trials: list[TrialConfig] = []
    for sample_index in range(1, 5):
        sample_name = f"sample{sample_index}"
        video_path = _REPO_ROOT / "data" / f"{sample_name}.mp4"
        trials.append(TrialConfig(sample_name=sample_name, video_path=video_path, mode="direct"))
        trials.append(TrialConfig(sample_name=sample_name, video_path=video_path, mode="roi_crop_input"))
    return trials


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def initialize_db(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(db_path), timeout=30.0)
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS roi_eval_runs (
            run_id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            sample_name TEXT NOT NULL,
            video_path TEXT NOT NULL,
            mode TEXT NOT NULL,
            roi_threshold INTEGER NOT NULL,
            model_name TEXT NOT NULL,
            device_mode TEXT NOT NULL,
            confidence REAL NOT NULL,
            frame_width INTEGER NOT NULL,
            frame_height INTEGER NOT NULL,
            source_fps REAL NOT NULL,
            max_seconds_requested REAL NOT NULL,
            calibration_frames INTEGER,
            calibration_elapsed_seconds REAL,
            roi_locked INTEGER,
            roi_bounds TEXT,
            processed_frames INTEGER,
            elapsed_seconds REAL,
            average_fps REAL,
            average_infer_fps REAL,
            average_infer_ms REAL,
            mean_detection_count REAL,
            mean_roi_area_ratio REAL,
            status TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS roi_eval_frames (
            frame_record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            frame_id INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            fps_actual REAL NOT NULL,
            infer_ms REAL NOT NULL,
            infer_fps REAL NOT NULL,
            detection_count INTEGER NOT NULL,
            roi_locked INTEGER NOT NULL,
            roi_area_ratio REAL NOT NULL,
            detection_source TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES roi_eval_runs(run_id)
        )
        """
    )
    connection.commit()
    return connection


def insert_run_row(
    connection: sqlite3.Connection,
    trial: TrialConfig,
    settings: dict[str, object],
    frame_resolution: tuple[int, int],
    source_fps: float,
    max_seconds: float,
) -> str:
    run_id = str(uuid.uuid4())
    connection.execute(
        """
        INSERT INTO roi_eval_runs (
            run_id, created_at_utc, sample_name, video_path, mode,
            roi_threshold, model_name, device_mode, confidence, frame_width,
            frame_height, source_fps, max_seconds_requested, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            datetime.now(timezone.utc).isoformat(),
            trial.sample_name,
            str(trial.video_path),
            trial.mode,
            int(settings["roi_threshold"]),
            str(settings["model"]),
            str(settings["device"]),
            float(settings["conf"]),
            int(frame_resolution[0]),
            int(frame_resolution[1]),
            float(source_fps),
            float(max_seconds),
            "running",
        ),
    )
    connection.commit()
    return run_id


def insert_frame_row(
    connection: sqlite3.Connection,
    run_id: str,
    frame_id: int,
    elapsed_seconds: float,
    fps_actual: float,
    infer_ms: float,
    infer_fps: float,
    detection_count: int,
    roi_locked: bool,
    roi_area_ratio: float,
    detection_source: str,
) -> None:
    connection.execute(
        """
        INSERT INTO roi_eval_frames (
            run_id, frame_id, elapsed_seconds, fps_actual, infer_ms, infer_fps,
            detection_count, roi_locked, roi_area_ratio, detection_source
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            int(frame_id),
            float(elapsed_seconds),
            float(fps_actual),
            float(infer_ms),
            float(infer_fps),
            int(detection_count),
            int(roi_locked),
            float(roi_area_ratio),
            str(detection_source),
        ),
    )


def finalize_run_row(
    connection: sqlite3.Connection,
    run_id: str,
    *,
    calibration_frames: int,
    calibration_elapsed_seconds: float,
    roi_locked: bool,
    roi_bounds: tuple[int, int, int, int] | None,
    processed_frames: int,
    elapsed_seconds: float,
    average_fps: float,
    average_infer_fps: float,
    average_infer_ms: float,
    mean_detection_count: float,
    mean_roi_area_ratio: float,
    status: str,
) -> None:
    connection.execute(
        """
        UPDATE roi_eval_runs
        SET calibration_frames = ?,
            calibration_elapsed_seconds = ?,
            roi_locked = ?,
            roi_bounds = ?,
            processed_frames = ?,
            elapsed_seconds = ?,
            average_fps = ?,
            average_infer_fps = ?,
            average_infer_ms = ?,
            mean_detection_count = ?,
            mean_roi_area_ratio = ?,
            status = ?
        WHERE run_id = ?
        """,
        (
            int(calibration_frames),
            float(calibration_elapsed_seconds),
            int(roi_locked),
            str(tuple(int(v) for v in roi_bounds)) if roi_bounds is not None else None,
            int(processed_frames),
            float(elapsed_seconds),
            float(average_fps),
            float(average_infer_fps),
            float(average_infer_ms),
            float(mean_detection_count),
            float(mean_roi_area_ratio),
            status,
            run_id,
        ),
    )
    connection.commit()


def package_to_dict(package) -> dict[str, object]:
    return {
        "input_layer_frame_id": package.input_layer_frame_id,
        "input_layer_timestamp": package.input_layer_timestamp,
        "input_layer_image": package.input_layer_image,
        "input_layer_source_type": package.input_layer_source_type,
        "input_layer_resolution": package.input_layer_resolution,
        "yolo_force_native_imgsz": True,
    }


def calibrate_roi_bounds(video_path: Path, frame_resolution: tuple[int, int], roi_threshold: int) -> tuple[tuple[int, int, int, int] | None, int, float]:
    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="video",
        config_frame_resolution=frame_resolution,
        config_input_path=str(video_path),
    )
    initialize_roi_layer(config_roi_enabled=True, config_roi_vehicle_count_threshold=int(roi_threshold))

    calibration_frames = 0
    calibration_start = time.perf_counter()
    locked_bounds: tuple[int, int, int, int] | None = None

    try:
        while True:
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            input_package = input_layer.build_input_layer_package(raw_frame)
            input_pkg = package_to_dict(input_package)
            raw_full = run_yolo_detection(input_pkg)
            filtered_full = filter_yolo_detections(raw_full)
            full_yolo_pkg = build_yolo_layer_package(input_package.input_layer_frame_id, filtered_full)
            roi_state = update_roi_state(input_package, full_yolo_pkg["yolo_layer_detections"])
            calibration_frames += 1

            if bool(roi_state["roi_layer_locked"]):
                locked_bounds = roi_state["roi_layer_bounds"]
                break
    finally:
        input_layer.close_input_layer()

    return locked_bounds, calibration_frames, (time.perf_counter() - calibration_start)


def load_plot_data(db_path: Path):
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    runs = connection.execute(
        """
        SELECT run_id, sample_name, mode, average_fps, average_infer_fps, status
        FROM roi_eval_runs
        WHERE status = 'completed'
        ORDER BY sample_name, mode
        """
    ).fetchall()
    frames = connection.execute(
        """
        SELECT run_id, fps_actual, infer_fps
        FROM roi_eval_frames
        ORDER BY frame_record_id
        """
    ).fetchall()
    connection.close()
    return runs, frames


def build_legend_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DIRECT_COLOR, markeredgecolor="none", markersize=8, label="Direct video input"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ROI_CROP_COLOR, markeredgecolor="none", markersize=8, label="ROI crop as YOLO input"),
        Line2D([0], [0], marker="o", color=TEXT, markerfacecolor=TEXT, markeredgecolor=TEXT, linewidth=0, markersize=5, alpha=0.35, label="Per-frame scatter"),
        Line2D([0], [0], color=TEXT, linewidth=2.4, marker="o", markerfacecolor=TEXT, markeredgecolor=TEXT, markersize=5, label="Run mean"),
    ]


def plot_metric(ax, runs, frames_by_run, metric_key: str, mean_key: str, title: str, ylabel: str) -> None:
    x_positions = np.arange(len(runs))
    labels: list[str] = []
    for index, run in enumerate(runs):
        labels.append(f"{run['sample_name']}\n{'ROI crop' if run['mode'] == 'roi_crop_input' else 'Direct'}")
        values = np.array(frames_by_run[run["run_id"]][metric_key], dtype=float)
        if values.size == 0:
            continue
        color = ROI_CROP_COLOR if run["mode"] == "roi_crop_input" else DIRECT_COLOR
        jitter = np.random.uniform(-0.18, 0.18, size=values.size)
        ax.scatter(
            np.full(values.shape, index, dtype=float) + jitter,
            values,
            s=7,
            alpha=SCATTER_ALPHA,
            color=color,
            edgecolors="none",
        )
        mean_value = float(run[mean_key])
        ax.hlines(mean_value, index - 0.28, index + 0.28, colors=color, linewidth=2.4)
        ax.scatter(index, mean_value, s=80, color=color, edgecolors=TEXT, linewidths=0.8, zorder=5)
        ax.text(index, mean_value, f" {mean_value:.1f}", color=TEXT, fontsize=8, va="center", ha="left")

    ax.set_title(title, color=TEXT, fontsize=16, pad=12)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, color=TEXT, fontsize=10)
    ax.tick_params(axis="y", colors=TEXT)
    ax.grid(True, axis="y", color=GRID, alpha=0.35, linewidth=0.8)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def create_summary_plot(db_path: Path, output_path: Path) -> None:
    runs, frames = load_plot_data(db_path)
    if not runs:
        raise RuntimeError("No completed ROI benchmark runs found in SQLite output.")

    frames_by_run = {run["run_id"]: {"fps_actual": [], "infer_fps": []} for run in runs}
    for frame_row in frames:
        run_id = frame_row["run_id"]
        if run_id not in frames_by_run:
            continue
        frames_by_run[run_id]["fps_actual"].append(frame_row["fps_actual"])
        frames_by_run[run_id]["infer_fps"].append(frame_row["infer_fps"])

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)

    plot_metric(axes[0], runs, frames_by_run, "fps_actual", "average_fps", "End-to-End FPS by Input Mode", "FPS")
    plot_metric(axes[1], runs, frames_by_run, "infer_fps", "average_infer_fps", "Inference FPS by Input Mode", "Infer FPS")

    fig.suptitle(f"ROI Optimization Summary\n{db_path.name}", color=TEXT, fontsize=18, y=0.98)
    legend = fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=True,
        fontsize=10,
        title="Legend",
        title_fontsize=11,
        borderpad=0.8,
        labelspacing=0.9,
        handlelength=1.8,
        handletextpad=0.7,
    )
    legend.get_frame().set_facecolor(PANEL)
    legend.get_frame().set_edgecolor(GRID)
    legend.get_frame().set_alpha(0.95)
    legend.get_title().set_color(TEXT)
    for text in legend.get_texts():
        text.set_color(TEXT)

    plt.tight_layout(rect=[0, 0.04, 1, 0.86])
    fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_trial(
    trial: TrialConfig,
    *,
    settings: dict[str, object],
    connection: sqlite3.Connection,
    max_seconds: float,
) -> None:
    if not trial.video_path.is_file():
        raise FileNotFoundError(f"Sample video not found: {trial.video_path}")

    frame_resolution = tuple(settings["frame_resolution"])
    source_fps = probe_video_fps(trial.video_path)
    run_id = insert_run_row(connection, trial, settings, frame_resolution, source_fps, max_seconds)

    calibration_frames = 0
    calibration_elapsed_seconds = 0.0
    roi_locked = False
    roi_bounds: tuple[int, int, int, int] | None = None

    if trial.mode == "roi_crop_input":
        roi_bounds, calibration_frames, calibration_elapsed_seconds = calibrate_roi_bounds(
            trial.video_path,
            frame_resolution,
            int(settings["roi_threshold"]),
        )
        roi_locked = roi_bounds is not None

    input_layer = InputLayer()
    input_layer.initialize_input_layer(
        config_input_source="video",
        config_frame_resolution=frame_resolution,
        config_input_path=str(trial.video_path),
    )

    total_frames = 0
    fps_values: list[float] = []
    infer_ms_values: list[float] = []
    infer_fps_values: list[float] = []
    detection_counts: list[int] = []
    roi_area_ratios: list[float] = []
    status = "completed"
    start_time = time.perf_counter()

    try:
        while True:
            raw_frame = input_layer.read_next_frame()
            if raw_frame is None:
                break

            loop_start = time.perf_counter()
            input_package = input_layer.build_input_layer_package(raw_frame)
            input_pkg = package_to_dict(input_package)

            if trial.mode == "roi_crop_input":
                if roi_bounds is None:
                    break
                roi_frame = crop_frame_to_roi(input_package.input_layer_image, roi_bounds)
                roi_pkg = {
                    "roi_layer_frame_id": input_package.input_layer_frame_id,
                    "roi_layer_timestamp": input_package.input_layer_timestamp,
                    "roi_layer_image": roi_frame,
                    "roi_layer_bounds": tuple(int(v) for v in roi_bounds),
                    "roi_layer_enabled": True,
                    "roi_layer_locked": True,
                    "yolo_force_native_imgsz": True,
                }
                infer_start = time.perf_counter()
                raw_dets = run_yolo_detection(roi_pkg)
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                filtered_dets = filter_yolo_detections(raw_dets)
                detection_count = len(filtered_dets)
                detection_source = "fixed_roi_crop"
                x1, y1, x2, y2 = roi_bounds
                roi_area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_resolution[0] * frame_resolution[1])
            else:
                infer_start = time.perf_counter()
                raw_dets = run_yolo_detection(input_pkg)
                infer_ms = (time.perf_counter() - infer_start) * 1000.0
                filtered_dets = filter_yolo_detections(raw_dets)
                detection_count = len(filtered_dets)
                detection_source = "direct_full_frame"
                roi_area_ratio = 1.0

            elapsed_seconds = time.perf_counter() - start_time
            total_frames += 1
            loop_elapsed = time.perf_counter() - loop_start
            fps_actual = 1.0 / loop_elapsed if loop_elapsed > 0 else 0.0
            infer_fps = 1000.0 / infer_ms if infer_ms > 0 else 0.0

            fps_values.append(fps_actual)
            infer_ms_values.append(infer_ms)
            infer_fps_values.append(infer_fps)
            detection_counts.append(detection_count)
            roi_area_ratios.append(roi_area_ratio)

            insert_frame_row(
                connection,
                run_id,
                input_package.input_layer_frame_id,
                elapsed_seconds,
                fps_actual,
                infer_ms,
                infer_fps,
                detection_count,
                roi_locked,
                roi_area_ratio,
                detection_source,
            )

            if elapsed_seconds >= max_seconds:
                break

        finalize_run_row(
            connection,
            run_id,
            calibration_frames=calibration_frames,
            calibration_elapsed_seconds=calibration_elapsed_seconds,
            roi_locked=roi_locked,
            roi_bounds=roi_bounds,
            processed_frames=total_frames,
            elapsed_seconds=(time.perf_counter() - start_time),
            average_fps=float(np.mean(fps_values)) if fps_values else 0.0,
            average_infer_fps=float(np.mean(infer_fps_values)) if infer_fps_values else 0.0,
            average_infer_ms=float(np.mean(infer_ms_values)) if infer_ms_values else 0.0,
            mean_detection_count=float(np.mean(detection_counts)) if detection_counts else 0.0,
            mean_roi_area_ratio=float(np.mean(roi_area_ratios)) if roi_area_ratios else 0.0,
            status=status,
        )
        print(
            f"[roi-benchmark] {trial.sample_name} | {trial.mode} | "
            f"frames={total_frames} avg_fps={np.mean(fps_values) if fps_values else 0.0:.1f} "
            f"avg_infer_fps={np.mean(infer_fps_values) if infer_fps_values else 0.0:.1f} "
            f"mean_input_area_ratio={np.mean(roi_area_ratios) if roi_area_ratios else 0.0:.3f} "
            f"roi_locked={roi_locked}"
        )
    except Exception:
        status = "failed"
        finalize_run_row(
            connection,
            run_id,
            calibration_frames=calibration_frames,
            calibration_elapsed_seconds=calibration_elapsed_seconds,
            roi_locked=roi_locked,
            roi_bounds=roi_bounds,
            processed_frames=total_frames,
            elapsed_seconds=(time.perf_counter() - start_time),
            average_fps=float(np.mean(fps_values)) if fps_values else 0.0,
            average_infer_fps=float(np.mean(infer_fps_values)) if infer_fps_values else 0.0,
            average_infer_ms=float(np.mean(infer_ms_values)) if infer_ms_values else 0.0,
            mean_detection_count=float(np.mean(detection_counts)) if detection_counts else 0.0,
            mean_roi_area_ratio=float(np.mean(roi_area_ratios)) if roi_area_ratios else 0.0,
            status=status,
        )
        raise
    finally:
        input_layer.close_input_layer()


def main() -> None:
    settings = load_runtime_settings()
    parser = argparse.ArgumentParser(description="Headless benchmark of direct video input vs ROI-cropped YOLO input.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for SQLite output and summary chart.")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS, help="Max seconds to process per trial.")
    parser.add_argument("--roi-threshold", type=int, default=None, help="Optional override for config_roi_vehicle_count_threshold during ROI calibration.")
    args = parser.parse_args()

    if args.roi_threshold is not None:
        settings["roi_threshold"] = int(args.roi_threshold)

    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir)

    initialize_yolo_layer(
        model_name=str(settings["model"]),
        conf_threshold=float(settings["conf"]),
        device=str(settings["device"]),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = output_dir / f"roi_eval_metrics_{timestamp}.sqlite"
    chart_path = output_dir / f"roi_eval_metrics_{timestamp}_summary.png"
    connection = initialize_db(db_path)

    try:
        for trial in build_trials():
            run_trial(
                trial,
                settings=settings,
                connection=connection,
                max_seconds=float(args.max_seconds),
            )
    finally:
        connection.close()

    create_summary_plot(db_path, chart_path)
    print(f"[roi-benchmark] SQLite: {db_path}")
    print(f"[roi-benchmark] Chart:  {chart_path}")
    print(f"[roi-benchmark] Output dir: {output_dir}")
    print(f"[roi-benchmark] ROI threshold used: {int(settings['roi_threshold'])}")
    print("[roi-benchmark] Comparison mode: direct full video input vs fixed ROI crop used as YOLO input after a separate calibration pass.")


if __name__ == "__main__":
    main()
