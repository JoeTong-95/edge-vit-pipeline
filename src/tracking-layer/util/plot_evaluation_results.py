#!/usr/bin/env python3
"""
plot_evaluation_results.py
Create styled scatter/mean plots from tracking evaluation SQLite output.

Examples:
    python .\plot_evaluation_results.py
    python .\plot_evaluation_results.py --db "E:\OneDrive\desktop\video\tracking_eval_metrics_20260408_215712.sqlite"
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_DIR = Path(r"E:\OneDrive\desktop\video")

BACKGROUND = "#171717"
PANEL = "#242424"
GRID = "#4a4a4a"
TEXT = "#f3f3f3"
MUTED = "#bdbdbd"
ORANGES = {
    ("cpu", 0): "#ffb347",
    ("cpu", 1): "#ff9f1c",
    ("cuda", 0): "#ff7b00",
    ("cuda", 1): "#ff5400",
}


def find_latest_db(search_dir: Path) -> Path:
    candidates = sorted(search_dir.glob("tracking_eval_metrics*.sqlite"), key=lambda item: item.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No tracking evaluation SQLite file found in {search_dir}")
    return candidates[0]


def load_data(db_path: Path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    runs = conn.execute(
        """
        SELECT run_id, model_family, device_mode, tracking_enabled,
               average_fps, average_infer_fps, processed_frames, elapsed_seconds,
               average_detections, average_tracks, status
        FROM evaluation_runs
        WHERE status = 'completed'
        ORDER BY model_family, device_mode, tracking_enabled
        """
    ).fetchall()
    frames = conn.execute(
        """
        SELECT run_id, frame_id, fps_actual, infer_fps
        FROM evaluation_frames
        ORDER BY frame_record_id
        """
    ).fetchall()
    conn.close()
    return runs, frames


def build_run_label(run):
    tracking_label = "track" if int(run["tracking_enabled"]) == 1 else "detect"
    return f"{run['model_family'].upper()}\n{run['device_mode']}\n{tracking_label}"


def build_legend_handles():
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGES[("cpu", 0)], markeredgecolor="none", markersize=8, label="CPU detect only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGES[("cpu", 1)], markeredgecolor="none", markersize=8, label="CPU with tracking"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGES[("cuda", 0)], markeredgecolor="none", markersize=8, label="CUDA detect only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGES[("cuda", 1)], markeredgecolor="none", markersize=8, label="CUDA with tracking"),
        Line2D([0], [0], marker="o", color=TEXT, markerfacecolor=TEXT, markeredgecolor=TEXT, linewidth=0, markersize=5, alpha=0.35, label="Per-frame scatter"),
        Line2D([0], [0], color=TEXT, linewidth=2.4, marker="o", markerfacecolor=TEXT, markeredgecolor=TEXT, markersize=5, label="Run mean"),
    ]
    return handles


def plot_metric(ax, frames_by_run, runs, metric_key, mean_key, title, ylabel):
    x_positions = np.arange(len(runs))
    labels = []
    for index, run in enumerate(runs):
        labels.append(build_run_label(run))
        values = np.array(frames_by_run[run["run_id"]][metric_key], dtype=float)
        if values.size == 0:
            continue
        color = ORANGES[(run["device_mode"], int(run["tracking_enabled"]))]
        jitter = np.random.uniform(-0.18, 0.18, size=values.size)
        ax.scatter(
            np.full(values.shape, index, dtype=float) + jitter,
            values,
            s=7,
            alpha=0.18,
            color=color,
            edgecolors="none",
        )
        mean_value = float(run[mean_key])
        ax.hlines(mean_value, index - 0.28, index + 0.28, colors=color, linewidth=2.4)
        ax.scatter(index, mean_value, s=80, color=color, edgecolors=TEXT, linewidths=0.8, zorder=5)
        ax.text(index, mean_value, f" {mean_value:.1f}", color=TEXT, fontsize=8, va="center", ha="left")

    ax.set_title(title, color=TEXT, fontsize=14, pad=12)
    ax.set_ylabel(ylabel, color=TEXT)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, color=TEXT, fontsize=9)
    ax.tick_params(axis="y", colors=TEXT)
    ax.grid(True, axis="y", color=GRID, alpha=0.35, linewidth=0.8)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def create_summary_plot(db_path: Path, output_path: Path):
    runs, frames = load_data(db_path)
    if not runs:
        raise RuntimeError("No completed runs found in evaluation_runs")

    frames_by_run = {
        run["run_id"]: {"fps_actual": [], "infer_fps": []}
        for run in runs
    }
    for row in frames:
        run_id = row["run_id"]
        if run_id not in frames_by_run:
            continue
        frames_by_run[run_id]["fps_actual"].append(row["fps_actual"])
        frames_by_run[run_id]["infer_fps"].append(row["infer_fps"])

    plt.style.use("default")
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), sharex=True)
    fig.patch.set_facecolor(BACKGROUND)

    plot_metric(
        axes[0],
        frames_by_run,
        runs,
        "fps_actual",
        "average_fps",
        "End-to-End FPS by Configuration",
        "FPS",
    )
    plot_metric(
        axes[1],
        frames_by_run,
        runs,
        "infer_fps",
        "average_infer_fps",
        "Inference FPS by Configuration",
        "Infer FPS",
    )

    title = f"Tracking Evaluation Summary\n{db_path.name}"
    fig.suptitle(title, color=TEXT, fontsize=18, y=0.98)
    legend = fig.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=3,
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


def _fallback_output_path(output_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")



def main():
    parser = argparse.ArgumentParser(description="Plot tracking evaluation SQLite results")
    parser.add_argument("--db", default="", help="SQLite database path. Defaults to the newest tracking_eval_metrics*.sqlite in E:\OneDrive\desktop\video")
    parser.add_argument("--output", default="", help="Output plot path. Defaults next to the DB with a _summary.png suffix")
    args = parser.parse_args()

    db_path = Path(args.db) if args.db else find_latest_db(_DEFAULT_OUTPUT_DIR)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database not found: {db_path}")

    output_path = Path(args.output) if args.output else db_path.with_name(f"{db_path.stem}_summary.png")
    try:
        create_summary_plot(db_path, output_path)
    except PermissionError:
        fallback_output = _fallback_output_path(output_path)
        print(f"Output plot path locked, using fallback: {fallback_output}")
        output_path = fallback_output
        create_summary_plot(db_path, output_path)
    print(f"DB: {db_path}")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
