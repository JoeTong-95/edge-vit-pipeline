#!/usr/bin/env python3
"""
Local human-review app for a GRACE/ViT run folder.

The app reads:
  data/grace_run_2hour/grace_results.jsonl

and writes:
  data/grace_run_2hour/human_review/review_sample.jsonl
  data/grace_run_2hour/human_review/human_labels.jsonl
  data/grace_run_2hour/human_review/human_labels_current.json
  data/grace_run_2hour/human_review/summary.json
  data/grace_run_2hour/human_review/summary.md

Only rows promoted to real GRACE inference are sampled by default
(`grace_skipped=false`).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = REPO_ROOT / "data" / "grace_run_2hour"
DEFAULT_TARGET = 200
DEFAULT_SEED = 4221

TYPE_LABELS = {"correct", "wrong"}
EXTRA_LABELS = {"correct_extra", "wrong_extra"}
QUALITY_LABELS = {"bad_image_quality"}
OTHER_LABELS = {"repetitive"}
ALLOWED_LABELS = TYPE_LABELS | EXTRA_LABELS | QUALITY_LABELS | OTHER_LABELS


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _pct(num: int, den: int) -> float:
    return round(float(num / den), 6) if den else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _normalize_relpath(path_text: str) -> str:
    return path_text.replace("\\", "/").lstrip("/")


def _resolve_crop(run_dir: Path, crop_path: str) -> Path:
    crop_rel = _normalize_relpath(crop_path)
    repo_candidate = (REPO_ROOT / crop_rel).resolve()
    if repo_candidate.exists():
        return repo_candidate
    run_candidate = (run_dir / crop_rel).resolve()
    if run_candidate.exists():
        return run_candidate
    return repo_candidate


def _item_id(row: dict[str, Any]) -> str:
    track_id = str(row.get("track_id") or "unknown")
    frame_id = str(row.get("frame_id") or "0")
    call_id = str(row.get("grace_call_index") or row.get("vlm_call_id") or row.get("_source_line") or "0")
    return f"track_{track_id}__frame_{frame_id}__call_{call_id}"


def _build_review_item(row: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    normalized = row.get("normalized_result") if isinstance(row.get("normalized_result"), dict) else {}
    crop_path = str(row.get("crop_path") or "")
    crop_abs = _resolve_crop(run_dir, crop_path) if crop_path else Path("")
    item = {
        "item_id": _item_id(row),
        "source_line": row.get("_source_line"),
        "track_id": str(row.get("track_id") or ""),
        "frame_id": _safe_int(row.get("frame_id")),
        "crop_path": _normalize_relpath(crop_path),
        "crop_exists": bool(crop_abs.exists()),
        "detector_class": str(row.get("detector_class") or ""),
        "bbox": row.get("bbox") or [],
        "dispatch_mode": str(row.get("dispatch_mode") or ""),
        "dispatch_reason": str(row.get("dispatch_reason") or ""),
        "query_type": str(row.get("query_type") or ""),
        "ack_status": str(row.get("ack_status") or normalized.get("vlm_ack_status") or ""),
        "ack_reason": str(row.get("ack_reason") or ""),
        "is_target_vehicle": bool(normalized.get("is_target_vehicle")),
        "vehicle_type": str(normalized.get("vehicle_type") or ""),
        "fhwa_class": str(normalized.get("fhwa_class") or ""),
        "fhwa_confidence": normalized.get("fhwa_confidence"),
        "fhwa_index": normalized.get("fhwa_index"),
        "axle_count": normalized.get("axle_count"),
        "trailer_count": normalized.get("trailer_count"),
        "grace_inference_ms": _safe_float(row.get("grace_inference_ms")),
        "avg_grace_inference_ms": row.get("avg_grace_inference_ms"),
        "elapsed_run_seconds": _safe_float(row.get("elapsed_run_seconds")),
        "grace_calls_per_minute": row.get("grace_calls_per_minute"),
        "raw_response": str(row.get("raw_response") or ""),
    }
    item["image_url"] = f"/image?item_id={quote(item['item_id'])}"
    return item


def _promoted_rows(rows: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    promoted = []
    for row in rows:
        if bool(row.get("grace_skipped")):
            continue
        crop_path = str(row.get("crop_path") or "")
        if crop_path and not _resolve_crop(run_dir, crop_path).exists():
            continue
        promoted.append(row)
    return promoted


def _balanced_sample(rows: list[dict[str, Any]], target: int, seed: int, run_dir: Path) -> list[dict[str, Any]]:
    if target <= 0 or target >= len(rows):
        selected = list(rows)
    else:
        rng = random.Random(seed)
        by_class: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            cls = str(row.get("detector_class") or "unknown")
            by_class.setdefault(cls, []).append(row)
        for cls_rows in by_class.values():
            rng.shuffle(cls_rows)

        classes = sorted(by_class)
        per_class = max(1, target // max(1, len(classes)))
        selected_rows: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        for cls in classes:
            for row in by_class[cls][:per_class]:
                selected_rows.append(row)
                selected_ids.add(_item_id(row))
        leftovers = [row for row in rows if _item_id(row) not in selected_ids]
        rng.shuffle(leftovers)
        selected_rows.extend(leftovers[: max(0, target - len(selected_rows))])
        selected = selected_rows[:target]

    selected.sort(key=lambda r: (_safe_int(r.get("frame_id")), str(r.get("track_id") or ""), _safe_int(r.get("_source_line"))))
    return [_build_review_item(row, run_dir) for row in selected]


def _load_or_create_sample(
    *,
    run_dir: Path,
    review_dir: Path,
    target: int,
    seed: int,
    resample: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results_path = run_dir / "grace_results.jsonl"
    all_rows = _read_jsonl(results_path)
    promoted = _promoted_rows(all_rows, run_dir)
    sample_path = review_dir / "review_sample.jsonl"
    metadata_path = review_dir / "review_sample_meta.json"

    if sample_path.exists() and not resample:
        items = _read_jsonl(sample_path)
        meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        meta.setdefault("reused_existing_sample", True)
        meta.setdefault("target_requested", target)
        meta.setdefault("promoted_grace_call_count", len(promoted))
        meta.setdefault("total_result_rows", len(all_rows))
        return items, meta

    items = _balanced_sample(promoted, target=target, seed=seed, run_dir=run_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(sample_path, items)
    class_counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("detector_class") or "unknown")
        class_counts[key] = class_counts.get(key, 0) + 1
    meta = {
        "schema": "grace_human_review_sample_v1",
        "created_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "results_path": str(results_path),
        "target_requested": target,
        "seed": seed,
        "total_result_rows": len(all_rows),
        "promoted_grace_call_count": len(promoted),
        "sample_count": len(items),
        "sample_detector_class_counts": class_counts,
        "promoted_only": True,
        "reused_existing_sample": False,
    }
    metadata_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return items, meta


def _load_current_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)


def _public_item(item: dict[str, Any], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = dict(item)
    out["label_state"] = labels.get(str(item["item_id"]), {})
    out["image_url"] = f"/image?item_id={quote(str(item['item_id']))}"
    return out


def _is_complete(label_state: dict[str, Any]) -> bool:
    labels = set(label_state.get("labels") or [])
    return bool(labels.intersection(ALLOWED_LABELS))


def _find_next_item(
    items: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    current_item_id: str = "",
    direction: int = 1,
) -> dict[str, Any] | None:
    if not items:
        return None
    start = 0
    if current_item_id:
        for idx, item in enumerate(items):
            if str(item["item_id"]) == current_item_id:
                start = (idx + direction) % len(items)
                break
    for offset in range(len(items)):
        idx = (start + offset * direction) % len(items)
        item = items[idx]
        if not _is_complete(labels.get(str(item["item_id"]), {})):
            return item
    return None


def _timing_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    infer_ms = [_safe_float(item.get("grace_inference_ms")) for item in items if _safe_float(item.get("grace_inference_ms")) > 0]
    elapsed = [_safe_float(item.get("elapsed_run_seconds")) for item in items if item.get("elapsed_run_seconds") is not None]
    calls_per_minute = [
        _safe_float(item.get("grace_calls_per_minute"))
        for item in items
        if item.get("grace_calls_per_minute") is not None
    ]
    return {
        "count": len(items),
        "avg_grace_inference_ms": round(_mean(infer_ms), 3),
        "median_grace_inference_ms": round(_median(infer_ms), 3),
        "min_grace_inference_ms": round(min(infer_ms), 3) if infer_ms else 0.0,
        "max_grace_inference_ms": round(max(infer_ms), 3) if infer_ms else 0.0,
        "max_elapsed_run_seconds": round(max(elapsed), 3) if elapsed else 0.0,
        "last_grace_calls_per_minute": round(calls_per_minute[-1], 3) if calls_per_minute else 0.0,
    }


def build_summary(
    *,
    run_dir: Path,
    review_dir: Path,
    items: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    sample_meta: dict[str, Any],
) -> dict[str, Any]:
    reviewed = {item_id: state for item_id, state in labels.items() if _is_complete(state)}
    type_correct = 0
    type_wrong = 0
    extra_correct = 0
    extra_wrong = 0
    bad_quality = 0
    repetitive = 0
    for state in labels.values():
        label_set = set(state.get("labels") or [])
        type_correct += int("correct" in label_set)
        type_wrong += int("wrong" in label_set)
        extra_correct += int("correct_extra" in label_set)
        extra_wrong += int("wrong_extra" in label_set)
        bad_quality += int("bad_image_quality" in label_set)
        repetitive += int("repetitive" in label_set)

    class_counts: dict[str, int] = {}
    reviewed_by_class: dict[str, int] = {}
    wrong_by_class: dict[str, int] = {}
    for item in items:
        cls = str(item.get("detector_class") or "unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1
        state = labels.get(str(item["item_id"]), {})
        if _is_complete(state):
            reviewed_by_class[cls] = reviewed_by_class.get(cls, 0) + 1
        if "wrong" in set(state.get("labels") or []):
            wrong_by_class[cls] = wrong_by_class.get(cls, 0) + 1

    type_judged = type_correct + type_wrong
    extra_judged = extra_correct + extra_wrong
    summary = {
        "schema": "grace_human_review_summary_v1",
        "created_at_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "review_dir": str(review_dir),
        "target_requested": sample_meta.get("target_requested"),
        "sample_count": len(items),
        "reviewed_count": len(reviewed),
        "remaining_count": max(0, len(items) - len(reviewed)),
        "vehicle_type": {
            "correct": type_correct,
            "wrong": type_wrong,
            "judged": type_judged,
            "accuracy_excluding_bad_quality": round(_pct(type_correct, type_judged), 6),
        },
        "extra_fields": {
            "correct_extra": extra_correct,
            "wrong_extra": extra_wrong,
            "judged": extra_judged,
            "accuracy": round(_pct(extra_correct, extra_judged), 6),
        },
        "bad_image_quality": bad_quality,
        "repetitive": repetitive,
        "class_counts": class_counts,
        "reviewed_by_class": reviewed_by_class,
        "wrong_by_class": wrong_by_class,
        "sample_timing": _timing_summary(items),
        "sample_meta": sample_meta,
    }
    return summary


def generate_accuracy_plot(
    *,
    review_dir: Path,
    labels: dict[str, dict[str, Any]],
    summary: dict[str, Any],
) -> dict[str, str]:
    out_png = review_dir / "accuracy_summary.png"
    out_svg = review_dir / "accuracy_summary.svg"

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local plotting deps
        return {"error": f"matplotlib unavailable: {exc}"}

    label_sets = {
        "Correct type": {k for k, v in labels.items() if "correct" in set(v.get("labels") or [])},
        "Wrong type": {k for k, v in labels.items() if "wrong" in set(v.get("labels") or [])},
        "Bad image": {k for k, v in labels.items() if "bad_image_quality" in set(v.get("labels") or [])},
        "Correct extra": {k for k, v in labels.items() if "correct_extra" in set(v.get("labels") or [])},
        "Wrong extra": {k for k, v in labels.items() if "wrong_extra" in set(v.get("labels") or [])},
        "Repetitive": {k for k, v in labels.items() if "repetitive" in set(v.get("labels") or [])},
    }
    sample_count = int(summary.get("sample_count") or 0)
    reviewed_count = int(summary.get("reviewed_count") or len(labels))

    navy = "#062a57"
    blue_dark = "#0d4776"
    blue = "#116a99"
    blue_mid = "#6aa0b7"
    blue_light = "#b5cfdf"
    ink = "#0c1824"
    grid = "#d9e4eb"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": ink,
            "axes.labelcolor": ink,
            "xtick.color": ink,
            "ytick.color": ink,
        }
    )

    fig, (ax_bar, ax_cls) = plt.subplots(
        1,
        2,
        figsize=(15.5, 7.6),
        gridspec_kw={"width_ratios": [1.08, 1.0]},
        constrained_layout=True,
    )

    bar_labels = [
        "Total sample",
        "Correct type",
        "Wrong type",
        "Bad image",
        "Correct extra",
        "Wrong extra",
        "Repetitive",
    ]
    bar_values = [
        sample_count,
        len(label_sets["Correct type"]),
        len(label_sets["Wrong type"]),
        len(label_sets["Bad image"]),
        len(label_sets["Correct extra"]),
        len(label_sets["Wrong extra"]),
        len(label_sets["Repetitive"]),
    ]
    bar_colors = [blue_dark, blue, blue_mid, blue_light, blue, blue_mid, navy]
    bars = ax_bar.barh(bar_labels, bar_values, color=bar_colors, edgecolor="white", linewidth=1.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, max(bar_values or [1]) * 1.24)
    ax_bar.set_title("Total Human Review Summary", pad=12)
    ax_bar.set_xlabel("Item count")
    ax_bar.grid(axis="x", color=grid, linewidth=0.9)
    ax_bar.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax_bar.spines[spine].set_visible(False)
    for bar, value in zip(bars, bar_values):
        ax_bar.text(
            value + max(bar_values or [1]) * 0.018,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontweight="bold",
        )

    vehicle = summary.get("vehicle_type") or {}
    extra = summary.get("extra_fields") or {}
    rate_lines = [
        f"Reviewed: {reviewed_count}/{sample_count}",
        f"Vehicle type accuracy: {float(vehicle.get('accuracy_excluding_bad_quality') or 0):.1%}",
        f"Extra-field accuracy: {float(extra.get('accuracy') or 0):.1%}",
        f"Bad-image rate: {len(label_sets['Bad image']) / sample_count:.1%}" if sample_count else "Bad-image rate: n/a",
        f"Repetitive rate: {len(label_sets['Repetitive']) / sample_count:.1%}" if sample_count else "Repetitive rate: n/a",
    ]
    ax_bar.text(
        0.02,
        -0.12,
        "\n".join(rate_lines),
        transform=ax_bar.transAxes,
        va="top",
        ha="left",
        fontsize=12.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f8fb", edgecolor=blue_light),
    )

    classes = sorted((summary.get("class_counts") or {}).keys())
    wrong_by_class = summary.get("wrong_by_class") or {}
    reviewed_by_class = summary.get("reviewed_by_class") or {}
    not_wrong = [max(0, int(reviewed_by_class.get(c, 0)) - int(wrong_by_class.get(c, 0))) for c in classes]
    wrong_vals = [int(wrong_by_class.get(c, 0)) for c in classes]
    ax_cls.bar(classes, not_wrong, label="not wrong type", color=blue, edgecolor="white")
    ax_cls.bar(classes, wrong_vals, bottom=not_wrong, label="wrong type", color=blue_light, edgecolor="white")
    ax_cls.set_title("Vehicle Type Review By Detector Class", pad=12)
    ax_cls.set_ylabel("Reviewed items")
    ax_cls.grid(axis="y", color=grid, linewidth=0.9)
    ax_cls.set_axisbelow(True)
    ax_cls.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=blue_light)
    for spine in ["top", "right"]:
        ax_cls.spines[spine].set_visible(False)
    for i, c in enumerate(classes):
        total = int(reviewed_by_class.get(c, 0))
        wrong = int(wrong_by_class.get(c, 0))
        ax_cls.text(i, total + 1.1, f"{wrong}/{total} wrong", ha="center", fontsize=10.5)
    ax_cls.set_ylim(0, max([int(reviewed_by_class.get(c, 0)) for c in classes] + [1]) * 1.18)

    fig.suptitle("GRACE/ViT Human Review: 200 Promoted Accepted Items", fontsize=18, fontweight="bold")
    fig.savefig(out_png, dpi=190)
    fig.savefig(out_svg)
    plt.close(fig)
    return {
        "accuracy_plot_png": str(out_png),
        "accuracy_plot_svg": str(out_svg),
    }


def write_summary_files(
    *,
    run_dir: Path,
    review_dir: Path,
    items: list[dict[str, Any]],
    labels: dict[str, dict[str, Any]],
    sample_meta: dict[str, Any],
) -> dict[str, Any]:
    summary = build_summary(
        run_dir=run_dir,
        review_dir=review_dir,
        items=items,
        labels=labels,
        sample_meta=sample_meta,
    )
    plot_result = generate_accuracy_plot(review_dir=review_dir, labels=labels, summary=summary)
    summary.update(plot_result)
    _atomic_write_json(review_dir / "summary.json", summary)
    v = summary["vehicle_type"]
    e = summary["extra_fields"]
    t = summary["sample_timing"]
    lines = [
        "# GRACE Human Review Summary",
        "",
        f"- Created at: `{summary['created_at_utc']}`",
        f"- Sample count: `{summary['sample_count']}`",
        f"- Reviewed: `{summary['reviewed_count']}`",
        f"- Remaining: `{summary['remaining_count']}`",
        "",
        "## Vehicle Type",
        "",
        f"- Correct: `{v['correct']}`",
        f"- Wrong: `{v['wrong']}`",
        f"- Accuracy excluding bad image quality: `{v['accuracy_excluding_bad_quality']}`",
        "",
        "## Extra Fields",
        "",
        f"- Correct extra: `{e['correct_extra']}`",
        f"- Wrong extra: `{e['wrong_extra']}`",
        f"- Extra accuracy: `{e['accuracy']}`",
        "",
        "## Image Quality",
        "",
        f"- Bad image quality: `{summary['bad_image_quality']}`",
        f"- Repetitive: `{summary['repetitive']}`",
        "",
        "## Timing",
        "",
        f"- Avg GRACE inference ms: `{t['avg_grace_inference_ms']}`",
        f"- Median GRACE inference ms: `{t['median_grace_inference_ms']}`",
        f"- Max elapsed run seconds: `{t['max_elapsed_run_seconds']}`",
        f"- Last GRACE calls per minute: `{t['last_grace_calls_per_minute']}`",
        "",
        "## Counts",
        "",
        f"- Class counts: `{json.dumps(summary['class_counts'], sort_keys=True)}`",
        f"- Reviewed by class: `{json.dumps(summary['reviewed_by_class'], sort_keys=True)}`",
        f"- Wrong by class: `{json.dumps(summary['wrong_by_class'], sort_keys=True)}`",
        "",
        "## Artifacts",
        "",
        f"- Accuracy plot PNG: `{summary.get('accuracy_plot_png', '')}`",
        f"- Accuracy plot SVG: `{summary.get('accuracy_plot_svg', '')}`",
        "",
    ]
    (review_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GRACE Human Review</title>
  <style>
    :root {
      --bg: #f6f7f2;
      --panel: #ffffff;
      --ink: #202124;
      --muted: #626a6f;
      --line: #d9dfd8;
      --accent: #176b5b;
      --accent2: #2f5d8c;
      --bad: #a93c2f;
      --warn: #9a6b00;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", Arial, sans-serif; color: var(--ink); background: var(--bg); }
    .top { display: grid; grid-template-columns: 1fr auto; gap: 12px; align-items: center; padding: 12px 16px; border-bottom: 1px solid var(--line); background: #fff; position: sticky; top: 0; z-index: 5; }
    .title { font-weight: 700; }
    .sub { font-size: 13px; color: var(--muted); margin-top: 3px; }
    .actions { display: flex; gap: 8px; align-items: center; }
    button { border: 1px solid var(--line); border-radius: 6px; background: #fff; padding: 8px 10px; cursor: pointer; font: inherit; min-height: 36px; }
    button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
    button.type { border-color: var(--accent2); }
    button.extra { border-color: var(--accent); }
    button.bad { border-color: var(--bad); color: var(--bad); }
    .wrap { padding: 14px; max-width: 1360px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: minmax(420px, 1.25fr) minmax(360px, 0.75fr); gap: 12px; align-items: start; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    .imageBox { min-height: 520px; height: calc(100vh - 190px); display: flex; align-items: center; justify-content: center; background: #ecefeb; }
    .imageBox img { width: 100%; height: 100%; object-fit: contain; image-rendering: auto; }
    .meta { padding: 10px; }
    .heroFields { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 8px; }
    .heroField { border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #fbfcfa; }
    .heroField.full { grid-column: 1 / -1; }
    .heroKey { color: var(--muted); font-size: 12px; margin-bottom: 3px; }
    .heroValue { font-size: 22px; font-weight: 700; line-height: 1.15; }
    .row { display: grid; grid-template-columns: 120px 1fr; gap: 8px; padding: 5px 0; border-bottom: 1px solid #edf0ec; font-size: 13px; }
    .key { color: var(--muted); }
    .value { word-break: break-word; }
    .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 10px; border-top: 1px solid var(--line); }
    .full { grid-column: 1 / -1; }
    textarea, input { width: 100%; border: 1px solid var(--line); border-radius: 6px; padding: 8px; font: inherit; }
    .note { padding: 0 10px 10px; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px; border-top: 1px solid var(--line); }
    .chip { border-radius: 999px; padding: 4px 8px; font-size: 12px; background: #eef4f2; color: var(--accent); }
    .chip.bad { background: #fbebe9; color: var(--bad); }
    .plotPanel { display: none; margin-top: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px; }
    .plotPanel img { display: block; width: 100%; height: auto; border-radius: 6px; }
    .plotTitle { font-weight: 700; margin-bottom: 8px; }
    .progressOuter { height: 8px; border-radius: 999px; background: #e7ebe5; overflow: hidden; margin-top: 7px; }
    .progressInner { height: 100%; width: 0%; background: var(--accent); }
    .empty { padding: 40px; text-align: center; color: var(--muted); }
    @media (max-width: 900px) {
      .top { grid-template-columns: 1fr; }
      .actions { flex-wrap: wrap; }
      .grid { grid-template-columns: 1fr; }
      .imageBox { height: 48vh; min-height: 300px; }
    }
  </style>
</head>
<body>
  <div class="top">
    <div>
      <div class="title">GRACE Human Review</div>
      <div class="sub" id="targetLine">Loading...</div>
      <div class="progressOuter"><div class="progressInner" id="progressBar"></div></div>
    </div>
    <div class="actions">
      <button id="prevBtn">Prev</button>
      <button class="primary" id="nextBtn">Next</button>
      <button id="summaryBtn">Write Summary</button>
    </div>
  </div>
  <div class="wrap">
    <div class="grid">
      <div class="panel">
        <div class="imageBox"><img id="reviewImage" alt="review crop"/></div>
      </div>
      <div class="panel">
        <div class="meta" id="meta"></div>
        <div class="buttons">
          <button class="type" id="correctBtn">1 Correct Type</button>
          <button class="type" id="wrongBtn">2 Wrong Type</button>
          <button class="extra" id="correctExtraBtn">Q Correct Extra</button>
          <button class="extra" id="wrongExtraBtn">W Wrong Extra</button>
      <button class="bad" id="badBtn">3 Bad Image Quality</button>
      <button id="repetitiveBtn">4 Repetitive</button>
        </div>
        <div class="note">
          <textarea id="note" rows="3" placeholder="optional note"></textarea>
        </div>
        <div class="chips" id="chips"></div>
      </div>
    </div>
    <div class="plotPanel" id="plotPanel">
      <div class="plotTitle">Saved Accuracy Summary</div>
      <img id="plotImage" alt="accuracy summary plot"/>
    </div>
  </div>
<script>
let state = { item: null, summary: null };

function $(id) { return document.getElementById(id); }

function labels() {
  if (!state.item || !state.item.label_state) return [];
  return state.item.label_state.labels || [];
}

function renderTop(summary) {
  const reviewed = summary.reviewed_count || 0;
  const total = summary.sample_count || 0;
  const pct = total ? Math.round(1000 * reviewed / total) / 10 : 0;
  $("targetLine").textContent =
    `Review target: ${summary.target_requested || total} promoted GRACE/ViT items. Progress: ${reviewed} / ${total} reviewed (${pct}%).`;
  $("progressBar").style.width = `${pct}%`;
}

function renderItem(item, summary) {
  state.item = item;
  state.summary = summary;
  renderTop(summary);
  if (!item) {
    $("reviewImage").removeAttribute("src");
    $("meta").innerHTML = `<div class="empty">No remaining unlabeled items.</div>`;
    $("chips").innerHTML = "";
    return;
  }
  $("reviewImage").src = item.image_url;
  $("note").value = (item.label_state && item.label_state.note) || "";
  const confidence = item.fhwa_confidence === null || item.fhwa_confidence === undefined ? "" : Number(item.fhwa_confidence).toFixed(4);
  const inferMs = item.grace_inference_ms === null || item.grace_inference_ms === undefined ? "" : `${Number(item.grace_inference_ms).toFixed(1)} ms`;
  const extra = [
    ["track", item.track_id],
    ["frame", item.frame_id],
    ["confidence", confidence],
    ["runtime", inferMs]
  ];
  $("meta").innerHTML = `
    <div class="heroFields">
      <div class="heroField"><div class="heroKey">detector tag</div><div class="heroValue">${item.detector_class || ""}</div></div>
      <div class="heroField"><div class="heroKey">final GRACE tag</div><div class="heroValue">${item.vehicle_type || ""}</div></div>
      <div class="heroField"><div class="heroKey">FHWA class</div><div class="heroValue">${item.fhwa_class || ""}</div></div>
      <div class="heroField"><div class="heroKey">axle count</div><div class="heroValue">${item.axle_count ?? ""}</div></div>
      <div class="heroField"><div class="heroKey">trailer count</div><div class="heroValue">${item.trailer_count ?? ""}</div></div>
      <div class="heroField"><div class="heroKey">target vehicle</div><div class="heroValue">${item.is_target_vehicle}</div></div>
    </div>
    ${extra.map(([k, v]) => `<div class="row"><div class="key">${k}</div><div class="value">${String(v ?? "")}</div></div>`).join("")}
  `;
  renderChips();
}

function renderChips() {
  const current = labels();
  if (!current.length) {
    $("chips").innerHTML = "";
    return;
  }
  $("chips").innerHTML = current.map(label => {
    const cls = label === "bad_image_quality" || label === "wrong" || label === "wrong_extra" ? "chip bad" : "chip";
    return `<span class="${cls}">${label}</span>`;
  }).join("");
}

async function loadNext(currentId="", direction=1) {
  const q = new URLSearchParams();
  if (currentId) q.set("current_item_id", currentId);
  q.set("direction", String(direction));
  const res = await fetch(`/api/next?${q.toString()}`);
  const data = await res.json();
  renderItem(data.item, data.summary);
}

async function saveLabel(label) {
  if (!state.item) return;
  if (document.activeElement && document.activeElement.tagName === "BUTTON") {
    document.activeElement.blur();
  }
  const payload = {
    item_id: state.item.item_id,
    label: label,
    note: $("note").value || ""
  };
  const res = await fetch("/api/label", {
    method: "POST",
    headers: {"content-type": "application/json"},
    body: JSON.stringify(payload)
  });
  if (!res.ok) {
    alert(await res.text());
    return;
  }
  const data = await res.json();
  renderItem(data.item, data.summary);
}

async function writeSummary() {
  const res = await fetch("/api/summary", {method: "POST"});
  const data = await res.json();
  renderTop(data.summary);
  const ts = Date.now();
  $("plotImage").src = `/artifact?name=accuracy_summary.png&ts=${ts}`;
  $("plotPanel").style.display = "block";
  $("plotPanel").scrollIntoView({behavior: "smooth", block: "start"});
}

$("nextBtn").onclick = () => loadNext(state.item ? state.item.item_id : "", 1);
$("prevBtn").onclick = () => loadNext(state.item ? state.item.item_id : "", -1);
$("summaryBtn").onclick = writeSummary;
$("correctBtn").onclick = () => saveLabel("correct");
$("wrongBtn").onclick = () => saveLabel("wrong");
$("correctExtraBtn").onclick = () => saveLabel("correct_extra");
$("wrongExtraBtn").onclick = () => saveLabel("wrong_extra");
$("badBtn").onclick = () => saveLabel("bad_image_quality");
$("repetitiveBtn").onclick = () => saveLabel("repetitive");

window.addEventListener("keydown", (ev) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
  if (document.activeElement.tagName === "BUTTON") {
    document.activeElement.blur();
  }
  const key = ev.key.toLowerCase();
  if (["1", "2", "3", "4", "q", "w", "b", "enter", "n", "arrowright", "p", "arrowleft"].includes(key)) {
    ev.preventDefault();
  }
  if (key === "1") saveLabel("correct");
  else if (key === "2") saveLabel("wrong");
  else if (key === "3") saveLabel("bad_image_quality");
  else if (key === "4") saveLabel("repetitive");
  else if (key === "q") saveLabel("correct_extra");
  else if (key === "w") saveLabel("wrong_extra");
  else if (key === "b") saveLabel("bad_image_quality");
  else if (key === "enter" || key === "n" || key === "arrowright") loadNext(state.item ? state.item.item_id : "", 1);
  else if (key === "p" || key === "arrowleft") loadNext(state.item ? state.item.item_id : "", -1);
});

loadNext();
</script>
</body>
</html>
"""


class GraceReviewHandler(BaseHTTPRequestHandler):
    server_version = "GraceReview/1.0"

    @property
    def app_state(self) -> dict[str, Any]:
        return self.server.app_state  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, payload: str, status: int = 200, content_type: str = "text/plain; charset=utf-8") -> None:
        body = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_len = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _summary(self) -> dict[str, Any]:
        return build_summary(
            run_dir=self.app_state["run_dir"],
            review_dir=self.app_state["review_dir"],
            items=self.app_state["items"],
            labels=self.app_state["labels"],
            sample_meta=self.app_state["sample_meta"],
        )

    def _item_by_id(self, item_id: str) -> dict[str, Any] | None:
        return self.app_state["items_by_id"].get(item_id)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return
        if parsed.path == "/api/next":
            params = parse_qs(parsed.query)
            current_item_id = str((params.get("current_item_id") or [""])[0])
            direction = _safe_int((params.get("direction") or ["1"])[0], 1)
            if direction not in (-1, 1):
                direction = 1
            item = _find_next_item(
                self.app_state["items"],
                self.app_state["labels"],
                current_item_id=current_item_id,
                direction=direction,
            )
            self._send_json(
                {
                    "item": _public_item(item, self.app_state["labels"]) if item else None,
                    "summary": self._summary(),
                }
            )
            return
        if parsed.path == "/api/summary":
            self._send_json({"summary": self._summary()})
            return
        if parsed.path == "/image":
            params = parse_qs(parsed.query)
            item_id = unquote(str((params.get("item_id") or [""])[0]))
            item = self._item_by_id(item_id)
            if item is None:
                self._send_text("Unknown item_id", status=HTTPStatus.NOT_FOUND)
                return
            image_path = _resolve_crop(self.app_state["run_dir"], str(item.get("crop_path") or ""))
            if not image_path.exists():
                self._send_text("Image not found", status=HTTPStatus.NOT_FOUND)
                return
            body = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/artifact":
            params = parse_qs(parsed.query)
            name = str((params.get("name") or [""])[0])
            if name not in {"accuracy_summary.png", "accuracy_summary.svg"}:
                self._send_text("Unsupported artifact", status=HTTPStatus.BAD_REQUEST)
                return
            artifact_path = (self.app_state["review_dir"] / name).resolve()
            if not artifact_path.exists():
                self._send_text("Artifact not found", status=HTTPStatus.NOT_FOUND)
                return
            ctype = "image/svg+xml" if artifact_path.suffix.lower() == ".svg" else "image/png"
            body = artifact_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_text("Not found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/label":
            try:
                payload = self._read_json()
            except json.JSONDecodeError:
                self._send_text("Invalid JSON", status=HTTPStatus.BAD_REQUEST)
                return
            item_id = str(payload.get("item_id") or "")
            label = str(payload.get("label") or "")
            note = str(payload.get("note") or "")
            if label not in ALLOWED_LABELS:
                self._send_text(f"Unsupported label: {label}", status=HTTPStatus.BAD_REQUEST)
                return
            item = self._item_by_id(item_id)
            if item is None:
                self._send_text("Unknown item_id", status=HTTPStatus.BAD_REQUEST)
                return

            labels = self.app_state["labels"]
            current = labels.setdefault(
                item_id,
                {
                    "item_id": item_id,
                    "labels": [],
                    "note": "",
                    "updated_at_utc": utc_now_iso(),
                    "history_count": 0,
                },
            )
            current_labels = set(current.get("labels") or [])
            if label in TYPE_LABELS:
                if label in current_labels:
                    current_labels.remove(label)
                else:
                    current_labels.difference_update(TYPE_LABELS | QUALITY_LABELS)
                    current_labels.add(label)
            elif label in EXTRA_LABELS:
                if label in current_labels:
                    current_labels.remove(label)
                else:
                    current_labels.difference_update(EXTRA_LABELS)
                    current_labels.add(label)
            elif label in QUALITY_LABELS:
                if label in current_labels:
                    current_labels.remove(label)
                else:
                    current_labels.difference_update(TYPE_LABELS | QUALITY_LABELS)
                    current_labels.add(label)
            elif label in OTHER_LABELS:
                if label in current_labels:
                    current_labels.remove(label)
                else:
                    current_labels.add(label)
            current["labels"] = sorted(current_labels)
            current["note"] = note
            current["updated_at_utc"] = utc_now_iso()
            current["history_count"] = int(current.get("history_count") or 0) + 1

            review_dir = self.app_state["review_dir"]
            history_path = review_dir / "human_labels.jsonl"
            with history_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps({"event": "label", **current}, sort_keys=True) + "\n")
            _atomic_write_json(review_dir / "human_labels_current.json", labels)
            summary = write_summary_files(
                run_dir=self.app_state["run_dir"],
                review_dir=review_dir,
                items=self.app_state["items"],
                labels=labels,
                sample_meta=self.app_state["sample_meta"],
            )
            self._send_json({"ok": True, "item": _public_item(item, labels), "summary": summary})
            return
        if parsed.path == "/api/summary":
            summary = write_summary_files(
                run_dir=self.app_state["run_dir"],
                review_dir=self.app_state["review_dir"],
                items=self.app_state["items"],
                labels=self.app_state["labels"],
                sample_meta=self.app_state["sample_meta"],
            )
            self._send_json({"ok": True, "summary": summary})
            return
        self._send_text("Not found", status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.app_state.get("quiet"):
            return
        super().log_message(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local GRACE human-review GUI.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="GRACE run folder.")
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Target number of promoted GRACE calls to review.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    parser.add_argument("--resample", action="store_true", help="Rebuild review_sample.jsonl.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8790, help="Bind port.")
    parser.add_argument("--quiet", action="store_true", help="Suppress request logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    review_dir = run_dir / "human_review"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run folder not found: {run_dir}")
    if not (run_dir / "grace_results.jsonl").exists():
        raise FileNotFoundError(f"Missing grace_results.jsonl under: {run_dir}")

    items, sample_meta = _load_or_create_sample(
        run_dir=run_dir,
        review_dir=review_dir,
        target=max(1, int(args.target)),
        seed=int(args.seed),
        resample=bool(args.resample),
    )
    labels_path = review_dir / "human_labels_current.json"
    labels = _load_current_labels(labels_path)
    write_summary_files(
        run_dir=run_dir,
        review_dir=review_dir,
        items=items,
        labels=labels,
        sample_meta=sample_meta,
    )
    items_by_id = {str(item["item_id"]): item for item in items}

    server = ThreadingHTTPServer((args.host, args.port), GraceReviewHandler)
    server.app_state = {  # type: ignore[attr-defined]
        "run_dir": run_dir,
        "review_dir": review_dir,
        "items": items,
        "items_by_id": items_by_id,
        "labels": labels,
        "sample_meta": sample_meta,
        "quiet": bool(args.quiet),
    }
    print("[grace-review] promoted_only=true")
    print(f"[grace-review] target_requested={sample_meta.get('target_requested')} sample_count={len(items)}")
    print(f"[grace-review] promoted_grace_call_count={sample_meta.get('promoted_grace_call_count')}")
    print(f"[grace-review] review_dir={review_dir}")
    print(f"[grace-review] open: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
