#!/usr/bin/env python3
"""
Compare pipeline review outputs against human_truth.sqlite labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_CLASSES = {"pickup", "van", "truck", "bus"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _to_bool(value: Any, default: bool = False) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _load_vlm_rows(review_root: Path, run_id_filter: str = "") -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    runs_dir = review_root / "runs"
    if not runs_dir.exists():
        return rows
    run_dirs = sorted(runs_dir.glob("*"), key=lambda p: p.name)
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "metadata" / "vlm_accepted_targets.csv"
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                row_run_id = str(row.get("run_id") or run_dir.name)
                if run_id_filter and row_run_id != run_id_filter:
                    continue
                target_class = str(row.get("target_class") or "").strip().lower()
                if target_class not in TARGET_CLASSES:
                    continue
                row["run_id"] = row_run_id
                rows.append(row)
    return rows


def _collect_truth_from_db(conn: sqlite3.Connection, run_id_filter: str = "") -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    params: list[Any] = []
    run_where = ""
    if run_id_filter:
        run_where = "AND ri.run_id = ?"
        params.append(run_id_filter)

    base_rows = conn.execute(
        f"""
        SELECT ri.id, ri.run_id, ri.item_type, ri.track_id, ri.target_class,
               rl.label_key, rl.label_value
        FROM review_items ri
        LEFT JOIN review_labels rl ON rl.review_item_id = ri.id
        WHERE ri.item_type = 'new_track'
        {run_where}
        ORDER BY ri.run_id, ri.track_id, rl.id
        """,
        params,
    ).fetchall()

    truth_by_track: dict[tuple[str, str], dict[str, Any]] = {}
    all_track_keys: set[tuple[str, str]] = set()
    for row in base_rows:
        key = (str(row["run_id"]), str(row["track_id"]))
        all_track_keys.add(key)
        ent = truth_by_track.setdefault(
            key,
            {
                "has_true_class": False,
                "has_wrong_class": False,
                "has_repeat": False,
                "true_class_value": "",
            },
        )
        label_key = str(row["label_key"] or "")
        label_value = str(row["label_value"] or "").strip()
        if label_key == "true_class":
            ent["has_true_class"] = True
            if label_value:
                ent["true_class_value"] = label_value
        elif label_key == "wrong_class":
            ent["has_wrong_class"] = True
        elif label_key == "repeat":
            ent["has_repeat"] = True

    total_tracked_targets = len(all_track_keys)
    total_human_confirmed_trucks = 0
    total_wrong_class_tracks = 0
    total_repeat_tracks = 0
    truth_resolved: dict[tuple[str, str], bool | None] = {}
    true_class_value_by_track: dict[tuple[str, str], str] = {}
    for key, info in truth_by_track.items():
        truth: bool | None
        if info["has_wrong_class"]:
            truth = False
            total_wrong_class_tracks += 1
        elif info["has_true_class"]:
            truth = True
            total_human_confirmed_trucks += 1
        else:
            truth = None
        if info["has_repeat"]:
            total_repeat_tracks += 1
        truth_resolved[key] = truth
        if info["true_class_value"]:
            true_class_value_by_track[key] = str(info["true_class_value"]).strip().lower()

    tracker_precision_like = _safe_div(
        total_human_confirmed_trucks,
        (total_human_confirmed_trucks + total_wrong_class_tracks),
    )

    return {
        "truth_resolved": truth_resolved,
        "true_class_value_by_track": true_class_value_by_track,
        "tracker_metrics": {
            "total_tracked_targets": total_tracked_targets,
            "total_human_confirmed_trucks": total_human_confirmed_trucks,
            "total_repeat_tracks": total_repeat_tracks,
            "tracker_precision_like": round(tracker_precision_like, 6),
        },
    }


def _metadata_quality_metrics(conn: sqlite3.Connection, run_id_filter: str = "", vlm_rows: list[dict[str, str]] | None = None) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    params: list[Any] = []
    run_where = ""
    if run_id_filter:
        run_where = "AND ri.run_id = ?"
        params.append(run_id_filter)

    highlight_rows = conn.execute(
        f"""
        SELECT ri.run_id, ri.track_id, COALESCE(rh.field_name, '') AS field_name
        FROM review_highlights rh
        JOIN review_items ri ON ri.id = rh.review_item_id
        WHERE ri.item_type = 'vlm_accepted_target'
        {run_where}
        """,
        params,
    ).fetchall()

    problematic_track_keys = {(str(r["run_id"]), str(r["track_id"])) for r in highlight_rows}
    field_counts: dict[str, int] = {}
    for r in highlight_rows:
        field_name = str(r["field_name"] or "").strip() or "unspecified"
        field_counts[field_name] = field_counts.get(field_name, 0) + 1

    vlm_rows = vlm_rows or []
    matched_id_keys = {(str(r.get("run_id", "")), str(r.get("track_id", ""))) for r in vlm_rows}
    matched_id_keys.discard(("", ""))
    matched_id_count = len(matched_id_keys)
    problematic_metadata_count = len(problematic_track_keys.intersection(matched_id_keys))

    return {
        "matched_id_count": matched_id_count,
        "problematic_metadata_count": problematic_metadata_count,
        "problematic_metadata_rate": round(_safe_div(problematic_metadata_count, matched_id_count), 6),
        "problematic_by_field": field_counts,
    }


def run_comparison(review_root: Path, run_id: str = "") -> dict[str, Any]:
    db_path = review_root / "human_truth.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"Missing DB: {db_path}")

    vlm_rows = _load_vlm_rows(review_root=review_root, run_id_filter=run_id)
    with sqlite3.connect(str(db_path)) as conn:
        truth = _collect_truth_from_db(conn=conn, run_id_filter=run_id)
        truth_resolved: dict[tuple[str, str], bool | None] = truth["truth_resolved"]
        true_class_value_by_track: dict[tuple[str, str], str] = truth["true_class_value_by_track"]
        tracker_metrics = truth["tracker_metrics"]

        total_vlm = len(vlm_rows)
        retry_count = 0
        matched_truth_count = 0
        type_agree_count = 0
        class_agree_count = 0
        class_comparable_count = 0
        false_accept_count = 0
        false_reject_count = 0

        for row in vlm_rows:
            run_key = str(row.get("run_id") or "")
            track_key = str(row.get("track_id") or "")
            key = (run_key, track_key)
            predicted_is_type = _to_bool(row.get("is_type", "false"), default=False)
            ack_status = str(row.get("ack_status") or "").strip().lower()
            if ack_status == "retry_requested":
                retry_count += 1

            truth_value = truth_resolved.get(key)
            if truth_value is None:
                continue
            matched_truth_count += 1
            if predicted_is_type == truth_value:
                type_agree_count += 1
            if truth_value and not predicted_is_type:
                false_reject_count += 1
            if (not truth_value) and predicted_is_type:
                false_accept_count += 1

            human_true_class = true_class_value_by_track.get(key, "").strip().lower()
            target_class = str(row.get("target_class") or "").strip().lower()
            if human_true_class:
                class_comparable_count += 1
                if target_class == human_true_class:
                    class_agree_count += 1

        if class_comparable_count == 0:
            class_agree_count = type_agree_count
            class_comparable_count = matched_truth_count

        vlm_metrics = {
            "total_vlm_accepted_targets": total_vlm,
            "vlm_type_agreement_rate": round(_safe_div(type_agree_count, matched_truth_count), 6),
            "vlm_class_agreement_rate": round(_safe_div(class_agree_count, class_comparable_count), 6),
            "vlm_class_agreement_basis": "detector_target_class_proxy",
            "vlm_retry_rate": round(_safe_div(retry_count, total_vlm), 6),
            "vlm_false_accept_count": false_accept_count,
            "vlm_false_reject_count": false_reject_count,
        }

        metadata_metrics = _metadata_quality_metrics(
            conn=conn,
            run_id_filter=run_id,
            vlm_rows=vlm_rows,
        )

    return {
        "schema": "review_truth_comparison_v1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "review_root": str(review_root),
        "run_id_filter": run_id or "",
        "tracker_metrics": tracker_metrics,
        "vlm_metrics": vlm_metrics,
        "metadata_quality_metrics": metadata_metrics,
    }


def write_markdown(report: dict[str, Any], out_md: Path) -> None:
    t = report["tracker_metrics"]
    v = report["vlm_metrics"]
    m = report["metadata_quality_metrics"]
    lines = [
        "# Human Truth Comparison",
        "",
        f"- Run filter: `{report.get('run_id_filter') or 'ALL'}`",
        f"- Generated at: `{report.get('created_at_utc')}`",
        "",
        "## Tracker Truth Evaluation",
        f"- total_tracked_targets: `{t['total_tracked_targets']}`",
        f"- total_human_confirmed_trucks: `{t['total_human_confirmed_trucks']}`",
        f"- total_repeat_tracks: `{t['total_repeat_tracks']}`",
        f"- tracker_precision_like: `{t['tracker_precision_like']}`",
        "",
        "## VLM Agreement Evaluation",
        f"- total_vlm_accepted_targets: `{v['total_vlm_accepted_targets']}`",
        f"- vlm_type_agreement_rate: `{v['vlm_type_agreement_rate']}`",
        f"- vlm_class_agreement_rate: `{v['vlm_class_agreement_rate']}`",
        f"- vlm_class_agreement_basis: `{v['vlm_class_agreement_basis']}`",
        f"- vlm_retry_rate: `{v['vlm_retry_rate']}`",
        f"- vlm_false_accept_count: `{v['vlm_false_accept_count']}`",
        f"- vlm_false_reject_count: `{v['vlm_false_reject_count']}`",
        "",
        "## Metadata Quality Evaluation",
        f"- matched_id_count: `{m['matched_id_count']}`",
        f"- problematic_metadata_count: `{m['problematic_metadata_count']}`",
        f"- problematic_metadata_rate: `{m['problematic_metadata_rate']}`",
        f"- problematic_by_field: `{json.dumps(m['problematic_by_field'], sort_keys=True)}`",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare review run outputs against human_truth.sqlite labels.")
    parser.add_argument("--review-root", default="review-package", help="Review package root path.")
    parser.add_argument("--run-id", default="", help="Optional run_id filter.")
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional output Markdown path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    review_root = Path(args.review_root)
    if not review_root.is_absolute():
        review_root = (repo_root / review_root).resolve()

    report = run_comparison(review_root=review_root, run_id=args.run_id.strip())
    artifacts_dir = review_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stamp = utc_stamp()
    out_json = Path(args.output_json) if args.output_json else artifacts_dir / f"compare_against_human_truth_{stamp}.json"
    out_md = Path(args.output_md) if args.output_md else artifacts_dir / f"compare_against_human_truth_{stamp}.md"
    if not out_json.is_absolute():
        out_json = (repo_root / out_json).resolve()
    if not out_md.is_absolute():
        out_md = (repo_root / out_md).resolve()

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_md)
    print(f"[compare] wrote json={out_json}")
    print(f"[compare] wrote md={out_md}")


if __name__ == "__main__":
    main()
