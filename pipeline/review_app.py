#!/usr/bin/env python3
"""
Lightweight local review app for human truth labeling.

Features:
- Boots/updates review-package/human_truth.sqlite schema.
- Ingests run metadata CSV files into review_items.
- Serves a keyboard-first browser UI for labeling:
  q=wrong_class, w=true_class, e=repeat, r=bad_crop
- Supports filters by run, item type, target class, source video.
- Saves metadata highlights/comments into review_highlights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse


ALLOWED_LABEL_KEYS = {"wrong_class", "true_class", "repeat", "bad_crop"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_relpath(relpath: str) -> str:
    return relpath.replace("\\", "/").lstrip("/")


def _ensure_within(base: Path, relpath: str) -> Path:
    candidate = (base / _sanitize_relpath(relpath)).resolve()
    base_resolved = base.resolve()
    if not str(candidate).startswith(str(base_resolved)):
        raise ValueError("Path escapes review root.")
    return candidate


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS review_items (
              id INTEGER PRIMARY KEY,
              run_id TEXT NOT NULL,
              item_type TEXT NOT NULL,
              source_video TEXT NOT NULL,
              frame_index INTEGER NOT NULL,
              track_id TEXT NOT NULL,
              target_class TEXT,
              image_relpath TEXT NOT NULL,
              metadata_relpath TEXT,
              created_at_utc TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS ux_review_items_identity
            ON review_items(run_id, item_type, source_video, frame_index, track_id, image_relpath);

            CREATE TABLE IF NOT EXISTS review_labels (
              id INTEGER PRIMARY KEY,
              review_item_id INTEGER NOT NULL,
              reviewer TEXT,
              label_key TEXT NOT NULL,
              label_value TEXT,
              created_at_utc TEXT NOT NULL,
              FOREIGN KEY(review_item_id) REFERENCES review_items(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_review_labels_item_id ON review_labels(review_item_id);

            CREATE TABLE IF NOT EXISTS review_highlights (
              id INTEGER PRIMARY KEY,
              review_item_id INTEGER NOT NULL,
              reviewer TEXT,
              field_name TEXT,
              highlight_text TEXT,
              comment_text TEXT,
              created_at_utc TEXT NOT NULL,
              FOREIGN KEY(review_item_id) REFERENCES review_items(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_review_highlights_item_id ON review_highlights(review_item_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def ingest_review_items(review_root: Path, db_path: Path) -> dict[str, int]:
    conn = sqlite3.connect(str(db_path))
    inserted = 0
    skipped = 0
    try:
        run_dirs = sorted((review_root / "runs").glob("*"), key=lambda p: p.name)
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            metadata_dir = run_dir / "metadata"
            new_tracks_csv = metadata_dir / "new_tracks.csv"
            accepted_csv = metadata_dir / "vlm_accepted_targets.csv"
            specs = [
                ("new_track", new_tracks_csv),
                ("vlm_accepted_target", accepted_csv),
            ]
            for item_type, csv_path in specs:
                rows = _read_rows(csv_path)
                metadata_relpath = str(csv_path.resolve().relative_to(review_root.resolve())).replace("\\", "/")
                for row in rows:
                    image_relpath = _sanitize_relpath(row.get("image_relpath", ""))
                    if not image_relpath:
                        skipped += 1
                        continue
                    image_abs = _ensure_within(review_root, image_relpath)
                    if not image_abs.exists():
                        skipped += 1
                        continue
                    run_id = str(row.get("run_id") or run_dir.name).strip()
                    source_video = str(row.get("source_video") or "").strip()
                    track_id = str(row.get("track_id") or "").strip()
                    frame_index_raw = str(row.get("frame_index") or "0").strip()
                    try:
                        frame_index = int(frame_index_raw)
                    except ValueError:
                        frame_index = 0
                    if not (run_id and source_video and track_id):
                        skipped += 1
                        continue
                    target_class = str(row.get("target_class") or "").strip()
                    cur = conn.execute(
                        """
                        INSERT OR IGNORE INTO review_items
                        (
                          run_id, item_type, source_video, frame_index, track_id,
                          target_class, image_relpath, metadata_relpath, created_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            item_type,
                            source_video,
                            frame_index,
                            track_id,
                            target_class,
                            image_relpath,
                            metadata_relpath,
                            utc_now_iso(),
                        ),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                    else:
                        skipped += 1
        conn.commit()
        return {"inserted": inserted, "skipped": skipped}
    finally:
        conn.close()


def _fetch_filters(conn: sqlite3.Connection) -> dict[str, list[str]]:
    conn.row_factory = sqlite3.Row
    results: dict[str, list[str]] = {}
    for col in ("run_id", "item_type", "target_class", "source_video"):
        rows = conn.execute(
            f"SELECT DISTINCT {col} AS value FROM review_items WHERE COALESCE({col}, '') != '' ORDER BY value"
        ).fetchall()
        results[col] = [str(r["value"]) for r in rows]
    return results


def _fetch_next_unlabeled(conn: sqlite3.Connection, filters: dict[str, str]) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    where_clauses = ["label_presence.review_item_id IS NULL"]
    params: list[Any] = []

    for key in ("run_id", "item_type", "target_class", "source_video"):
        value = (filters.get(key) or "").strip()
        if value:
            where_clauses.append(f"ri.{key} = ?")
            params.append(value)

    sql = f"""
    SELECT
      ri.id, ri.run_id, ri.item_type, ri.source_video, ri.frame_index,
      ri.track_id, ri.target_class, ri.image_relpath, ri.metadata_relpath
    FROM review_items ri
    LEFT JOIN (
      SELECT DISTINCT review_item_id FROM review_labels
    ) AS label_presence
      ON label_presence.review_item_id = ri.id
    WHERE {' AND '.join(where_clauses)}
    ORDER BY ri.run_id DESC, ri.frame_index ASC, ri.id ASC
    LIMIT 1
    """
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return None
    payload = {k: row[k] for k in row.keys()}
    payload["image_url"] = f"/image?relpath={quote(str(payload['image_relpath']))}"
    payload["labels"] = [
        dict(label_row)
        for label_row in conn.execute(
            """
            SELECT id, reviewer, label_key, label_value, created_at_utc
            FROM review_labels
            WHERE review_item_id = ?
            ORDER BY id ASC
            """,
            (payload["id"],),
        ).fetchall()
    ]
    payload["highlights"] = [
        dict(h_row)
        for h_row in conn.execute(
            """
            SELECT id, reviewer, field_name, highlight_text, comment_text, created_at_utc
            FROM review_highlights
            WHERE review_item_id = ?
            ORDER BY id ASC
            """,
            (payload["id"],),
        ).fetchall()
    ]
    return payload


def _fetch_stats(conn: sqlite3.Connection) -> dict[str, int]:
    conn.row_factory = sqlite3.Row
    total_items = int(conn.execute("SELECT COUNT(*) AS c FROM review_items").fetchone()["c"])
    labeled_items = int(
        conn.execute("SELECT COUNT(DISTINCT review_item_id) AS c FROM review_labels").fetchone()["c"]
    )
    total_labels = int(conn.execute("SELECT COUNT(*) AS c FROM review_labels").fetchone()["c"])
    total_highlights = int(conn.execute("SELECT COUNT(*) AS c FROM review_highlights").fetchone()["c"])
    unlabeled_items = max(0, total_items - labeled_items)
    return {
        "total_items": total_items,
        "labeled_items": labeled_items,
        "unlabeled_items": unlabeled_items,
        "total_labels": total_labels,
        "total_highlights": total_highlights,
    }


INDEX_HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Review App</title>
  <style>
    :root {
      --bg: #f4f2ee;
      --panel: #fffdf8;
      --ink: #252422;
      --muted: #67615b;
      --accent: #1f6f8b;
      --warn: #b66a00;
      --line: #ded7cd;
    }
    body { margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: linear-gradient(160deg, #f7f3ec, #efebe3 60%, #ebe7e0); }
    .wrap { max-width: 1260px; margin: 0 auto; padding: 16px; }
    .bar { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 8px; align-items: end; margin-bottom: 12px; }
    .bar label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 4px; }
    select, input, button, textarea { width: 100%; box-sizing: border-box; font: inherit; }
    select, input, textarea { padding: 8px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    button { padding: 9px 10px; border: 1px solid #0f5168; background: var(--accent); color: #fff; border-radius: 8px; cursor: pointer; }
    button.secondary { background: #fff; color: var(--ink); border-color: var(--line); }
    .hint { font-size: 12px; color: var(--muted); margin-bottom: 10px; }
    .grid { display: grid; grid-template-columns: 1.2fr 1fr; gap: 12px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; min-height: 420px; }
    .imgbox { display:flex; justify-content:center; align-items:center; min-height: 380px; background: #ece8e1; border-radius: 8px; overflow: hidden; }
    .imgbox img { max-width: 100%; max-height: 70vh; object-fit: contain; }
    .meta-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .meta-table td { border-top: 1px solid var(--line); padding: 6px 4px; vertical-align: top; }
    .meta-table td.k { width: 36%; color: var(--muted); }
    .stats { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .row { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:8px; margin-top:8px; }
    .warn { color: var(--warn); font-weight: 600; }
    .ok { color: #2f7a28; font-weight: 600; }
    .list { margin-top: 8px; font-size: 12px; color: var(--ink); }
    .list div { border-top: 1px dashed var(--line); padding: 5px 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="bar">
      <div><label>Run</label><select id="run_id"></select></div>
      <div><label>Item Type</label><select id="item_type"></select></div>
      <div><label>Target Class</label><select id="target_class"></select></div>
      <div><label>Source Video</label><select id="source_video"></select></div>
      <div><label>Reviewer</label><input id="reviewer" placeholder="name or initials" /></div>
      <div><button id="btn-next">Next Unlabeled</button></div>
    </div>
    <div class="hint">
      Shortcuts: <b>Q</b>=wrong_class, <b>W</b>=true_class, <b>E</b>=repeat, <b>R</b>=bad_crop.
      Double-click a metadata row to add a highlight + comment.
    </div>
    <div id="status" class="stats"></div>
    <div class="grid">
      <div class="panel">
        <div class="imgbox"><img id="item-image" alt="review item image"/></div>
      </div>
      <div class="panel">
        <table class="meta-table" id="meta-table"></table>
        <div class="row">
          <button class="secondary" id="btn-wrong">Q wrong_class</button>
          <button class="secondary" id="btn-true">W true_class</button>
          <button class="secondary" id="btn-repeat">E repeat</button>
          <button class="secondary" id="btn-bad">R bad_crop</button>
        </div>
        <div style="margin-top:8px;">
          <label style="font-size:12px;color:#67615b;">Optional label note</label>
          <textarea id="label-value" rows="2" placeholder="optional"></textarea>
        </div>
        <div class="list" id="highlights"></div>
      </div>
    </div>
  </div>
<script>
let currentItem = null;
const filters = ["run_id", "item_type", "target_class", "source_video"];

function setStatus(text, good=false) {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = good ? "stats ok" : "stats";
}

function selectedFilters() {
  const out = {};
  for (const f of filters) out[f] = document.getElementById(f).value || "";
  return out;
}

function fillSelect(id, values) {
  const sel = document.getElementById(id);
  const prev = sel.value;
  sel.innerHTML = "";
  const all = document.createElement("option");
  all.value = "";
  all.textContent = "All";
  sel.appendChild(all);
  for (const v of values || []) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  }
  if ([...sel.options].some(o => o.value === prev)) sel.value = prev;
}

async function loadFilters() {
  const r = await fetch("/api/filters");
  const data = await r.json();
  fillSelect("run_id", data.run_id || []);
  fillSelect("item_type", data.item_type || []);
  fillSelect("target_class", data.target_class || []);
  fillSelect("source_video", data.source_video || []);
}

function renderMeta(item) {
  const table = document.getElementById("meta-table");
  table.innerHTML = "";
  const fields = [
    ["id", item.id], ["run_id", item.run_id], ["item_type", item.item_type],
    ["source_video", item.source_video], ["frame_index", item.frame_index],
    ["track_id", item.track_id], ["target_class", item.target_class || ""],
    ["image_relpath", item.image_relpath], ["metadata_relpath", item.metadata_relpath || ""]
  ];
  for (const [k,v] of fields) {
    const tr = document.createElement("tr");
    tr.dataset.field = k;
    tr.dataset.value = String(v ?? "");
    tr.innerHTML = `<td class="k">${k}</td><td>${String(v ?? "")}</td>`;
    tr.ondblclick = async () => {
      if (!currentItem) return;
      const comment = prompt(`Comment for ${k}:`, "");
      if (comment === null) return;
      const reviewer = document.getElementById("reviewer").value || "";
      const payload = {
        review_item_id: currentItem.id,
        reviewer: reviewer,
        field_name: k,
        highlight_text: String(v ?? ""),
        comment_text: comment
      };
      const res = await fetch("/api/highlight", {method:"POST", headers: {"content-type":"application/json"}, body: JSON.stringify(payload)});
      if (!res.ok) {
        const txt = await res.text();
        setStatus(`Highlight failed: ${txt}`);
        return;
      }
      await loadNext(false);
      setStatus("Highlight saved", true);
    };
    table.appendChild(tr);
  }
}

function renderHighlights(item) {
  const el = document.getElementById("highlights");
  const list = item.highlights || [];
  if (!list.length) { el.textContent = "No highlights yet."; return; }
  el.innerHTML = "<b>Highlights</b>" + list.map(h =>
    `<div><b>${h.field_name || "field"}</b>: ${h.highlight_text || ""}<br/>${h.comment_text || ""}</div>`
  ).join("");
}

async function loadNext(showNoItems=true) {
  const q = new URLSearchParams(selectedFilters());
  const res = await fetch(`/api/next?${q.toString()}`);
  const data = await res.json();
  if (!data.item) {
    currentItem = null;
    document.getElementById("item-image").src = "";
    document.getElementById("meta-table").innerHTML = "";
    document.getElementById("highlights").innerHTML = "";
    if (showNoItems) setStatus("No unlabeled items for current filters.");
  } else {
    currentItem = data.item;
    document.getElementById("item-image").src = currentItem.image_url;
    renderMeta(currentItem);
    renderHighlights(currentItem);
    setStatus(`Reviewing item ${currentItem.id} | track=${currentItem.track_id} | frame=${currentItem.frame_index}`, true);
  }
  const stats = data.stats || {};
  const extra = `  Total=${stats.total_items||0}  Labeled=${stats.labeled_items||0}  Unlabeled=${stats.unlabeled_items||0}  Highlights=${stats.total_highlights||0}`;
  document.getElementById("status").textContent += extra;
}

async function saveLabel(labelKey) {
  if (!currentItem) return;
  const reviewer = document.getElementById("reviewer").value || "";
  const labelValue = document.getElementById("label-value").value || "";
  const payload = {
    review_item_id: currentItem.id,
    reviewer: reviewer,
    label_key: labelKey,
    label_value: labelValue
  };
  const res = await fetch("/api/label", {method:"POST", headers: {"content-type":"application/json"}, body: JSON.stringify(payload)});
  if (!res.ok) {
    const txt = await res.text();
    setStatus(`Label failed: ${txt}`);
    return;
  }
  document.getElementById("label-value").value = "";
  await loadNext(false);
  setStatus(`Saved label: ${labelKey}`, true);
}

document.getElementById("btn-next").onclick = () => loadNext(true);
document.getElementById("btn-wrong").onclick = () => saveLabel("wrong_class");
document.getElementById("btn-true").onclick = () => saveLabel("true_class");
document.getElementById("btn-repeat").onclick = () => saveLabel("repeat");
document.getElementById("btn-bad").onclick = () => saveLabel("bad_crop");
for (const id of filters) document.getElementById(id).onchange = () => loadNext(true);

window.addEventListener("keydown", (ev) => {
  if (["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName)) return;
  const k = ev.key.toLowerCase();
  if (k === "q") saveLabel("wrong_class");
  if (k === "w") saveLabel("true_class");
  if (k === "e") saveLabel("repeat");
  if (k === "r") saveLabel("bad_crop");
  if (k === "n") loadNext(true);
});

async function bootstrap() {
  await loadFilters();
  await loadNext(true);
}
bootstrap();
</script>
</body>
</html>
"""


class ReviewAppHandler(BaseHTTPRequestHandler):
    server_version = "ReviewApp/1.0"

    @property
    def app_state(self) -> dict[str, Any]:
        return self.server.app_state  # type: ignore[attr-defined]

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
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

    def _read_json_body(self) -> dict[str, Any]:
        content_len = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(content_len) if content_len > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _open_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.app_state["db_path"]))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(INDEX_HTML, content_type="text/html; charset=utf-8")
            return

        if parsed.path == "/api/filters":
            with self._open_conn() as conn:
                self._send_json(_fetch_filters(conn))
            return

        if parsed.path == "/api/next":
            params = {k: (v[0] if v else "") for k, v in parse_qs(parsed.query).items()}
            with self._open_conn() as conn:
                item = _fetch_next_unlabeled(conn, params)
                stats = _fetch_stats(conn)
            self._send_json({"item": item, "stats": stats})
            return

        if parsed.path == "/api/stats":
            with self._open_conn() as conn:
                self._send_json(_fetch_stats(conn))
            return

        if parsed.path == "/image":
            params = parse_qs(parsed.query)
            relpath = unquote((params.get("relpath") or [""])[0])
            if not relpath:
                self._send_text("Missing relpath", status=HTTPStatus.BAD_REQUEST)
                return
            try:
                image_path = _ensure_within(self.app_state["review_root"], relpath)
            except ValueError as exc:
                self._send_text(str(exc), status=HTTPStatus.BAD_REQUEST)
                return
            if not image_path.exists():
                self._send_text("Image not found", status=HTTPStatus.NOT_FOUND)
                return
            ext = image_path.suffix.lower()
            ctype = "image/jpeg" if ext in {".jpg", ".jpeg"} else "image/png"
            body = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self._send_text("Not found", status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_text("Invalid JSON", status=HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/label":
            review_item_id = int(payload.get("review_item_id") or 0)
            label_key = str(payload.get("label_key") or "").strip()
            label_value = str(payload.get("label_value") or "")
            reviewer = str(payload.get("reviewer") or "").strip()
            if review_item_id <= 0:
                self._send_text("review_item_id must be > 0", status=HTTPStatus.BAD_REQUEST)
                return
            if label_key not in ALLOWED_LABEL_KEYS:
                self._send_text(f"label_key must be one of {sorted(ALLOWED_LABEL_KEYS)}", status=HTTPStatus.BAD_REQUEST)
                return
            with self._open_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO review_labels (review_item_id, reviewer, label_key, label_value, created_at_utc)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (review_item_id, reviewer, label_key, label_value, utc_now_iso()),
                )
                conn.commit()
            self._send_json({"ok": True})
            return

        if parsed.path == "/api/highlight":
            review_item_id = int(payload.get("review_item_id") or 0)
            reviewer = str(payload.get("reviewer") or "").strip()
            field_name = str(payload.get("field_name") or "").strip()
            highlight_text = str(payload.get("highlight_text") or "")
            comment_text = str(payload.get("comment_text") or "")
            if review_item_id <= 0:
                self._send_text("review_item_id must be > 0", status=HTTPStatus.BAD_REQUEST)
                return
            with self._open_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO review_highlights
                    (review_item_id, reviewer, field_name, highlight_text, comment_text, created_at_utc)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (review_item_id, reviewer, field_name, highlight_text, comment_text, utc_now_iso()),
                )
                conn.commit()
            self._send_json({"ok": True})
            return

        self._send_text("Not found", status=HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("REVIEW_APP_QUIET", "").strip().lower() in {"1", "true", "yes"}:
            return
        super().log_message(fmt, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight local review app for human truth labeling.")
    parser.add_argument("--review-root", default="review-package", help="Review package root (default: review-package).")
    parser.add_argument("--db-path", default="", help="SQLite path (default: <review-root>/human_truth.sqlite).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    review_root = Path(args.review_root)
    if not review_root.is_absolute():
        review_root = (repo_root / review_root).resolve()
    db_path = Path(args.db_path) if args.db_path else (review_root / "human_truth.sqlite")
    if not db_path.is_absolute():
        db_path = (repo_root / db_path).resolve()

    review_root.mkdir(parents=True, exist_ok=True)
    init_db(db_path)
    ingest_result = ingest_review_items(review_root=review_root, db_path=db_path)

    server = ThreadingHTTPServer((args.host, args.port), ReviewAppHandler)
    server.app_state = {  # type: ignore[attr-defined]
        "review_root": review_root,
        "db_path": db_path,
    }
    print(
        f"[review_app] host={args.host} port={args.port} review_root={review_root} db={db_path} "
        f"ingested_inserted={ingest_result['inserted']} ingested_skipped={ingest_result['skipped']}",
        flush=True,
    )
    print(f"[review_app] open: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
