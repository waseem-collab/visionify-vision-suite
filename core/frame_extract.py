#!/usr/bin/env python3
"""
Extract one frame per video from a CSV of Azure blob video URLs.

The Frame Extractor card: the user drops a CSV, picks the column holding the
video URLs and a frame number (default 24). For every row this module first
mints a FRESH SAS token for the URL (the stored ones expire — same approach as
the standalone sas_update/refresh_sas.py tool), then opens the video straight
from blob storage, seeks to the requested frame and saves it as

    <video-stem>_<frame:06d>.jpg          (the suite's naming scheme)

under ``paths.FRAMES_ROOT/<batch name>/``. Runs on a background thread with a
polled status, mirroring label_import / cvat_sync. Best-effort per row.
"""
import csv
import io
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit, unquote

from core import convex_client, cropnames, paths

_lock = threading.Lock()
_state = {
    "running": False, "done": False, "error": "", "stopping": False,
    "total": 0, "processed": 0, "saved": 0, "failed": 0, "skipped": 0,
    "sas_refreshed": 0, "cancelled": 0,
    "frame": 24, "column": "", "out_dir": "", "message": "", "errors": [],
}

_stop_evt = threading.Event()


def stop():
    """Ask a running extraction to stop (in-flight grabs finish, the rest are
    cancelled). Returns an error string, or None."""
    with _lock:
        if not _state["running"]:
            return "Nothing is running."
        _state.update(stopping=True, message="Stopping…")
    _stop_evt.set()
    return None

_SAS_EXPIRY_HOURS = 12  # short-lived: these URLs are used immediately


def status():
    with _lock:
        return dict(_state)


def _set(**kw):
    with _lock:
        _state.update(kw)


def _env(key):
    convex_client.convex_url()  # side-effect: loads .env into os.environ
    return (os.environ.get(key) or "").strip()


def _sas_credentials():
    """(account_name, account_key) from MODELS_AZURE_BLOB_CONNECTION_STRING."""
    conn = _env("MODELS_AZURE_BLOB_CONNECTION_STRING")
    if not conn:
        raise RuntimeError("MODELS_AZURE_BLOB_CONNECTION_STRING missing in .env")
    parts = dict(seg.split("=", 1) for seg in conn.split(";") if "=" in seg)
    name, key = parts.get("AccountName"), parts.get("AccountKey")
    if not (name and key):
        raise RuntimeError("Connection string lacks AccountName/AccountKey")
    return name, key


def refresh_sas_url(url, account_name, account_key):
    """Blob URL (stale SAS or none) -> same URL with a freshly minted SAS."""
    from azure.storage.blob import BlobSasPermissions, generate_blob_sas
    parsed = urlsplit(url)
    if not parsed.netloc.lower().startswith(f"{account_name}.".lower()):
        raise ValueError(f"URL account does not match '{account_name}'")
    path = parsed.path.lstrip("/")
    if "/" not in path:
        raise ValueError("URL has no container/blob path")
    container, blob = path.split("/", 1)
    token = generate_blob_sas(
        account_name=account_name, account_key=account_key,
        container_name=container, blob_name=unquote(blob),
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=_SAS_EXPIRY_HOURS),
    )
    base = urlunsplit((parsed.scheme, parsed.netloc, f"/{container}/{blob}", "", ""))
    return f"{base}?{token}"


def read_headers(csv_text):
    """Header names of an uploaded CSV (BOM/whitespace-normalised)."""
    reader = csv.reader(io.StringIO(csv_text.lstrip("\r\n")))
    headers = next(reader, None) or []
    return [h.lstrip("﻿").strip() for h in headers]


def _grab_frame(url, frame_idx):
    """Download the blob to a temp file and read frame ``frame_idx`` locally.

    Streaming straight off HTTPS makes ffmpeg issue dozens of range requests
    (measured ~25s per video just to open+seek); one full GET plus a local
    decode is ~6x faster, and these event clips are only ~1 MB."""
    import cv2
    import requests
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="frx_")
    try:
        with os.fdopen(fd, "wb") as fh:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        cap = cv2.VideoCapture(tmp)
        try:
            if not cap.isOpened():
                return None
            if frame_idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
            # Seek can overshoot short clips — fall back to the last readable frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            last = None
            for _ in range(frame_idx + 1):
                ok, f = cap.read()
                if not ok:
                    break
                last = f
            return last
        finally:
            cap.release()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


_WORKERS = 20  # parallel SAS-refresh + frame grabs (network/decode bound)


def _run(csv_path, column, frame_idx, batch):
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    try:
        account_name, account_key = _sas_credentials()
    except Exception as exc:
        _set(running=False, done=True, error=str(exc))
        return

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        text = fh.read()
    reader = csv.DictReader(io.StringIO(text.lstrip("\r\n")))
    reader.fieldnames = [h.lstrip("﻿").strip() for h in (reader.fieldnames or [])]
    urls = []
    for row in reader:
        v = (row.get(column) or "").strip()
        if v.lower().startswith("http"):
            urls.append(v)

    out_dir = paths.FRAMES_ROOT / batch
    out_dir.mkdir(parents=True, exist_ok=True)

    # Name and dedupe up front so the workers never race on "have we done this
    # video already" — each job is an independent (url, filename) pair.
    jobs = []
    seen = set()
    skipped = 0
    for url in urls:
        stem = cropnames.clean_video_name(unquote(urlsplit(url).path))
        name = f"{stem}_{int(frame_idx):06d}.jpg"
        if name in seen or (out_dir / name).exists():
            skipped += 1  # duplicate video in the CSV, or already extracted
            continue
        seen.add(name)
        jobs.append((url, name, stem))
    _set(total=len(urls), out_dir=str(out_dir), skipped=skipped,
         message=f"{len(jobs)} video(s) to extract with {_WORKERS} workers")

    counters = {"done": 0, "saved": 0, "failed": 0, "sas": 0, "cancelled": 0}
    errors = []
    clock = threading.Lock()

    def work(url, name, stem):
        if _stop_evt.is_set():
            with clock:
                counters["cancelled"] += 1
                _set(cancelled=counters["cancelled"])
            return
        err = None
        try:
            fresh = refresh_sas_url(url, account_name, account_key)
            with clock:
                counters["sas"] += 1
                _set(sas_refreshed=counters["sas"])
            frame = _grab_frame(fresh, int(frame_idx))
            if frame is None:
                raise RuntimeError("could not read the frame (video unreadable?)")
            if not cv2.imwrite(str(out_dir / name), frame):
                raise RuntimeError("failed to write the image")
        except Exception as exc:
            err = f"{stem}: {exc}"
        with clock:
            counters["done"] += 1
            counters["failed" if err else "saved"] += 1
            if err and len(errors) < 25:
                errors.append(err)
            _set(processed=skipped + counters["done"], saved=counters["saved"],
                 failed=counters["failed"], errors=list(errors),
                 message=f"({skipped + counters['done']}/{len(urls)}) "
                         f"extracting with {_WORKERS} workers…")

    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for f in [ex.submit(work, *j) for j in jobs]:
            f.result()   # work() swallows its own errors; this just waits

    saved, failed = counters["saved"], counters["failed"]
    stopped = _stop_evt.is_set()
    msg = ("Stopped — saved" if stopped else "Saved") + f" {saved} frame(s) to {out_dir}."
    msg += f" SAS refreshed for {counters['sas']} URL(s)."
    if skipped:
        msg += f" {skipped} skipped (duplicate/already there)."
    if counters["cancelled"]:
        msg += f" {counters['cancelled']} cancelled."
    if failed:
        msg += f" {failed} failed."
    _set(running=False, done=True, stopping=False, saved=saved, failed=failed,
         processed=skipped + counters["done"] + counters["cancelled"],
         skipped=skipped, errors=list(errors), message=msg)


def start(csv_path, column, frame_idx, batch):
    """Kick off an extraction. Returns an error string, or None if started."""
    if not os.path.isfile(csv_path):
        return "No uploaded CSV found — drop the file again."
    batch = re.sub(r"[^A-Za-z0-9._-]+", "_", str(batch).strip()) or "batch"
    try:
        frame_idx = max(0, int(frame_idx))
    except (TypeError, ValueError):
        frame_idx = 24
    with _lock:
        if _state["running"]:
            return "An extraction is already running."
        _stop_evt.clear()
        _state.update({"running": True, "done": False, "error": "", "errors": [],
                       "stopping": False, "total": 0, "processed": 0, "saved": 0,
                       "failed": 0, "skipped": 0, "sas_refreshed": 0, "cancelled": 0,
                       "frame": frame_idx, "column": str(column),
                       "out_dir": "", "message": "Starting…"})
    threading.Thread(target=_run, args=(str(csv_path), str(column), frame_idx, batch),
                     daemon=True).start()
    return None
