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
    "frames": [24], "column": "", "out_dir": "", "message": "", "errors": [],
    "mode": "frames", "model": "",
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

# Some exports store bare container-relative blob paths instead of full URLs
# (e.g. "ANSA-McAL/.../videos/raw/Camera_20260826-085104.mp4"). They live in
# this container (Azure container names are lowercase, so the path's first
# segment can't be one). Override with AZURE_DEFAULT_CONTAINER if that changes.
_MEDIA_EXT = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".jpg", ".jpeg", ".png")


def default_container():
    return _env("AZURE_DEFAULT_CONTAINER") or "visionai"


def is_bare_blob_path(value):
    """A container-relative blob path — no scheme, has slashes, media suffix."""
    v = (value or "").strip()
    if not v or v.startswith("#") or "://" in v or " " in v or "/" not in v:
        return False
    return v.lower().endswith(_MEDIA_EXT)


def bare_to_url(value, account_name, container=None):
    """'<path>.mp4' -> 'https://<account>.blob.core.windows.net/<container>/<path>.mp4'"""
    container = container or default_container()
    return (f"https://{account_name}.blob.core.windows.net/"
            f"{container}/{value.strip().lstrip('/')}")


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


def parse_frames(spec):
    """"24, 48 120" (or a list) -> sorted unique non-negative ints. [] -> [24]."""
    vals = spec if isinstance(spec, (list, tuple)) else re.split(r"[,\s]+", str(spec or ""))
    out = set()
    for v in vals:
        v = str(v).strip()
        if v.isdigit():
            out.add(int(v))
    return sorted(out) or [24]


def _download_video(url):
    """Download the blob whole to a temp file and return its path.

    Streaming straight off HTTPS makes ffmpeg issue dozens of range requests
    (measured ~25s per video just to open+seek); one full GET plus a local
    decode is ~6x faster, and these event clips are only ~1 MB."""
    import requests
    import tempfile
    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="frx_")
    try:
        with os.fdopen(fd, "wb") as fh:
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        return tmp
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _read_frames(path, frame_idxs):
    """{frame_idx: BGR frame} for each requested index of a local video. A seek
    past the end of a short clip falls back to the last readable frame."""
    import cv2
    out = {}
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return out
        last = None
        for idx in sorted(frame_idxs):
            if idx > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                out[idx] = frame
                last = frame
                continue
            if last is None:
                # walk from the start once to find the clip's final frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    last = f
            if last is not None:
                out[idx] = last
        return out
    finally:
        cap.release()


def _grab_frame(url, frame_idx):
    """Single-frame convenience (used by Event Review): download + read."""
    tmp = _download_video(url)
    try:
        return _read_frames(tmp, [int(frame_idx)]).get(int(frame_idx))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


_CROP_PAD = 10        # px of context around each person box (like the PPE path)
_CROP_CONF = 0.4
_pred_lock = threading.Lock()   # one prediction at a time — YOLO isn't thread-safe


def list_models():
    """Every .pt in the model pool, [{label, value}], shallow paths first."""
    out = []
    root = paths.MODELS_DIR
    if root.is_dir():
        for pt in root.rglob("*.pt"):
            rel = str(pt.relative_to(root))
            out.append({"label": rel, "value": str(pt)})
    out.sort(key=lambda m: (m["label"].count("/"), m["label"].lower()))
    return out


def _person_crops(model, frame, stem, frame_idx, out_dir):
    """Detect persons in one frame and save each as a suite-named crop
    (<video>_<frame>_<cx-cy-w-h>.jpg). Returns crops saved."""
    import cv2
    with _pred_lock:
        result = model.predict(frame, conf=_CROP_CONF, verbose=False)[0]
    if result.boxes is None:
        return 0
    fh_, fw_ = frame.shape[:2]
    names = result.names or {}
    saved = 0
    for box in result.boxes:
        cls = int(box.cls[0])
        if str(names.get(cls, cls)).lower() != "person" and cls != 0:
            continue
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
        x1 = max(0, x1 - _CROP_PAD); y1 = max(0, y1 - _CROP_PAD)
        x2 = min(fw_, x2 + _CROP_PAD); y2 = min(fh_, y2 + _CROP_PAD)
        if x2 <= x1 or y2 <= y1:
            continue
        name = cropnames.yolo_crop_name(stem, frame_idx, (x1, y1, x2, y2), fw_, fh_)
        if cv2.imwrite(str(out_dir / name), frame[y1:y2, x1:x2]):
            saved += 1
    return saved


_WORKERS = 20  # parallel SAS-refresh + frame grabs (network/decode bound)


def _run(csv_path, column, frames, batch, mode, model_path):
    import cv2
    from concurrent.futures import ThreadPoolExecutor
    try:
        account_name, account_key = _sas_credentials()
    except Exception as exc:
        _set(running=False, done=True, error=str(exc))
        return
    model = None
    if mode == "crops":
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
        except Exception as exc:
            _set(running=False, done=True, error=f"Could not load model: {exc}")
            return

    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        text = fh.read()
    reader = csv.DictReader(io.StringIO(text.lstrip("\r\n")))
    reader.fieldnames = [h.lstrip("﻿").strip() for h in (reader.fieldnames or [])]
    urls = []
    promoted = 0
    for row in reader:
        v = (row.get(column) or "").strip()
        if v.lower().startswith("http"):
            urls.append(v)
        elif is_bare_blob_path(v):
            # bare container-relative path -> full URL (then signed like the rest)
            urls.append(bare_to_url(v, account_name))
            promoted += 1

    out_dir = paths.FRAMES_ROOT / batch
    out_dir.mkdir(parents=True, exist_ok=True)

    # One job per VIDEO carrying every frame still needed — the video is
    # downloaded once and all requested frames come out of that single copy.
    jobs = []
    seen = set()
    skipped = 0
    for url in urls:
        stem = cropnames.clean_video_name(unquote(urlsplit(url).path))
        if stem in seen:
            skipped += len(frames)  # duplicate video row in the CSV
            continue
        seen.add(stem)
        wanted = []
        for f in frames:
            if mode == "crops":
                # crop names depend on detections — always (re)process
                wanted.append((int(f), None))
                continue
            name = f"{stem}_{int(f):06d}.jpg"
            if (out_dir / name).exists():
                skipped += 1        # this frame is already extracted
            else:
                wanted.append((int(f), name))
        if wanted:
            jobs.append((url, stem, wanted))
    _set(total=len(jobs), out_dir=str(out_dir), skipped=skipped, frames=list(frames),
         message=f"Refreshing SAS for {len(jobs)} URL(s)…")

    counters = {"done": 0, "saved": 0, "failed": 0, "sas": 0, "cancelled": 0}
    errors = []
    clock = threading.Lock()

    # ---- Phase 1: refresh the SAS of the ENTIRE sheet up front ----
    # (local HMAC signing, no network — near-instant even for thousands of rows)
    refreshed = []
    for url, stem, wanted in jobs:
        if _stop_evt.is_set():
            counters["cancelled"] += 1
            continue
        try:
            fresh = refresh_sas_url(url, account_name, account_key)
            counters["sas"] += 1
            refreshed.append((fresh, stem, wanted))
        except Exception as exc:
            counters["failed"] += 1
            if len(errors) < 25:
                errors.append(f"{stem}: SAS refresh failed: {exc}")
        _set(sas_refreshed=counters["sas"], failed=counters["failed"],
             cancelled=counters["cancelled"], errors=list(errors),
             message=f"Refreshing SAS… {counters['sas']}/{len(jobs)}")
    _set(message=f"SAS refreshed for {counters['sas']} URL(s) — "
                 f"extracting with {_WORKERS} workers…")

    # ---- Phase 2: download each video ONCE, save every requested frame ----
    def work(fresh, stem, wanted):
        if _stop_evt.is_set():
            with clock:
                counters["cancelled"] += 1
                _set(cancelled=counters["cancelled"])
            return
        ok_frames = 0
        err = None
        try:
            tmp = _download_video(fresh)
            try:
                got = _read_frames(tmp, [f for f, _ in wanted])
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            for f, name in wanted:
                frame = got.get(f)
                if frame is None:
                    continue
                if mode == "crops":
                    ok_frames += _person_crops(model, frame, stem, f, out_dir)
                elif cv2.imwrite(str(out_dir / name), frame):
                    ok_frames += 1
            if ok_frames == 0 and not got:
                raise RuntimeError("could not read any frame (video unreadable?)")
            if mode != "crops" and ok_frames < len(wanted):
                err = f"{stem}: only {ok_frames}/{len(wanted)} frame(s) readable"
        except Exception as exc:
            err = err or f"{stem}: {exc}"
        with clock:
            counters["done"] += 1
            counters["saved"] += ok_frames
            if mode == "crops":
                # crop count is unrelated to frame count (0..N persons per
                # frame) — only a video that errored counts as failed
                counters["failed"] += 1 if err else 0
            else:
                counters["failed"] += len(wanted) - ok_frames
            if err and len(errors) < 25:
                errors.append(err)
            _set(processed=counters["done"], saved=counters["saved"],
                 failed=counters["failed"], errors=list(errors),
                 message=f"({counters['done']}/{len(jobs)}) videos — "
                         f"extracting with {_WORKERS} workers…")

    with ThreadPoolExecutor(max_workers=_WORKERS) as ex:
        for f in [ex.submit(work, *j) for j in refreshed]:
            f.result()   # work() swallows its own errors; this just waits

    saved, failed = counters["saved"], counters["failed"]
    stopped = _stop_evt.is_set()
    flist = ",".join(str(f) for f in frames)
    thing = "person crop(s)" if mode == "crops" else "frame(s)"
    msg = ("Stopped — saved" if stopped else "Saved") + \
        f" {saved} {thing} (frames {flist}) to {out_dir}."
    msg += f" SAS refreshed for {counters['sas']} URL(s)."
    if skipped:
        msg += f" {skipped} skipped (duplicate/already there)."
    if counters["cancelled"]:
        msg += f" {counters['cancelled']} cancelled."
    if failed:
        msg += f" {failed} failed."
    _set(running=False, done=True, stopping=False, saved=saved, failed=failed,
         processed=counters["done"] + counters["cancelled"],
         skipped=skipped, errors=list(errors), message=msg)


def start(csv_path, column, frames_spec, batch, mode="frames", model_path=""):
    """Kick off an extraction. ``frames_spec`` is one or more frame numbers
    ("24" or "24, 48, 120" or a list). ``mode`` is "frames" (save whole
    frames) or "crops" (run ``model_path`` on each frame and save each person
    as a suite-named crop). Returns an error string, or None."""
    if not os.path.isfile(csv_path):
        return "No uploaded CSV found — drop the file again."
    mode = "crops" if str(mode) == "crops" else "frames"
    if mode == "crops" and not (model_path and os.path.isfile(model_path)):
        return "Pick a model to crop persons with."
    batch = re.sub(r"[^A-Za-z0-9._-]+", "_", str(batch).strip()) or "batch"
    frames = parse_frames(frames_spec)
    with _lock:
        if _state["running"]:
            return "An extraction is already running."
        _stop_evt.clear()
        _state.update({"running": True, "done": False, "error": "", "errors": [],
                       "stopping": False, "total": 0, "processed": 0, "saved": 0,
                       "failed": 0, "skipped": 0, "sas_refreshed": 0, "cancelled": 0,
                       "frames": frames, "column": str(column),
                       "mode": mode, "model": os.path.basename(str(model_path or "")),
                       "out_dir": "", "message": "Starting…"})
    threading.Thread(target=_run,
                     args=(str(csv_path), str(column), frames, batch, mode,
                           str(model_path or "")),
                     daemon=True).start()
    return None
