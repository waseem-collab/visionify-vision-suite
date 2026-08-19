#!/usr/bin/env python3
import atexit
import base64
import hashlib
import html
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

# Repo root on the path so this module can be imported directly as well as via
# the suite launcher (python3 apps/web_app.py still works).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import cropnames, paths, ports
# Imported under an alias: `theme` is already a local variable name inside the
# page renderer, and shadowing the module there would be a nasty surprise.
from core import theme as suite_theme

paths.ensure_dirs()


def _load_dotenv():
    """Load KEY=VALUE lines from the repo-root .env into os.environ (no dependency)."""
    env_path = paths.ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv()

import cv2
from flask import Flask, Response, jsonify, redirect, request, send_file
from ultralytics import YOLO

from PPE import ppe_inference
from SM import sm_cropper


APP_TITLE = "Inference Web App (PPE + SM)"
APP_SETTINGS_FILE = paths.APP_SETTINGS_FILE
MODELS_DIR = paths.MODELS_DIR
NONE_MODEL_VALUE = "__none__"
PERSON_CROP_PADDING = 10
CROPS_ROOT = paths.CROPS_ROOT


app = Flask(__name__)


@app.after_request
def _add_cors_headers(resp):
    # Allow the React dev server (and the camera-manager app) to call these
    # endpoints and embed the MJPEG stream cross-origin.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/<path:_rest>", methods=["OPTIONS"])
def _cors_preflight(_rest):
    return ("", 204)


# state_lock guards `state`/`runtime` and OpenCV capture access; held only briefly.
# inference_lock serializes the heavy YOLO predict calls so they never run under
# state_lock — that keeps control/seek/status endpoints snappy while playing.
state_lock = threading.Lock()
inference_lock = threading.Lock()
model_cache = {}

state = {
    "person_model_path": ppe_inference.DEFAULT_PERSON_MODEL,
    "ppe_model_path": ppe_inference.DEFAULT_MODEL_PATH,
    "sm_model_path": sm_cropper.DEFAULT_SINGLE_MODEL_PATH,
    "video_folder": str(Path.cwd()),
    "selected_video": "",
    # A remote video/stream URL (http/https/rtsp/rtmp). When set, it is streamed
    # in place and used as the source instead of a local file.
    "video_url": "",
    # Which source is active: "file" (local folder + file) or "url" (stream). Only
    # the active one is used; the other is inert even if it still holds a value.
    "source_mode": "file",
    "person_conf": 0.4,
    "ppe_conf": 0.4,
    "sm_conf": 0.7,
    "frame_step": 1,
    "simulate_realtime": True,
    "ppe_inside_person": True,
    # Which PPE class names to show as per-person status chips. None = all classes
    # of the selected model. A list means only those classes are shown.
    "ppe_classes": None,
    # Same idea for the SM (single) model's own classes. None = all.
    "sm_classes": None,
    "playing": False,
}

runtime = {
    "cap": None,
    "active_video": "",
    "fps": 25.0,
    "total_frames": 0,
    "frame_idx": -1,
    "displayed_count": 0,
    "started_at": time.time(),
    "last_jpg": None,
    "last_error": "",
    "last_raw_frame": None,
    "last_person_boxes": [],
    "last_action": "",
    "manual_boxes_by_frame": {},
    # Bumped on every seek / video-change / stop. Buffered frames tagged with an
    # older epoch are discarded so the stream never shows stale prefetched frames.
    "epoch": 0,
}

# --- Persistent annotation cache (YouTube-style multi-segment "loaded" bar) ---
# Each frame is annotated AT MOST ONCE per configuration and kept in frame_cache.
# Revisiting an already-annotated frame is instant — we only re-annotate when the
# configuration changes (model / conf / step / video → cache cleared) or when a
# frame was dropped by the LRU memory cap. The seek bar shows every cached region
# as a grey segment.
#
# Locking: cache_lock guards frame_cache + cache_state; never held with state_lock.
# The worker owns _worker_cap exclusively.
cache_lock = threading.Lock()
DISPLAY_PACE_FACTOR = 1.0           # playback pacing vs real time (1.0 = real time)
# LRU cap on annotated frames kept in RAM. Each entry is a JPEG (~50–200 KB), so
# this bounds the cache to a few hundred MB. Kept well below the old 8000 because
# on a CPU-only machine holding thousands of frames pushes RAM into swap and
# freezes the box. With the bounded prerender below, the cache rarely fills.
CACHE_MAX_FRAMES = 1500
# How far ahead of the playhead the prerender worker looks (frames, on the step
# grid). It fills this window and then IDLES — instead of racing to annotate the
# whole video, which pins every CPU core the moment a video is loaded (there is
# no GPU to offload to). ~300 frames ≈ 10–15 s of lookahead at typical FPS.
PRERENDER_AHEAD = 300

frame_cache = OrderedDict()        # frame_idx -> (jpg_bytes, person_boxes); LRU order
cache_state = {"cfg_key": "", "epoch": 0, "fill_cursor": 0}

# Capture owned solely by the prerender worker (sequential reads).
_worker_cap = {"cap": None, "video": "", "pos": -2}


def compute_cfg_key() -> str:
    """Fingerprint of everything that changes rendering. Caller holds state_lock."""
    return "|".join(str(x) for x in (
        state["selected_video"], state["person_model_path"], state["ppe_model_path"],
        state["sm_model_path"], state["person_conf"], state["ppe_conf"],
        state["sm_conf"], state["ppe_inside_person"], state["frame_step"],
        state["ppe_classes"], state["sm_classes"],
    ))


def cache_sync_cfg(cfg_key: str) -> int:
    """Clear the cache if config changed. Returns current epoch. Holds cache_lock."""
    if cache_state["cfg_key"] != cfg_key:
        frame_cache.clear()
        cache_state["cfg_key"] = cfg_key
        cache_state["fill_cursor"] = 0
        cache_state["epoch"] += 1
    return cache_state["epoch"]


def cache_get(idx: int):
    """Return cached (jpg, boxes), marking it recently used. Holds cache_lock."""
    item = frame_cache.get(idx)
    if item is not None:
        frame_cache.move_to_end(idx)
    return item


def cache_put(idx: int, jpg, boxes):
    """Store an annotated frame, evicting least-recently-used past the cap. Holds cache_lock."""
    frame_cache[idx] = (jpg, boxes)
    frame_cache.move_to_end(idx)
    while len(frame_cache) > CACHE_MAX_FRAMES:
        frame_cache.popitem(last=False)


def cached_ranges(step: int):
    """Cached frames grouped into contiguous [start, end] runs (on the step grid)."""
    keys = sorted(frame_cache.keys())
    ranges = []
    for k in keys:
        if ranges and (k - ranges[-1][1]) == step:
            ranges[-1][1] = k
        else:
            ranges.append([k, k])
    return ranges


def pick_next_to_annotate(playhead: int, step: int, total: int, fill_cursor: int):
    """
    Choose the next frame to annotate. Holds cache_lock. Returns (idx, new_cursor).

    Strategy: fill only a BOUNDED window ahead of the playhead
    (PRERENDER_AHEAD frames), then stop. While playing, the playhead advances and
    the window slides with it — so there is always at most a window's worth of
    work pending, never the whole video. While paused it fills the window once and
    then idles. This is what keeps the worker from pinning every CPU core (there is
    no GPU here). Scrubbing outside the window re-renders on demand and re-caches.

    Returns (None, cursor) when the window is fully cached (nothing to do now).
    """
    start = max(int(playhead), 0)
    limit = start + PRERENDER_AHEAD * step
    if total > 0:
        limit = min(limit, total - 1)

    # The cursor rides forward as we fill; a seek (or the playhead moving back)
    # can leave it past the window, so snap it back to the window start.
    c = max(int(fill_cursor), start)
    if c > limit:
        c = start

    while c <= limit:
        if c not in frame_cache:
            return c, c + step       # annotate c, advance the cursor past it
        c += step

    return None, c                   # window fully cached — idle until the playhead moves


def worker_read(video: str, idx: int):
    """Read one frame for the worker, reusing a sequential capture when possible."""
    wc = _worker_cap
    if wc["cap"] is None or wc["video"] != video:
        if wc["cap"] is not None:
            wc["cap"].release()
        wc["cap"] = cv2.VideoCapture(video)
        wc["video"] = video
        wc["pos"] = -2
    cap = wc["cap"]
    if not cap.isOpened():
        # Drop the dead capture so the next tick re-attempts the open (handles a
        # slow/flaky remote URL that isn't ready on the very first try).
        cap.release()
        wc["cap"] = None
        wc["video"] = ""
        return None
    if idx != wc["pos"] + 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    wc["pos"] = idx
    return frame


def read_raw_frame(video: str, frame_idx: int):
    """Fetch a single raw (unannotated) frame on demand — used by crop actions."""
    if not video:
        return None
    cap = cv2.VideoCapture(video)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def ensure_video_meta(video: str):
    """Populate runtime fps/total_frames for a video if not already known."""
    if not video:
        return
    # For a stream URL, don't re-open once we've learned about it — a remote
    # open is a network round-trip and the length may legitimately be unknown.
    if runtime["active_video"] == video and (
        runtime["total_frames"] > 0 or is_stream_url(video)
    ):
        return
    cap = cv2.VideoCapture(video)
    try:
        if cap.isOpened():
            runtime["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            runtime["fps"] = fps if fps and fps > 1e-6 else 25.0
            runtime["active_video"] = video
    finally:
        cap.release()

# Guards the export worker so only one render-to-file job runs at a time.
export_lock = threading.Lock()
export_state = {
    "running": False,
    "done": False,
    "error": "",
    "progress": 0,
    "total": 0,
    "output_path": "",
    "download_name": "",
}


def configure_quiet_logging():
    # Keep terminal clean: hide per-request access logs, keep errors.
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)
    app.logger.setLevel(logging.ERROR)


def is_stream_url(value: str) -> bool:
    """True for a remote http(s) source we can fetch to a local temp file."""
    return isinstance(value, str) and value.strip().lower().startswith(
        ("http://", "https://")
    )


# --- Remote URL → local temp download ------------------------------------------
# A URL source is downloaded to a temp file and played from disk (smooth seeking),
# rather than streamed. The previous download is deleted when a new link is used.
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "inference_url_downloads"
download_lock = threading.Lock()
download_state = {
    "url": "",          # the URL currently being fetched / cached
    "path": "",         # local temp file once ready
    "status": "idle",   # idle | downloading | ready | error
    "progress": 0.0,    # 0..1 (0 when server sends no Content-Length)
    "error": "",
}


def _remove_file(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _download_dest(url: str) -> Path:
    ext = Path(urlparse(url).path).suffix.lower()
    if ext not in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"):
        ext = ".mp4"
    return DOWNLOAD_DIR / (hashlib.md5(url.encode("utf-8")).hexdigest() + ext)


def start_url_download(url: str) -> None:
    """Kick off (or reuse) a background download of `url` to a temp file. Deletes
    the previously-downloaded file when the link changes. Safe to call repeatedly."""
    url = (url or "").strip()
    if not url:
        return
    with download_lock:
        if download_state["url"] == url and download_state["status"] in ("downloading", "ready"):
            return  # already fetching or cached
        old_path = download_state["path"]
        download_state.update(
            {"url": url, "path": "", "status": "downloading", "progress": 0.0, "error": ""}
        )
    _remove_file(old_path)  # drop the previous link's file
    threading.Thread(target=_download_worker, args=(url,), daemon=True).start()


def _download_worker(url: str) -> None:
    dest = _download_dest(url)
    try:
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            got = 0
            with open(dest, "wb") as fh:
                while True:
                    with download_lock:
                        if download_state["url"] != url:  # superseded by a newer link
                            _remove_file(str(dest))
                            return
                    chunk = resp.read(1 << 20)  # 1 MB
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        with download_lock:
                            download_state["progress"] = min(1.0, got / total)
        with download_lock:
            if download_state["url"] != url:
                _remove_file(str(dest))
                return
            download_state.update({"path": str(dest), "status": "ready", "progress": 1.0})
        # Switch the active source to the downloaded file and reload the capture.
        with state_lock:
            switch = state["source_mode"] == "url" and state["video_url"] == url
            if switch:
                state["selected_video"] = str(dest)
        if switch:
            release_capture()
    except Exception as exc:  # network / HTTP / disk error
        _remove_file(str(dest))
        with download_lock:
            if download_state["url"] == url:
                download_state.update({"status": "error", "error": str(exc)})
        runtime["last_error"] = f"Download failed: {exc}"


def url_download_path(url: str):
    """Return the ready local path for `url`, or None if not downloaded yet."""
    url = (url or "").strip()
    with download_lock:
        if (
            download_state["url"] == url
            and download_state["status"] == "ready"
            and download_state["path"]
            and os.path.exists(download_state["path"])
        ):
            return download_state["path"]
    return None


@atexit.register
def _cleanup_downloads():
    with download_lock:
        _remove_file(download_state["path"])


def list_videos(folder_path: str) -> list[str]:
    folder = Path(folder_path).expanduser().resolve()
    if not folder.is_dir():
        return []
    videos = []
    for name in os.listdir(folder):
        full_path = folder / name
        if full_path.is_file() and name.lower().endswith(ppe_inference.VIDEO_EXTENSIONS):
            videos.append(str(full_path))
    videos.sort()
    return videos


def discover_pt_models(root_dirs: list[str]) -> list[str]:
    found = set()
    for root in root_dirs:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(".pt"):
                    found.add(str(Path(dirpath) / filename))
    return sorted(found)


def _is_generic_yolo_pt(path: Path) -> bool:
    """Basenames like yolo11n.pt — prefer real PPE/SM weights inside the same package folder."""
    name = path.name.lower()
    return name.startswith("yolo") and name.endswith(".pt")


def discover_model_packages(models_dir: Path) -> list[tuple[str, str]]:
    """
    One entry per immediate subfolder of models/ that contains a .pt file.
    Returns (folder_display_name, absolute_path_to_chosen_pt).
    """
    if not models_dir.is_dir():
        return []
    packages: list[tuple[str, str]] = []
    for child in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        pt_files: list[Path] = []
        for dirpath, _, filenames in os.walk(child):
            for fn in filenames:
                if fn.lower().endswith(".pt"):
                    pt_files.append(Path(dirpath) / fn)
        if not pt_files:
            continue
        non_yolo = [p for p in pt_files if not _is_generic_yolo_pt(p)]
        candidates = non_yolo if non_yolo else pt_files
        candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
        chosen = str(candidates[0].resolve())
        packages.append((child.name, chosen))
    return packages


def load_settings():
    if not APP_SETTINGS_FILE.exists():
        return
    try:
        with open(APP_SETTINGS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            return
        for key in list(state.keys()):
            if key in loaded:
                state[key] = loaded[key]
    except (OSError, json.JSONDecodeError):
        return


def save_settings():
    try:
        with open(APP_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


def get_or_load_model(model_path: str):
    if model_path not in model_cache:
        model_cache[model_path] = YOLO(model_path)
    return model_cache[model_path]


def get_ppe_class_names(model_path: str) -> list[str]:
    """Class names of a PPE model, in class-id order. [] for None/unloadable."""
    if not model_path or model_path == NONE_MODEL_VALUE:
        return []
    try:
        with inference_lock:
            model = get_or_load_model(model_path)
            return ordered_model_class_names(model)
    except Exception:
        return []


def release_capture():
    """Tear down the annotation cache + worker capture. Does NOT change playing."""
    with cache_lock:
        frame_cache.clear()
        cache_state["cfg_key"] = ""
        cache_state["epoch"] += 1
    if _worker_cap["cap"] is not None:
        _worker_cap["cap"].release()
        _worker_cap["cap"] = None
        _worker_cap["video"] = ""
        _worker_cap["pos"] = -2
    runtime["cap"] = None
    runtime["active_video"] = ""
    runtime["frame_idx"] = -1
    runtime["total_frames"] = 0


def ensure_capture_open():
    """Validate that a video is selected and openable, and learn its fps/total."""
    selected_video = state["selected_video"]
    if not selected_video:
        runtime["last_error"] = "Choose a valid video."
        return False
    ensure_video_meta(selected_video)
    if runtime["active_video"] != selected_video:
        runtime["last_error"] = "Failed to open selected video."
        return False
    runtime["last_error"] = ""
    return True


_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colors are BGR.
COLOR_PERSON = (255, 170, 40)
COLOR_PPE = (90, 220, 70)
COLOR_SM = (80, 200, 255)
COLOR_MANUAL = (235, 110, 200)
COLOR_HUD_BG = (24, 26, 32)
# Per-person PPE status chips: grey = class not detected, green = detected.
COLOR_CHIP_OFF = (78, 82, 90)
COLOR_CHIP_ON = (90, 200, 90)
COLOR_CHIP_BORDER = (28, 32, 38)


def sanitize_class_name(name: str) -> str:
    """Make a class label safe to use as a folder name."""
    cleaned = "".join(c if (c.isalnum() or c in (" ", "-", "_")) else "_" for c in str(name)).strip()
    cleaned = cleaned.replace(" ", "_")
    return cleaned or "unlabeled"


def draw_label_chip(img, x, y, text, bg_color, fg_color=(20, 20, 22)):
    """Draw a filled label chip whose bottom edge sits just above (x, y)."""
    scale, thickness = 0.5, 1
    (tw, th), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    pad_x, pad_y = 6, 4
    chip_w = tw + pad_x * 2
    chip_h = th + baseline + pad_y * 2
    cx = max(0, x)
    cy = y - chip_h
    if cy < 0:
        cy = max(0, y)
    cv2.rectangle(img, (cx, cy), (cx + chip_w, cy + chip_h), bg_color, -1, cv2.LINE_AA)
    cv2.putText(
        img, text, (cx + pad_x, cy + pad_y + th),
        _FONT, scale, fg_color, thickness, cv2.LINE_AA,
    )


def draw_status_chip(img, x, y, chip_w, chip_h, name, present, conf, scale, thickness, pad_x):
    """Draw one PPE status chip: green (with the detection %) if the class was
    detected, grey if not. Font `scale`/`thickness` come from the caller so chips
    can size themselves to the person box."""
    bg = COLOR_CHIP_ON if present else COLOR_CHIP_OFF
    fg = (18, 30, 18) if present else (206, 210, 218)
    text = f"{name} {int(round(conf * 100))}%" if (present and conf is not None) else name
    cv2.rectangle(img, (x, y), (x + chip_w, y + chip_h), bg, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x, y), (x + chip_w, y + chip_h), COLOR_CHIP_BORDER, 1, cv2.LINE_AA)
    # Left-aligned, and exactly centered vertically: the baseline sits so the glyph
    # box (ascent `th` above, descent `tb` below) is centered in the chip.
    (_, th), tb = cv2.getTextSize(text, _FONT, scale, thickness)
    ty = y + (chip_h + th - tb) // 2
    cv2.putText(img, text, (x + pad_x, ty), _FONT, scale, fg, thickness, cv2.LINE_AA)


def ordered_model_class_names(model) -> list[str]:
    """Return a YOLO model's class names in class-id order (handles dict or list)."""
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names.keys())]
    if names:
        return [str(n) for n in names]
    return []


def draw_hud(img, lines, accent_color, align="left"):
    """Draw a translucent stat panel with an accent bar in a top corner."""
    if not lines:
        return
    scale, thickness = 0.58, 1
    sizes = [cv2.getTextSize(t, _FONT, scale, thickness)[0] for t in lines]
    max_w = max(s[0] for s in sizes)
    line_h = max(s[1] for s in sizes)
    pad, gap, bar = 10, 9, 6
    panel_w = bar + pad + max_w + pad
    panel_h = pad * 2 + len(lines) * line_h + (len(lines) - 1) * gap
    y0 = 10
    x0 = (img.shape[1] - panel_w - 10) if align == "right" else 10
    x0 = max(0, x0)
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.rectangle(img, (x0, y0), (x0 + bar, y0 + panel_h), accent_color, -1)
    y = y0 + pad + line_h
    for text in lines:
        cv2.putText(
            img, text, (x0 + bar + pad, y),
            _FONT, scale, (236, 239, 245), thickness, cv2.LINE_AA,
        )
        y += line_h + gap


def run_ppe(
    frame,
    person_model,
    ppe_model,
    person_conf,
    ppe_conf,
    manual_boxes=None,
    ppe_inside_person=True,
    selected_classes=None,
):
    annotated = frame.copy()
    h, w = frame.shape[:2]
    person_count = 0
    ppe_count = 0
    person_boxes = []
    manual_boxes = manual_boxes or []

    # Which PPE classes to surface as per-person status chips. selected_classes
    # is None => all classes of this model; otherwise the chosen subset. We keep
    # them in the model's own class-id order so every person shows the same rows.
    all_class_names = ordered_model_class_names(ppe_model)
    if selected_classes is None:
        strip_names = all_class_names
    else:
        wanted = set(selected_classes)
        strip_names = [n for n in all_class_names if n in wanted]
    strip_set = set(strip_names)

    def draw_ppe_box(gx1, gy1, gx2, gy2, label, bconf):
        nonlocal ppe_count
        gx1, gy1, gx2, gy2 = ppe_inference.clamp_box(gx1, gy1, gx2, gy2, w, h)
        if gx2 <= gx1 or gy2 <= gy1:
            return
        ppe_count += 1
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), COLOR_PPE, 2, cv2.LINE_AA)
        draw_label_chip(annotated, gx1, gy1, f"{label} {bconf:.2f}", COLOR_PPE)

    def draw_status_strip(x1, y1, x2, y2, detected):
        """Stack one chip per selected class down the person box's right edge
        (flips to the left edge when there is no room), green if detected. The
        chips scale with the person box — small/far person → small chips,
        large/near person → large readable chips."""
        if not strip_names:
            return
        n = len(strip_names)
        box_h = max(1, y2 - y1)

        # Each chip is a fixed fraction of the person box height (0.5/10 = 5%),
        # so the labels track the person's on-screen size.
        CHIP_HEIGHT_RATIO = 0.05
        chip_h = max(8, int(round(box_h * CHIP_HEIGHT_RATIO)))
        gap = max(1, int(round(chip_h * 0.18)))
        # Font scale chosen so the glyphs (ascender + descender) fill ~80% of the
        # chip height — i.e. the font matches the box.
        (_, ref_h), ref_b = cv2.getTextSize("Ag", _FONT, 1.0, 1)
        scale = max(0.2, (chip_h * 0.80) / (ref_h + ref_b))
        thickness = 2 if scale >= 0.8 else 1
        pad_x = max(2, int(round(scale * 8)))
        chip_w = max(
            cv2.getTextSize(f"{name} 100%", _FONT, scale, thickness)[0][0]
            for name in strip_names
        ) + pad_x * 2

        # Sit to the right of the box, flipping to the left when there's no room.
        if x2 + gap + chip_w <= w:
            sx = x2 + gap
        else:
            sx = max(0, x1 - gap - chip_w)

        # Anchor at the box top, keeping the whole strip on-screen.
        strip_h = n * chip_h + (n - 1) * gap
        sy = 0 if strip_h >= h else max(0, min(y1, h - strip_h))

        for i, name in enumerate(strip_names):
            cy = sy + i * (chip_h + gap)
            draw_status_chip(
                annotated, sx, cy, chip_w, chip_h,
                name, name in detected, detected.get(name),
                scale, thickness, pad_x,
            )

    def process_person_region(x1, y1, x2, y2, person_label):
        nonlocal person_count, ppe_count
        x1, y1, x2, y2 = ppe_inference.clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            return
        person_count += 1
        person_boxes.append((x1, y1, x2, y2))
        person_idx = len(person_boxes)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_PERSON, 2, cv2.LINE_AA)
        draw_label_chip(annotated, x1, y1, f"P{person_idx} {person_label}", COLOR_PERSON)

        # When PPE is scoped to the person box, detect inside this crop only.
        if not ppe_inside_person:
            return

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            draw_status_strip(x1, y1, x2, y2, {})
            return

        # Collect which of the selected classes appear on this person (keeping the
        # highest confidence per class), then show the status strip instead of
        # drawing every individual PPE box.
        detected = {}
        ppe_result = ppe_model.predict(crop, conf=ppe_conf, verbose=False)[0]
        if ppe_result.boxes is not None:
            for bbox in ppe_result.boxes:
                bconf = float(bbox.conf[0])
                bcls = int(bbox.cls[0])
                label = ppe_result.names.get(bcls, str(bcls))
                if label in strip_set and bconf > detected.get(label, 0.0):
                    detected[label] = bconf
        ppe_count += len(detected)
        draw_status_strip(x1, y1, x2, y2, detected)

    # Force person-model stage to detect only "person" class.
    person_result = person_model.predict(
        frame,
        conf=person_conf,
        classes=[ppe_inference.PERSON_CLASS_ID],
        verbose=False,
    )[0]

    if person_result.boxes is not None:
        for pbox in person_result.boxes:
            cls_id = int(pbox.cls[0])
            if cls_id != ppe_inference.PERSON_CLASS_ID:
                continue
            x1, y1, x2, y2 = map(int, pbox.xyxy[0])
            pconf = float(pbox.conf[0])
            process_person_region(x1, y1, x2, y2, f"person {pconf:.2f}")

    # Manual boxes are user-drawn class annotations: draw them with their own
    # label/color, don't treat them as persons or run PPE inside them.
    for mb in manual_boxes:
        mx1, my1, mx2, my2 = map(int, mb[:4])
        cls_label = mb[4] if len(mb) > 4 and mb[4] else "manual"
        mx1, my1, mx2, my2 = ppe_inference.clamp_box(mx1, my1, mx2, my2, w, h)
        if mx2 <= mx1 or my2 <= my1:
            continue
        cv2.rectangle(annotated, (mx1, my1), (mx2, my2), COLOR_MANUAL, 2, cv2.LINE_AA)
        draw_label_chip(annotated, mx1, my1, str(cls_label), COLOR_MANUAL, fg_color=(255, 255, 255))

    # When disabled, run the PPE model once across the whole frame.
    if not ppe_inside_person:
        # The PPE model is trained on close-up person crops, so on a full frame
        # its targets are tiny. Predict near native resolution (multiple of 32,
        # capped) so small PPE objects are not shrunk away by the default imgsz.
        ppe_imgsz = int(min(1920, max(640, ((max(h, w) + 31) // 32) * 32)))
        ppe_result = ppe_model.predict(
            frame, conf=ppe_conf, imgsz=ppe_imgsz, verbose=False
        )[0]
        if ppe_result.boxes is not None:
            for bbox in ppe_result.boxes:
                bconf = float(bbox.conf[0])
                bcls = int(bbox.cls[0])
                label = ppe_result.names.get(bcls, str(bcls))
                # No persons to attach chips to in whole-frame mode, so keep the
                # boxes here — but still honor the selected-class filter.
                if label not in strip_set:
                    continue
                gx1, gy1, gx2, gy2 = map(int, bbox.xyxy[0])
                draw_ppe_box(gx1, gy1, gx2, gy2, label, bconf)

    draw_hud(
        annotated,
        [f"PPE  Persons: {person_count}", f"PPE items: {ppe_count}"],
        COLOR_PPE,
    )
    return annotated, person_boxes


def run_sm(frame, sm_model, sm_conf, selected_classes=None):
    annotated, det_count = sm_cropper.draw_detections(frame, sm_model, sm_conf,
                                                      selected_classes=selected_classes)
    draw_hud(annotated, [f"SM  Detections: {det_count}"], COLOR_SM, align="right")
    return annotated


def build_model_options_html(model_paths: list[str], selected_path: str) -> str:
    options = [
        f'<option value="{NONE_MODEL_VALUE}"'
        + (' selected="selected"' if selected_path == NONE_MODEL_VALUE else "")
        + ">None</option>"
    ]
    for path in model_paths:
        selected_attr = ' selected="selected"' if path == selected_path else ""
        label = html.escape(Path(path).name)
        value = html.escape(path, quote=True)
        options.append(f'<option value="{value}"{selected_attr}>{label}</option>')
    return "".join(options)


def build_model_package_options_html(
    packages: list[tuple[str, str]], selected_path: str
) -> str:
    """Dropdown shows top-level folder name; value is the resolved .pt path inside."""
    options = [
        f'<option value="{NONE_MODEL_VALUE}"'
        + (' selected="selected"' if selected_path == NONE_MODEL_VALUE else "")
        + ">None</option>"
    ]
    for folder_name, pt_path in packages:
        selected_attr = ' selected="selected"' if pt_path == selected_path else ""
        label = html.escape(folder_name)
        value = html.escape(pt_path, quote=True)
        tip = html.escape(Path(pt_path).name, quote=True)
        options.append(
            f'<option value="{value}" title="{tip}"{selected_attr}>{label}</option>'
        )
    return "".join(options)


def build_ppe_classes_html(class_names: list[str], selected, css_class="ppe-cls") -> str:
    """Checkbox rows for a model-class picker. selected=None => all checked."""
    sel_set = None if selected is None else set(selected)
    rows = []
    for name in class_names:
        checked = " checked" if (sel_set is None or name in sel_set) else ""
        safe = html.escape(name)
        value = html.escape(name, quote=True)
        rows.append(
            f'<label class="cls-row"><input type="checkbox" class="{css_class}" '
            f'value="{value}"{checked} />{safe}</label>'
        )
    return "".join(rows) or '<div class="cls-empty">No classes</div>'


def snapshot_inference_cfg(frame_idx: int) -> dict:
    """Copy just the settings the renderer needs. Caller must hold state_lock."""
    return {
        "person_model_path": state["person_model_path"],
        "ppe_model_path": state["ppe_model_path"],
        "sm_model_path": state["sm_model_path"],
        "person_conf": float(state["person_conf"]),
        "ppe_conf": float(state["ppe_conf"]),
        "sm_conf": float(state["sm_conf"]),
        "ppe_inside_person": bool(state["ppe_inside_person"]),
        "ppe_classes": (
            list(state["ppe_classes"]) if state["ppe_classes"] is not None else None
        ),
        "sm_classes": (
            list(state["sm_classes"]) if state["sm_classes"] is not None else None
        ),
        "manual_boxes": list(runtime["manual_boxes_by_frame"].get(frame_idx, [])),
    }


def annotate_frame(frame, cfg: dict):
    """
    Run inference and annotate a single frame, returning the annotated BGR image.
    Touches no shared state, so it can run WITHOUT state_lock; the heavy predicts
    are serialized by inference_lock.
    Returns (annotated_bgr_or_None, person_boxes, error_or_None).
    """
    out = frame.copy()
    person_boxes = []
    try:
        with inference_lock:
            ran_any = False
            if (
                cfg["person_model_path"] != NONE_MODEL_VALUE
                and cfg["ppe_model_path"] != NONE_MODEL_VALUE
            ):
                person_model = get_or_load_model(cfg["person_model_path"])
                ppe_model = get_or_load_model(cfg["ppe_model_path"])
                out, person_boxes = run_ppe(
                    out,
                    person_model,
                    ppe_model,
                    cfg["person_conf"],
                    cfg["ppe_conf"],
                    manual_boxes=cfg["manual_boxes"],
                    ppe_inside_person=cfg["ppe_inside_person"],
                    selected_classes=cfg["ppe_classes"],
                )
                ran_any = True

            if cfg["sm_model_path"] != NONE_MODEL_VALUE:
                sm_model = get_or_load_model(cfg["sm_model_path"])
                out = run_sm(out, sm_model, cfg["sm_conf"], cfg["sm_classes"])
                ran_any = True

        if not ran_any:
            draw_hud(out, ["No model selected", "(all set to None)"], (60, 170, 255))
    except Exception as exc:
        return None, [], f"Inference error: {exc}"

    return out, person_boxes, None


def render_annotated(frame, cfg: dict):
    """
    Annotate a frame and JPEG-encode it for the MJPEG stream.
    Returns (jpg_bytes_or_None, person_boxes, raw_frame, error_or_None).
    """
    raw = frame.copy()
    out, person_boxes, error = annotate_frame(frame, cfg)
    if error:
        return None, [], raw, error

    ok, buf = cv2.imencode(".jpg", out)
    if not ok:
        return None, person_boxes, raw, None
    return buf.tobytes(), person_boxes, raw, None


def commit_render(jpg, person_boxes, raw, error):
    """Write render results into runtime. Caller must hold state_lock."""
    if error:
        runtime["last_error"] = error
        state["playing"] = False
        return runtime["last_jpg"]
    runtime["last_raw_frame"] = raw
    runtime["last_person_boxes"] = person_boxes
    if jpg is not None:
        runtime["last_jpg"] = jpg
        runtime["displayed_count"] += 1
    return runtime["last_jpg"]


def process_frame_to_jpg(frame):
    """Synchronous render+commit used by seek/crop paths (caller holds state_lock)."""
    frame_idx = int(runtime.get("frame_idx", -1))
    cfg = snapshot_inference_cfg(frame_idx)
    jpg, person_boxes, raw, error = render_annotated(frame, cfg)
    return commit_render(jpg, person_boxes, raw, error)


def save_frame_image(frame, subdir: str, suffix: str = "",
                     crop_box=None, frame_size=None) -> str:
    target_dir = CROPS_ROOT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    base = runtime["active_video"] or "frame"
    frame_idx = max(int(runtime["frame_idx"]), 0)
    if crop_box is not None and frame_size is not None:
        # A person/region crop → the shared <video>_<frame>_<yolo coords> name.
        fw, fh = frame_size
        filename = cropnames.yolo_crop_name(base, frame_idx, crop_box, fw, fh)
    else:
        # A whole-frame capture (Save frame / Save background) — no sub-region to
        # encode, so keep the timestamped name that lets repeated saves coexist.
        stem = cropnames.clean_video_name(base)
        stamp = int(time.time() * 1000)
        suffix_part = f"_{suffix}" if suffix else ""
        filename = f"{stem}_frame_{frame_idx:06d}{suffix_part}_{stamp}.jpg"
    save_path = target_dir / filename
    cv2.imwrite(str(save_path), frame)
    # Crop saves do NOT log to Convex — the database only updates when an image
    # gets annotated (see db_log.record_annotation).
    return str(save_path)


def save_person_crop(person_index: int) -> str:
    frame = runtime["last_raw_frame"]
    if frame is None:
        raise ValueError("No frame available for person crop.")

    boxes = runtime["last_person_boxes"] or []
    if person_index < 1 or person_index > len(boxes):
        raise ValueError("Selected person number is not available in current frame.")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = boxes[person_index - 1]
    x1 = max(0, x1 - PERSON_CROP_PADDING)
    y1 = max(0, y1 - PERSON_CROP_PADDING)
    x2 = min(w - 1, x2 + PERSON_CROP_PADDING)
    y2 = min(h - 1, y2 + PERSON_CROP_PADDING)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid person crop bounds.")

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop generated.")
    return save_frame_image(crop, "person", crop_box=(x1, y1, x2, y2), frame_size=(w, h))


def save_person_crop_from_box(box, suffix: str = "person_manual", subdir: str = "person") -> str:
    frame = runtime["last_raw_frame"]
    if frame is None:
        raise ValueError("No frame available for person crop.")
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box[:4])
    x1 = max(0, x1 - PERSON_CROP_PADDING)
    y1 = max(0, y1 - PERSON_CROP_PADDING)
    x2 = min(w - 1, x2 + PERSON_CROP_PADDING)
    y2 = min(h - 1, y2 + PERSON_CROP_PADDING)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid manual box bounds.")
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop generated.")
    return save_frame_image(crop, subdir, suffix=suffix, crop_box=(x1, y1, x2, y2), frame_size=(w, h))


def add_manual_box_for_current_frame(
    x1_ratio: float, y1_ratio: float, x2_ratio: float, y2_ratio: float, class_name: str = "manual"
):
    frame = runtime["last_raw_frame"]
    if frame is None:
        raise ValueError("No frame loaded yet.")
    frame_idx = int(runtime["frame_idx"])
    if frame_idx < 0:
        raise ValueError("No active frame index.")
    h, w = frame.shape[:2]
    x1 = int(max(0.0, min(1.0, x1_ratio)) * (w - 1))
    y1 = int(max(0.0, min(1.0, y1_ratio)) * (h - 1))
    x2 = int(max(0.0, min(1.0, x2_ratio)) * (w - 1))
    y2 = int(max(0.0, min(1.0, y2_ratio)) * (h - 1))
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        raise ValueError("Manual box is too small.")
    box = (x1, y1, x2, y2, str(class_name))
    runtime["manual_boxes_by_frame"].setdefault(frame_idx, []).append(box)
    return box, frame_idx


# Default per-class prompts (shipped in ppe_prompts.json) used to prefill the
# Ask AI sections until the user saves their own.
try:
    PPE_DEFAULT_PROMPTS = json.loads(
        paths.PPE_PROMPTS_FILE.read_text(encoding="utf-8")
    )
    if not isinstance(PPE_DEFAULT_PROMPTS, dict):
        PPE_DEFAULT_PROMPTS = {}
except Exception:
    PPE_DEFAULT_PROMPTS = {}


# --- Azure AI Foundry (vision) --------------------------------------------------
FOUNDRY_ENDPOINT = os.environ.get("AZURE_FOUNDRY_API_ENDPOINT", "")
FOUNDRY_TOKEN = os.environ.get("AZURE_FOUNDRY_API_TOKEN", "")
FOUNDRY_DEPLOYMENT = os.environ.get("AZURE_FOUNDRY_DEPLOYMENT", "gpt-5-mini")
FOUNDRY_API_VERSION = os.environ.get("AZURE_FOUNDRY_API_VERSION", "2025-04-01-preview")


def _foundry_base_url() -> str:
    """The account base URL, stripped of the /openai/responses path + query."""
    ep = FOUNDRY_ENDPOINT
    if "/openai/responses" in ep:
        ep = ep.split("/openai/responses")[0]
    elif "?" in ep:
        ep = ep.split("?")[0]
    return ep.rstrip("/")


def call_foundry(image_bgr, prompt: str) -> str:
    """Send a BGR crop + prompt to the Azure vision model; return the answer text."""
    base = _foundry_base_url()
    if not base or not FOUNDRY_TOKEN:
        raise RuntimeError("Azure endpoint/token not configured (.env).")
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("Could not encode the crop.")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    url = f"{base}/openai/deployments/{FOUNDRY_DEPLOYMENT}/chat/completions?api-version={FOUNDRY_API_VERSION}"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_completion_tokens": 1500,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"api-key": FOUNDRY_TOKEN, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    choices = out.get("choices") or []
    if not choices:
        raise RuntimeError("The model returned no answer.")
    return (choices[0].get("message", {}).get("content") or "").strip()


def get_person_crop_from_point(x_ratio: float, y_ratio: float):
    """Return the BGR crop of the person box under the clicked point (no save)."""
    frame = runtime["last_raw_frame"]
    boxes = runtime["last_person_boxes"] or []
    if frame is None:
        raise ValueError("No frame available.")
    if not boxes:
        raise ValueError("No person detected in the current frame.")
    h, w = frame.shape[:2]
    px = int(max(0.0, min(1.0, x_ratio)) * (w - 1))
    py = int(max(0.0, min(1.0, y_ratio)) * (h - 1))
    chosen = next(((x1, y1, x2, y2) for (x1, y1, x2, y2) in boxes
                   if x1 <= px <= x2 and y1 <= py <= y2), None)
    if chosen is None:
        raise ValueError("Click directly on a detected person box.")
    x1, y1, x2, y2 = chosen
    x1 = max(0, x1 - PERSON_CROP_PADDING)
    y1 = max(0, y1 - PERSON_CROP_PADDING)
    x2 = min(w - 1, x2 + PERSON_CROP_PADDING)
    y2 = min(h - 1, y2 + PERSON_CROP_PADDING)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Empty crop.")
    return crop


def save_person_crop_from_point(x_ratio: float, y_ratio: float) -> str:
    frame = runtime["last_raw_frame"]
    boxes = runtime["last_person_boxes"] or []
    if frame is None:
        raise ValueError("No frame available for person crop.")
    if not boxes:
        raise ValueError("No person in the current frame.")

    h, w = frame.shape[:2]
    px = int(max(0.0, min(1.0, x_ratio)) * (w - 1))
    py = int(max(0.0, min(1.0, y_ratio)) * (h - 1))

    # Prefer a box that contains the clicked point.
    chosen_idx = None
    for idx, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        if x1 <= px <= x2 and y1 <= py <= y2:
            chosen_idx = idx
            break

    if chosen_idx is None:
        raise ValueError("Click on a detected person in the video.")
    return save_person_crop(chosen_idx)


def _show_cached(idx, jpg, boxes, advance=True):
    """Make a cached frame the currently-displayed one. Caller must NOT hold locks."""
    with state_lock:
        runtime["frame_idx"] = idx
        runtime["last_person_boxes"] = boxes
        runtime["last_raw_frame"] = None  # fetched on demand for crops
        if jpg is not None:
            runtime["last_jpg"] = jpg
            if advance:
                runtime["displayed_count"] += 1
        runtime["last_error"] = ""


def seek_to_frame(target_idx: int):
    """Jump to a frame. Instant if already annotated (cached); otherwise render it
    once and cache it. Never clears other cached frames. Hold no lock when calling."""
    with state_lock:
        video = state["selected_video"]
        step = max(int(state["frame_step"]), 1)
        cfg_key = compute_cfg_key()
    if not video:
        runtime["last_error"] = "Choose a valid video."
        return False
    ensure_video_meta(video)
    total = max(int(runtime["total_frames"]), 0)
    target_idx = max(0, min(int(target_idx), total - 1)) if total > 0 else max(0, int(target_idx))
    target_idx = (target_idx // step) * step  # snap to the step grid (shared with the worker)

    # Already annotated? Show instantly (and keep everything else cached).
    with cache_lock:
        cache_sync_cfg(cfg_key)
        cache_state["fill_cursor"] = target_idx  # background fill follows the seek
        cached = cache_get(target_idx)
        epoch = cache_state["epoch"]
    if cached is not None:
        _show_cached(target_idx, cached[0], cached[1])
        return True

    # First visit to this frame under the current config: render once and cache it.
    frame = read_raw_frame(video, target_idx)
    if frame is None:
        runtime["last_error"] = "Unable to seek to requested frame."
        return False
    with state_lock:
        cfg = snapshot_inference_cfg(target_idx)
    jpg, person_boxes, raw, error = render_annotated(frame, cfg)
    if error:
        runtime["last_error"] = error
        return False
    with cache_lock:
        if cache_state["epoch"] == epoch and jpg is not None:
            cache_put(target_idx, jpg, person_boxes)
    _show_cached(target_idx, jpg, person_boxes)
    return True


def prerender_worker():
    """
    Always-on background producer: whenever a video is selected (playing OR paused),
    find the nearest frame ahead of the playhead that is NOT yet annotated and
    annotate it — at most once per config. Already-cached frames are never redone.
    """
    while True:
        with state_lock:
            video = state["selected_video"]
            step = max(int(state["frame_step"]), 1)
            cfg_key = compute_cfg_key()
            playhead = runtime["frame_idx"] if runtime["frame_idx"] >= 0 else 0

        if not video:
            time.sleep(0.1)
            continue
        ensure_video_meta(video)
        total = max(int(runtime["total_frames"]), 0)

        with cache_lock:
            epoch = cache_sync_cfg(cfg_key)
            nxt, new_cursor = pick_next_to_annotate(
                playhead, step, total, cache_state["fill_cursor"]
            )
            cache_state["fill_cursor"] = new_cursor

        if nxt is None:
            time.sleep(0.05)  # whole video loaded (or memory cap reached)
            continue

        frame = worker_read(video, nxt)
        if frame is None:
            # If we've produced nothing at all for the very first frame, the
            # source didn't open — tell the user why instead of retrying silently.
            with cache_lock:
                nothing_yet = len(frame_cache) == 0
            if nothing_yet and nxt <= 0:
                runtime["last_error"] = (
                    "Could not open the stream URL — check the link is valid and reachable."
                    if is_stream_url(video)
                    else "Could not read the selected video file."
                )
            time.sleep(0.15)  # transient read failure / past end; try again later
            continue

        with state_lock:
            cfg = snapshot_inference_cfg(nxt)
        jpg, person_boxes, raw, error = render_annotated(frame, cfg)
        with cache_lock:
            if cache_state["epoch"] != epoch:
                continue  # config changed mid-render; discard
            if not error and jpg is not None:
                cache_put(nxt, jpg, person_boxes)


def mjpeg_generator():
    """Consumer: display cached frames, advancing at playback pace while playing."""
    while True:
        with state_lock:
            playing = bool(state["playing"])
            realtime = bool(state["simulate_realtime"])
            fps = float(runtime["fps"]) if runtime["fps"] > 0 else 25.0
            step = max(int(state["frame_step"]), 1)
            cur = runtime["frame_idx"]
            total = max(int(runtime["total_frames"]), 0)
            out_jpg = runtime["last_jpg"]

        advanced = False
        if playing:
            nxt = 0 if cur < 0 else cur + step
            if total > 0 and nxt > total - 1:
                with state_lock:
                    state["playing"] = False  # reached the end
            else:
                with cache_lock:
                    item = cache_get(nxt)
                if item is not None:
                    _show_cached(nxt, item[0], item[1])
                    out_jpg = item[0] if item[0] is not None else out_jpg
                    advanced = True
                # else: not annotated yet — wait briefly for the worker.
        elif out_jpg is None:
            # Paused with nothing shown yet: preview the earliest cached frame.
            with cache_lock:
                if frame_cache:
                    k = min(frame_cache)
                    item = frame_cache[k]
                else:
                    k, item = None, None
            if item is not None:
                _show_cached(k, item[0], item[1], advance=False)
                out_jpg = item[0] if item[0] is not None else out_jpg

        if out_jpg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + out_jpg + b"\r\n"
            )

        if playing and realtime and advanced:
            time.sleep(max(0.0, (step / fps) * DISPLAY_PACE_FACTOR))
        else:
            time.sleep(0.02 if playing else 0.05)




@app.get("/")
def index():
    return _render_inference_page()


def _render_inference_page():
    with state_lock:
        videos = list_videos(state["video_folder"])
        if state.get("source_mode") == "url" and state.get("video_url"):
            # Resume/kick the download; play the local copy once it's ready.
            start_url_download(state["video_url"])
            state["selected_video"] = url_download_path(state["video_url"]) or ""
        elif state["selected_video"] not in videos:
            state["selected_video"] = videos[0] if videos else ""
        all_pt = discover_pt_models([str(MODELS_DIR)])
        packages = discover_model_packages(MODELS_DIR)
        package_paths = [p for _, p in packages]
        if packages:
            allowed_pkg = set(package_paths)
            allowed_pkg.add(NONE_MODEL_VALUE)
            if state["ppe_model_path"] not in allowed_pkg:
                state["ppe_model_path"] = package_paths[0]
            if state["sm_model_path"] not in allowed_pkg:
                state["sm_model_path"] = package_paths[0]
        else:
            state["ppe_model_path"] = NONE_MODEL_VALUE
            state["sm_model_path"] = NONE_MODEL_VALUE
        if all_pt:
            allowed_pt = set(all_pt)
            allowed_pt.add(NONE_MODEL_VALUE)
            if state["person_model_path"] not in allowed_pt:
                state["person_model_path"] = all_pt[0]
        else:
            state["person_model_path"] = NONE_MODEL_VALUE
        person_model_options_html = build_model_options_html(all_pt, state["person_model_path"])
        ppe_model_options_html = build_model_package_options_html(packages, state["ppe_model_path"])
        sm_model_options_html = build_model_package_options_html(packages, state["sm_model_path"])
        discovered_lines = [
            f"{name} → {Path(pt).name}" for name, pt in packages
        ]
        discovered_models_text = (
            os.linesep.join(discovered_lines) if discovered_lines else "No model packages in models/"
        )
        # Persist the validated/normalized selections so the file always matches
        # exactly what the page shows (and self-heals stale paths).
        save_settings()
        ppe_model_path = state["ppe_model_path"]
        ppe_selected_classes = state["ppe_classes"]
        sm_model_path_val = state["sm_model_path"]
        sm_selected_classes = state["sm_classes"]
        video_folder_val = state["video_folder"]
        selected_video_val = state["selected_video"]
        video_url_val = state["video_url"]
        source_mode_val = state["source_mode"]
        person_conf_val = state["person_conf"]
        ppe_conf_val = state["ppe_conf"]
        sm_conf_val = state["sm_conf"]
        frame_step_val = state["frame_step"]
        simulate_realtime_val = state["simulate_realtime"]
        ppe_inside_val = state["ppe_inside_person"]

    # Loading the PPE model to read its class list can be slow, so do it outside
    # state_lock. Populates the "Classes" picker beside the PPE model dropdown.
    ppe_class_names = get_ppe_class_names(ppe_model_path)
    ppe_classes_html = build_ppe_classes_html(ppe_class_names, ppe_selected_classes)
    sm_class_names = get_ppe_class_names(sm_model_path_val)
    sm_classes_html = build_ppe_classes_html(sm_class_names, sm_selected_classes, "sm-cls")

    template = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__APP_TITLE__</title>
__THEME_SCRIPT__
<style>
__THEME__
html,body{height:100%;}
body{margin:0;overflow:hidden;}

.app{height:100vh;display:flex;flex-direction:column;padding:12px 16px 16px;gap:12px;}

/* Top bar */
.topbar{display:flex;align-items:center;gap:16px;flex:0 0 auto;}
.brand{display:flex;align-items:center;gap:10px;flex:0 0 auto;}
.brand-dot{width:9px;height:9px;border-radius:50%;background:var(--muted-2);transition:background .2s,box-shadow .2s;}
.brand-dot.on{background:var(--go);box-shadow:0 0 12px 1px var(--ok-soft);animation:pulse 1.8s ease-in-out infinite;}
@keyframes pulse{50%{box-shadow:0 0 5px 0 var(--ok-soft);}}
.brand-name{font-weight:800;letter-spacing:2.5px;font-size:15px;}
.brand-sub{font-family:var(--mono);font-size:10px;letter-spacing:2px;color:var(--muted);border:1px solid var(--line-2);border-radius:5px;padding:2px 7px;}
.statusrail{display:flex;align-items:center;gap:8px;flex:1 1 auto;min-width:0;overflow:hidden;}
.pill{font-family:var(--mono);font-size:12px;letter-spacing:.4px;color:var(--muted);background:var(--panel-2);border:1px solid var(--line);border-radius:999px;padding:5px 12px;white-space:nowrap;}
.pill.playing{color:var(--ok-fg);background:var(--go);border-color:var(--go);font-weight:700;}
.pill.err{background:var(--danger-soft);border-color:var(--bad);color:var(--danger-fg);max-width:38vw;overflow:hidden;text-overflow:ellipsis;}
.ghost-btn{font-family:var(--sans);font-size:12.5px;color:var(--muted);cursor:pointer;background:var(--panel-2);border:1px solid var(--line-2);border-radius:8px;padding:8px 14px;flex:0 0 auto;transition:border-color .15s,color .15s;}
.ghost-btn:hover{color:var(--text);border-color:var(--hivis);}

/* Layout */
.layout{flex:1 1 auto;min-height:0;display:grid;grid-template-columns:minmax(0,1fr) 372px;gap:14px;transition:grid-template-columns .28s cubic-bezier(.4,0,.2,1);}
.layout.right-hidden{grid-template-columns:minmax(0,1fr) 0;}
.layout.right-hidden .right-col{opacity:0;pointer-events:none;}
.stage-col{min-width:0;min-height:0;display:flex;}
.right-col{min-width:0;min-height:0;display:flex;flex-direction:column;gap:14px;overflow:auto;padding-right:2px;transition:opacity .2s;}

/* Stage */
.stage{flex:1 1 auto;min-width:0;min-height:0;display:flex;flex-direction:column;gap:12px;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:12px;}
.video-wrap{position:relative;flex:1 1 auto;min-height:0;display:flex;border-radius:12px;overflow:hidden;background:var(--stage);border:1px solid var(--border);}
.video{width:100%;height:100%;object-fit:contain;display:block;cursor:default;}
.video.pick-mode{cursor:crosshair;}

.toolbar{position:absolute;top:12px;right:12px;z-index:10;display:flex;flex-direction:column;gap:6px;width:182px;background:var(--overlay);backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px);border:1px solid var(--line-2);border-radius:11px;padding:9px;}
.tool-btn > span:first-child{white-space:nowrap;}
.toolbar-head{display:flex;align-items:center;justify-content:space-between;gap:8px;cursor:pointer;}
.toolbar-head:hover .toolbar-label,.toolbar-head:hover .toolbar-toggle{color:var(--hivis);}
.toolbar-head:hover .toolbar-toggle{border-color:var(--hivis);}
.toolbar-toggle{pointer-events:none;}
.toolbar-label{font-size:9.5px;letter-spacing:1.6px;color:var(--muted);text-transform:uppercase;padding:0 2px 1px;}
.toolbar-toggle{width:20px;height:20px;flex:0 0 auto;padding:0;line-height:1;border-radius:6px;cursor:pointer;
  background:var(--panel-2);border:1px solid var(--line-2);color:var(--muted);font-size:14px;
  display:inline-flex;align-items:center;justify-content:center;transition:border-color .14s,color .14s;}
.toolbar-toggle:hover{border-color:var(--hivis);color:var(--hivis);}
.toolbar-body{display:flex;flex-direction:column;gap:6px;margin-top:6px;}
.toolbar.collapsed{width:auto;}
.toolbar.collapsed .toolbar-body{display:none;}
.toolbar.collapsed .toolbar-head{margin:0;}

/* Fullscreen video + overlay transport controls */
.fs-btn{position:absolute;bottom:12px;right:12px;z-index:12;width:34px;height:34px;padding:0;border-radius:9px;cursor:pointer;
  background:var(--overlay);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);border:1px solid var(--line-2);
  color:var(--text);font-size:16px;display:inline-flex;align-items:center;justify-content:center;
  transition:border-color .14s,color .14s,opacity .25s;}
.fs-btn:hover{border-color:var(--hivis);color:var(--hivis);}
.fs-controls{display:none;}
.fs-t{width:auto;height:44px;padding:0 18px;border-radius:10px;cursor:pointer;font-size:14.5px;color:var(--text);
  background:var(--overlay-solid);border:1px solid var(--line-2);display:inline-flex;align-items:center;justify-content:center;
  transition:border-color .14s,background .14s;}
.fs-t:hover{border-color:var(--hivis);background:var(--overlay-strong);}
.fs-play{width:60px;height:60px;flex:0 0 auto;padding:0;border:none;border-radius:50%;color:var(--ok-fg);cursor:pointer;
  background:radial-gradient(120% 120% at 30% 25%, var(--go), var(--go-deep));box-shadow:0 6px 22px var(--ok-soft);
  display:inline-flex;align-items:center;justify-content:center;transition:background .2s,filter .12s;}
.fs-play:hover{filter:brightness(1.06);}
.fs-play.playing{color:var(--accent-fg);background:radial-gradient(120% 120% at 30% 25%, var(--hivis), var(--accent-hover));box-shadow:0 6px 22px var(--accent-soft);}
.video-wrap:fullscreen{background:var(--canvas-bg);border-radius:0;border:none;}
.video-wrap:fullscreen .toolbar{display:none;}
.video-wrap:fullscreen .fs-controls{display:flex;align-items:center;gap:14px;position:absolute;left:50%;bottom:34px;
  transform:translateX(-50%);z-index:14;padding:14px 18px;border-radius:18px;
  background:var(--overlay);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border:1px solid var(--border);
  box-shadow:var(--sh-lg);transition:opacity .25s;}
.video-wrap:fullscreen.fs-idle .fs-controls,
.video-wrap:fullscreen.fs-idle .fs-btn{opacity:0;pointer-events:none;}
.video-wrap:fullscreen.fs-idle{cursor:none;}
.tool-btn{display:flex;align-items:center;justify-content:space-between;gap:8px;width:100%;font-family:var(--sans);font-size:12.5px;color:var(--text);cursor:pointer;background:var(--panel-2);border:1px solid var(--line-2);border-radius:8px;padding:7px 9px;transition:border-color .14s,background .14s,transform .05s;}
.tool-btn:hover{border-color:var(--hivis);background:var(--panel-3);}
.tool-btn:active{transform:translateY(1px);}
.tool-btn.active{border-color:var(--hivis);background:var(--hivis-soft);color:var(--hivis);}
.tool-btn.dl{border-color:var(--border-2);color:var(--accent);}
.tool-btn.dl:hover{border-color:var(--accent);background:var(--panel-3);}
.tool-btn.cancel{border-color:var(--bad);color:var(--danger-fg);}
.kbd{font-family:var(--mono);font-size:10px;color:var(--muted);background:var(--bg);border:1px solid var(--line-2);border-bottom-width:2px;border-radius:4px;padding:1px 5px;min-width:20px;text-align:center;}
.tool-div{height:1px;background:var(--line);margin:2px 0;}

.draw-overlay{position:absolute;inset:0;z-index:8;display:none;cursor:crosshair;}
.draw-box{position:absolute;border:2px solid var(--hivis);background:var(--accent-soft);pointer-events:none;display:none;}
.video-note{position:absolute;left:12px;bottom:12px;z-index:11;max-width:60%;font-size:12.5px;color:var(--text);background:var(--overlay-strong);border:1px solid var(--line-2);border-left:3px solid var(--hivis);border-radius:8px;padding:8px 11px;display:none;}

/* Dock */
.dock{flex:0 0 auto;display:flex;flex-direction:column;gap:11px;}
.transport{display:flex;align-items:center;justify-content:center;gap:10px;}
.t-btn{width:auto;min-width:74px;height:40px;padding:0 14px;font-family:var(--sans);font-size:12.5px;color:var(--text);cursor:pointer;background:var(--panel-2);border:1px solid var(--line-2);border-radius:9px;display:inline-flex;align-items:center;justify-content:center;gap:6px;transition:border-color .14s,background .14s,transform .05s;}
.t-btn:hover{border-color:var(--hivis);background:var(--panel-3);}
.t-btn:active{transform:translateY(1px);}
.frame-jump{display:inline-flex;align-items:center;gap:6px;margin-left:6px;}
.frame-jump label{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--text-muted);}
.frame-jump input{width:84px;height:40px;padding:0 10px;font-family:var(--mono,monospace);font-size:13px;text-align:center;
  color:var(--text);background:var(--panel-2);border:1px solid var(--line-2);border-radius:9px;
  transition:border-color .14s,box-shadow .14s;appearance:textfield;-moz-appearance:textfield;}
.frame-jump input::-webkit-outer-spin-button,.frame-jump input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0;}
.frame-jump input:focus{outline:none;border-color:var(--hivis);}
.play-btn{width:56px;min-width:56px;height:56px;flex:0 0 auto;padding:0;border:none;border-radius:50%;color:var(--ok-fg);cursor:pointer;background:radial-gradient(120% 120% at 30% 25%, var(--go), var(--go-deep));box-shadow:0 6px 20px var(--ok-soft),inset 0 1px 0 var(--border-2);display:inline-flex;align-items:center;justify-content:center;transition:transform .1s,box-shadow .18s,background .2s,filter .12s;}
.play-btn:hover{filter:brightness(1.06);transform:translateY(-1px);}
.play-btn:active{transform:translateY(0);}
.play-btn.playing{color:var(--accent-fg);background:radial-gradient(120% 120% at 30% 25%, var(--hivis), var(--accent-hover));box-shadow:0 6px 20px var(--accent-soft),inset 0 1px 0 var(--border-2);}
.play-btn svg{display:block;}
/* Download-progress ring around the play button (shown while a URL downloads). */
.play-wrap{position:relative;width:56px;height:56px;flex:0 0 auto;display:inline-flex;align-items:center;justify-content:center;}
/* The download ring/percent are pure decoration layered over the play button.
   They must NEVER intercept the button's clicks (pointer-events:none on the svg
   AND its children), and are shown ONLY while downloading via a class on
   .play-wrap — not via the .hidden DOM property, which doesn't reflect on SVG. */
.dl-ring{position:absolute;top:50%;left:50%;width:68px;height:68px;transform:translate(-50%,-50%) rotate(-90deg);z-index:2;display:none;}
.dl-ring,.dl-ring *,.dl-pct{pointer-events:none !important;}
.dl-ring-bg{fill:none;stroke:var(--border-2);stroke-width:4;}
.dl-ring-fg{fill:none;stroke:var(--hivis);stroke-width:4;stroke-linecap:round;
  stroke-dasharray:195;stroke-dashoffset:195;transition:stroke-dashoffset .2s linear;}
.dl-pct{position:absolute;left:50%;bottom:-16px;transform:translateX(-50%);font-family:var(--mono);
  font-size:10px;color:var(--hivis);white-space:nowrap;z-index:2;display:none;}
.play-wrap.downloading .dl-ring,.play-wrap.downloading .dl-pct{display:block;}
.play-wrap.downloading .play-btn{filter:brightness(.7);}

.seek-row{display:flex;align-items:center;gap:12px;}
.time-readout{font-family:var(--mono);font-size:12px;color:var(--muted);white-space:nowrap;}
.slider-wrap{position:relative;flex:1 1 auto;height:32px;display:flex;align-items:center;}
.slider-wrap input[type=range]{position:relative;z-index:3;width:100%;-webkit-appearance:none;appearance:none;background:transparent;height:22px;margin:0;padding:0;}
.slider-wrap input[type=range]::-webkit-slider-runnable-track{background:transparent;height:22px;}
.slider-wrap input[type=range]::-moz-range-track{background:transparent;}
.slider-wrap input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;margin-top:3px;border-radius:50%;background:var(--surface);border:3px solid var(--hivis);cursor:pointer;box-shadow:var(--sh-sm);}
.slider-wrap input[type=range]::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:var(--surface);border:3px solid var(--hivis);cursor:pointer;}
.seek-track{position:absolute;left:0;right:0;top:50%;transform:translateY(-50%);height:6px;border-radius:5px;background:var(--surface-2);overflow:hidden;z-index:0;}
.seek-seg{position:absolute;top:0;height:100%;background:var(--border-2);}
.seek-played{position:absolute;left:0;top:0;height:100%;width:0%;background:linear-gradient(90deg,var(--hivis),var(--hivis-2));}
.buffer-pill{font-family:var(--mono);font-size:11px;color:var(--muted);white-space:nowrap;min-width:118px;text-align:right;}
.buffer-pill.ready{color:var(--go);}

/* Right cards */
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px;}
.card-head{display:flex;align-items:center;gap:8px;margin:0 0 12px;}
.card-head .tick{width:3px;height:14px;border-radius:2px;background:var(--hivis);}
.card-title{font-size:11px;letter-spacing:1.4px;text-transform:uppercase;color:var(--text-muted);font-weight:700;}
.card-head{cursor:pointer;}
.card-head:hover .card-title{color:var(--hivis);}
.card-head:hover .card-min{border-color:var(--hivis);color:var(--hivis);}
.card-min{margin-left:auto;width:22px;height:22px;flex:0 0 auto;padding:0;line-height:1;border-radius:6px;
  pointer-events:none;background:var(--panel-2);border:1px solid var(--line-2);color:var(--muted);font-size:15px;
  display:inline-flex;align-items:center;justify-content:center;transition:border-color .14s,color .14s;}
.card.collapsed{padding-bottom:14px;}
.card.collapsed .card-head{margin-bottom:0;}
.card.collapsed > *:not(.card-head){display:none;}

/* Ask AI container */
textarea{width:100%;min-height:58px;padding:9px 11px;font-family:var(--sans);font-size:13px;color:var(--text);
  background:var(--panel-2);border:1px solid var(--line-2);border-radius:8px;outline:none;resize:vertical;
  transition:border-color .14s,box-shadow .14s;}
textarea:focus{border-color:var(--hivis);box-shadow:0 0 0 3px var(--hivis-soft);}
.ai-send{width:100%;min-height:40px;border:none;border-radius:9px;cursor:pointer;font-weight:700;font-size:14px;
  color:var(--ok-fg);background:linear-gradient(180deg,var(--go),var(--go-deep));box-shadow:0 4px 14px var(--ok-soft);
  transition:filter .12s,background .2s;}
.ai-send:hover{filter:brightness(1.06);}
.ai-send.picking{background:linear-gradient(180deg,var(--hivis),var(--accent-hover));color:var(--accent-fg);}
.ai-note{font-size:12px;color:var(--muted);line-height:1.5;}
.ai-note b{color:var(--text-muted);}
.ai-result{font-size:13px;line-height:1.55;color:var(--text);background:var(--panel);border:1px solid var(--line);
  border-radius:9px;padding:10px 11px;white-space:pre-wrap;max-height:300px;overflow:auto;}
.ai-result.err{border-color:var(--bad);color:var(--danger-fg);}
.ai-result.loading{color:var(--muted);}
.ai-sec{border:1px solid var(--line-2);border-radius:10px;padding:11px;margin-bottom:9px;background:var(--panel-2);}
.ai-sec-head{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:7px;}
.ai-sec-label{font-size:12px;font-weight:700;color:var(--hivis);letter-spacing:.3px;text-transform:capitalize;}
.ai-sec-save{width:auto;height:auto;min-height:0;padding:4px 12px;font-size:11px;font-weight:700;border-radius:6px;cursor:pointer;
  background:var(--panel);border:1px solid var(--line-2);color:var(--muted);
  transition:border-color .14s,color .14s,background .14s;}
.ai-sec-save:hover{border-color:var(--hivis);color:var(--hivis);}
.ai-sec-save.dirty{border-color:var(--hivis);color:var(--hivis);background:var(--hivis-soft);}
.ai-sec-prompt{width:100%;min-height:46px;padding:8px 10px;font-family:var(--sans);font-size:12.5px;color:var(--text);
  background:var(--panel);border:1px solid var(--line-2);border-radius:7px;outline:none;resize:vertical;
  transition:border-color .14s,box-shadow .14s;}
.ai-sec-prompt:focus{border-color:var(--hivis);box-shadow:0 0 0 3px var(--hivis-soft);}
.ai-sec-result{margin-top:8px;}
.ai-empty{font-size:12.5px;color:var(--muted);padding:8px 2px;line-height:1.5;}
.stack{display:flex;flex-direction:column;gap:11px;}
.field{display:flex;flex-direction:column;gap:5px;min-width:0;}
.field > label{color:var(--muted);font-size:11px;letter-spacing:.3px;}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.sub-label{font-size:10px;letter-spacing:1px;text-transform:uppercase;color:var(--muted-2);margin:3px 0 -2px;}

input,select{width:100%;height:36px;padding:8px 10px;font-family:var(--sans);font-size:13px;color:var(--text);background:var(--panel-2);border:1px solid var(--line-2);border-radius:8px;outline:none;transition:border-color .14s,box-shadow .14s;}
input[type=number]{font-family:var(--mono);}
input:focus,select:focus{border-color:var(--hivis);box-shadow:0 0 0 3px var(--hivis-soft);}
/* Custom-styled dropdowns: hide the native arrow, draw a hi-vis chevron. */
select{cursor:pointer;-webkit-appearance:none;appearance:none;padding-right:34px;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1.5 6 6.5 11 1.5' fill='none' stroke='%23ffb200' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;}
select:hover{border-color:var(--line-2);}
select option{background:var(--panel-2);color:var(--text);}
input[type=checkbox]{width:17px;height:17px;accent-color:var(--hivis);cursor:pointer;}

/* Fully-themed custom dropdown (progressively enhances the native <select>). */
.dd{position:relative;min-width:0;}
.model-row .dd{flex:1 1 auto;min-width:0;}
.dd-btn{width:100%;height:36px;padding:8px 34px 8px 11px;text-align:left;font-family:var(--sans);font-size:13px;
  color:var(--text);background:var(--panel-2);border:1px solid var(--line-2);border-radius:8px;cursor:pointer;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;transition:border-color .14s,box-shadow .14s;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1.5 6 6.5 11 1.5' fill='none' stroke='%23ffb200' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 12px center;}
.dd-btn:hover{border-color:var(--hivis);}
.dd.open .dd-btn{border-color:var(--hivis);box-shadow:0 0 0 3px var(--hivis-soft);
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 6.5 6 1.5 11 6.5' fill='none' stroke='%23ffb200' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");}
.dd-menu{position:fixed;z-index:60;max-height:300px;background:var(--panel-3);border:1px solid var(--line-2);
  border-radius:10px;padding:6px;box-shadow:var(--sh-lg);display:none;flex-direction:column;}
.dd-menu.open{display:flex;}
.dd-search{width:100%;height:32px;margin-bottom:6px;padding:6px 9px;font-size:12.5px;color:var(--text);
  background:var(--panel);border:1px solid var(--line-2);border-radius:7px;outline:none;}
.dd-search:focus{border-color:var(--hivis);}
.dd-scroll{overflow-y:auto;}
.dd-opt{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:7px;font-size:13px;color:var(--text);
  cursor:pointer;white-space:nowrap;}
.dd-opt::before{content:"";width:12px;flex:0 0 auto;color:var(--hivis);font-size:11px;text-align:center;}
.dd-opt:hover{background:var(--hivis-soft);color:var(--hivis);}
.dd-opt.sel{color:var(--hivis);}
.dd-opt.sel::before{content:"✓";}

/* Source accordion: only the open section's input is the active source. */
.src{display:flex;flex-direction:column;gap:8px;}
.src-sec{border:1px solid var(--line-2);border-radius:10px;overflow:hidden;background:var(--panel-2);}
.src-head{display:flex;align-items:center;justify-content:space-between;width:100%;height:auto;
  padding:11px 12px;background:transparent;border:none;border-radius:0;color:var(--muted);
  font-size:13px;font-weight:600;cursor:pointer;transition:color .14s,background .14s;}
.src-head:hover{color:var(--text);}
.src-h-left{display:flex;align-items:center;gap:9px;}
.src-ico{font-size:14px;filter:grayscale(1) opacity(.7);}
.src-chevron{color:var(--muted);transition:transform .2s,color .2s;font-size:11px;}
.src-sec.open{border-color:var(--hivis);background:var(--panel-3);}
.src-sec.open .src-head{color:var(--hivis);}
.src-sec.open .src-ico{filter:none;}
.src-sec.open .src-chevron{transform:rotate(180deg);color:var(--hivis);}
.src-body{display:none;padding:0 12px 12px;flex-direction:column;gap:10px;}
.src-sec.open .src-body{display:flex;}
.url-row{display:flex;gap:8px;align-items:stretch;}
.url-row input{flex:1 1 auto;min-width:0;}
.url-play{flex:0 0 auto;width:auto;height:36px;padding:0 16px;border:none;border-radius:8px;cursor:pointer;
  color:var(--ok-fg);font-weight:700;font-size:13px;white-space:nowrap;
  background:linear-gradient(180deg,var(--go),var(--go-deep,var(--ok-deep)));box-shadow:0 3px 12px var(--ok-soft);}
.url-play:hover{filter:brightness(1.06);}
.url-play:active{transform:translateY(1px);}
.url-hint{font-size:11px;color:var(--muted-2);}
.toggle-row{display:flex;align-items:center;gap:10px;cursor:pointer;font-size:13px;color:var(--text);background:var(--panel-2);border:1px solid var(--line-2);border-radius:9px;padding:10px 12px;transition:border-color .14s;}
.toggle-row:hover{border-color:var(--hivis);}
.discovered{font-family:var(--mono);font-size:11px;color:var(--muted);white-space:pre-wrap;line-height:1.6;max-height:120px;overflow:auto;background:var(--panel-2);border:1px solid var(--line);border-radius:9px;padding:9px 11px;}

/* Class picker */
.model-row{display:flex;gap:8px;align-items:stretch;}
.model-row select{flex:1 1 auto;min-width:0;}
.cls-dropdown{position:relative;flex:0 0 auto;}
.cls-btn{height:36px;white-space:nowrap;cursor:pointer;font-family:var(--mono);font-size:12px;color:var(--text);background:var(--panel-2);border:1px solid var(--line-2);border-radius:8px;padding:0 12px;transition:border-color .14s,color .14s;}
.cls-btn:hover{border-color:var(--hivis);color:var(--hivis);}
.cls-panel{position:absolute;right:0;top:calc(100% + 7px);z-index:40;width:232px;max-height:300px;overflow:hidden;display:flex;flex-direction:column;background:var(--panel-3);border:1px solid var(--line-2);border-radius:11px;padding:8px;box-shadow:var(--sh-lg);}
.cls-panel[hidden]{display:none;}
.cls-search{width:100%;height:34px;margin-bottom:7px;padding:7px 9px;font-size:12.5px;color:var(--text);background:var(--panel);border:1px solid var(--line-2);border-radius:7px;outline:none;}
.cls-search:focus{border-color:var(--hivis);}
.cls-scroll{overflow-y:auto;}
.cls-row{display:flex;align-items:center;gap:9px;padding:6px 7px;border-radius:7px;font-size:12.5px;cursor:pointer;}
.cls-row:hover{background:var(--hivis-soft);color:var(--hivis);}
.cls-all{border-bottom:1px solid var(--line);border-radius:0;margin-bottom:5px;padding-bottom:8px;font-weight:700;color:var(--hivis);}
.cls-empty{color:var(--muted);font-size:12.5px;padding:8px;}
/* PPE Classes card: 2-column checkbox grid + "All" toggle. */
.cls-all-row{display:flex;align-items:center;gap:9px;font-size:12.5px;font-weight:700;color:var(--hivis);cursor:pointer;
  padding:2px 6px 9px;border-bottom:1px solid var(--line);}
.cls-grid{display:grid;grid-template-columns:1fr 1fr;gap:1px 12px;max-height:300px;overflow-y:auto;}
.cls-grid .cls-row{text-transform:capitalize;}
.cls-grid input[type=checkbox]{flex:0 0 auto;}

/* Toast */
.toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%) translateY(10px);z-index:2000;font-size:13px;font-weight:600;color:var(--ok-fg);background:var(--go);border:1px solid var(--go-deep);border-radius:10px;padding:10px 16px;opacity:0;pointer-events:none;box-shadow:var(--sh-md);transition:opacity .16s,transform .16s;}
.toast.show{opacity:1;transform:translateX(-50%) translateY(0);}
.toast.error{color:#fff;background:var(--danger);border-color:var(--danger);}

:focus-visible{outline:2px solid var(--hivis);outline-offset:2px;}

@media (max-width:1180px){ .layout{grid-template-columns:minmax(0,1fr) 330px;} }
@media (max-width:920px){
  body{overflow:auto;}
  .app{height:auto;min-height:100vh;}
  .layout{grid-template-columns:1fr;} .layout.right-hidden{grid-template-columns:1fr;}
  .video-wrap{min-height:52vh;}
}
@media (prefers-reduced-motion:reduce){ *{animation:none !important;transition:none !important;} }
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand">
      <span class="brand-dot" id="live_dot"></span>
      <span class="brand-name">INFERENCE</span>
      <span class="brand-sub">PPE · SM</span>
    </div>
    <div class="statusrail">
      <span class="pill" id="pill_state">Connecting…</span>
      <span class="pill mono" id="pill_source" title="Active source">—</span>
      <span class="pill mono" id="pill_frame">frame —</span>
      <span class="pill mono" id="pill_persons">— persons</span>
      <span class="pill err" id="pill_error" hidden></span>
    </div>
    <a class="home-btn" href="/home" title="Back to the app picker"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5"/><path d="M9.5 21v-6h5v6"/></svg>Suite</a>
    __THEME_BUTTON__
    <button id="right_toggle_btn" class="ghost-btn" onclick="toggleRightPanel()">Hide panel</button>
  </header>

  <div class="layout" id="main_layout">
    <section class="stage-col">
      <div class="stage">
        <div class="video-wrap" id="video_wrap">
          <img id="video_stream" class="video" src="/stream.mjpg" alt="Live inference stream" />
          <div id="draw_overlay" class="draw-overlay"><div id="draw_box" class="draw-box"></div></div>
          <button class="fs-btn" id="fs_btn" onclick="toggleFullscreen()" title="Fullscreen (double-click video)" aria-label="Toggle fullscreen">⛶</button>
          <div class="fs-controls" id="fs_controls">
            <button class="fs-t" onclick="seekBy(-5,'sec')" title="Back 5 seconds">« 5s</button>
            <button class="fs-t" onclick="seekBy(-1,'frame')" title="Previous frame">‹ Frame</button>
            <button class="fs-play" id="fs_play" onclick="togglePlayPause()" title="Play / Pause" aria-label="Play or pause"></button>
            <button class="fs-t" onclick="seekBy(1,'frame')" title="Next frame">Frame ›</button>
            <button class="fs-t" onclick="seekBy(5,'sec')" title="Forward 5 seconds">5s »</button>
          </div>
          <div class="toolbar collapsed" id="capture_toolbar">
            <div class="toolbar-head" onclick="toggleCapture()" title="Show / hide capture tools">
              <span class="toolbar-label">Capture</span>
              <button type="button" class="toolbar-toggle" id="capture_toggle" tabindex="-1">+</button>
            </div>
            <div class="toolbar-body">
              <button id="crop_person_btn" class="tool-btn" onclick="togglePersonCrop()"><span>Crop person</span><span class="kbd">P</span></button>
              <button class="tool-btn" onclick="cropFrame()"><span>Save frame</span><span class="kbd">F</span></button>
              <button class="tool-btn" onclick="cropBackground()"><span>Save background</span><span class="kbd">B</span></button>
              <button id="crop_manual_btn" class="tool-btn" onclick="toggleManualMode()"><span>Manual box</span><span class="kbd">M</span></button>
              <button id="crop_cancel_btn" class="tool-btn cancel" style="display:none;" onclick="closePersonCrop()"><span>Cancel</span><span class="kbd">Esc</span></button>
              <div class="tool-div"></div>
              <button id="download_video_btn" class="tool-btn dl" onclick="downloadInferenceVideo()" title="Render the whole video with detections and download it"><span>⬇ Download video</span></button>
            </div>
          </div>
          <div id="video_action_note" class="video-note"></div>
        </div>

        <div class="dock">
          <div class="transport">
            <button class="t-btn" onclick="seekBy(-5,'sec')" title="Back 5 seconds">« 5s</button>
            <button class="t-btn" onclick="seekBy(-1,'frame')" title="Previous frame (Left arrow)">‹ Frame</button>
            <div class="play-wrap">
              <svg class="dl-ring" id="dl_ring" viewBox="0 0 68 68">
                <circle class="dl-ring-bg" cx="34" cy="34" r="31"></circle>
                <circle class="dl-ring-fg" id="dl_ring_fg" cx="34" cy="34" r="31"></circle>
              </svg>
              <span class="dl-pct" id="dl_pct"></span>
              <button id="play_pause_btn" class="play-btn" onclick="togglePlayPause()" title="Play / Pause (Space)" aria-label="Play or pause"></button>
            </div>
            <button class="t-btn" onclick="seekBy(1,'frame')" title="Next frame (Right arrow)">Frame ›</button>
            <button class="t-btn" onclick="seekBy(5,'sec')" title="Forward 5 seconds">5s »</button>
            <span class="frame-jump" title="Type a frame number and press Enter to jump there">
              <label for="frame_jump">Frame</label>
              <input id="frame_jump" type="number" min="0" step="1" placeholder="#" inputmode="numeric" />
            </span>
          </div>
          <div class="seek-row">
            <span class="time-readout" id="time_cur">0:00</span>
            <div class="slider-wrap">
              <div class="seek-track"><div id="seek_buffers"></div><div class="seek-played" id="seek_played"></div></div>
              <input id="seek_slider" type="range" min="0" max="0" step="1" value="0" title="Drag to seek" />
            </div>
            <span class="time-readout" id="time_total">0:00</span>
            <span class="buffer-pill" id="buffer_pill"></span>
          </div>
        </div>
      </div>
    </section>

    <aside class="right-col">
      <div class="card collapsed" id="models_card">
        <div class="card-head"><span class="tick"></span><span class="card-title">Models</span><button type="button" class="card-min" onclick="toggleCard('models_card')" title="Expand">+</button></div>
        <div class="stack">
          <div class="field"><label for="person_model_path">Person model</label><select id="person_model_path">__PERSON_OPTIONS__</select></div>
          <div class="field"><label for="ppe_model_path">PPE model</label><select id="ppe_model_path">__PPE_OPTIONS__</select></div>
          <div class="field"><label for="sm_model_path">SM model</label><select id="sm_model_path">__SM_OPTIONS__</select></div>
          <label class="toggle-row" for="ppe_inside_person"><input id="ppe_inside_person" type="checkbox" __PPE_INSIDE_CHECKED__ />Detect PPE only inside person box</label>
        </div>
      </div>

      <div class="card" id="classes_card">
        <div class="card-head"><span class="tick"></span><span class="card-title">PPE Classes</span><button type="button" class="card-min" onclick="toggleCard('classes_card')" title="Collapse">–</button></div>
        <div class="stack">
          <label class="cls-all-row"><input type="checkbox" id="ppe_cls_all" /> All classes</label>
          <div id="ppe_classes_list" class="cls-grid">__PPE_CLASSES__</div>
        </div>
      </div>

      <div class="card" id="sm_classes_card">
        <div class="card-head"><span class="tick"></span><span class="card-title">SM Classes</span><button type="button" class="card-min" onclick="toggleCard('sm_classes_card')" title="Collapse">–</button></div>
        <div class="stack">
          <label class="cls-all-row"><input type="checkbox" id="sm_cls_all" /> All classes</label>
          <div id="sm_classes_list" class="cls-grid">__SM_CLASSES__</div>
        </div>
      </div>

      <div class="card" id="source_card">
        <div class="card-head"><span class="tick"></span><span class="card-title">Source &amp; Playback</span><button type="button" class="card-min" onclick="toggleCard('source_card')" title="Collapse">–</button></div>
        <div class="stack">
          <div class="src" id="source_accordion">
            <div class="src-sec" data-mode="file">
              <button type="button" class="src-head" onclick="setSourceMode('file')">
                <span class="src-h-left"><span class="src-ico">📁</span> Local file</span>
                <span class="src-chevron">▾</span>
              </button>
              <div class="src-body">
                <div class="field"><label for="video_folder">Video folder</label><input id="video_folder" value="__VIDEO_FOLDER__" /></div>
                <div class="field"><label for="selected_video">Video file</label><select id="selected_video"></select></div>
              </div>
            </div>
            <div class="src-sec" data-mode="url">
              <button type="button" class="src-head" onclick="setSourceMode('url')">
                <span class="src-h-left"><span class="src-ico">🔗</span> Stream URL</span>
                <span class="src-chevron">▾</span>
              </button>
              <div class="src-body">
                <div class="field">
                  <label for="video_url">Video / stream URL</label>
                  <div class="url-row">
                    <input id="video_url" value="__VIDEO_URL__" placeholder="https://…/video.mp4?…  — streamed, not downloaded" spellcheck="false" />
                    <button type="button" class="url-play" onclick="playUrl()" title="Stream this URL and start playing">▶ Play</button>
                  </div>
                  <div class="url-hint">Paste a link and hit Play — it streams live, no download.</div>
                </div>
              </div>
            </div>
          </div>
          <div class="sub-label">Confidence thresholds</div>
          <div class="grid-3">
            <div class="field"><label for="person_conf">Person</label><input id="person_conf" type="number" min="0" max="1" step="0.05" value="__PERSON_CONF__" /></div>
            <div class="field"><label for="ppe_conf">PPE</label><input id="ppe_conf" type="number" min="0" max="1" step="0.05" value="__PPE_CONF__" /></div>
            <div class="field"><label for="sm_conf">SM</label><input id="sm_conf" type="number" min="0" max="1" step="0.05" value="__SM_CONF__" /></div>
          </div>
          <div class="grid-2">
            <div class="field"><label for="frame_step">Frame step</label><input id="frame_step" type="number" min="1" max="60" step="1" value="__FRAME_STEP__" /></div>
            <div class="field"><label for="simulate_realtime">Realtime</label><select id="simulate_realtime"><option value="true" __RT_ENABLED_SEL__>Enabled</option><option value="false" __RT_DISABLED_SEL__>Disabled</option></select></div>
          </div>
        </div>
      </div>

      <div class="card collapsed" id="ai_card">
        <div class="card-head"><span class="tick"></span><span class="card-title">Ask AI</span><button type="button" class="card-min" onclick="toggleCard('ai_card')" title="Expand">+</button></div>
        <div class="stack">
          <div class="ai-note">One prompt per PPE class you selected under <b>Models</b>. Click Send, then click a person — each selected class's prompt runs on that crop.</div>
          <div id="ai_sections"></div>
          <button id="ai_send_btn" class="ai-send" onclick="startAiPick()">Send</button>
        </div>
      </div>
    </aside>
  </div>
</div>
<div id="save_toast" class="toast"></div>

<script>
__THEME_JS__

  let allVideos = __VIDEOS_JSON__;
  let selectedVideo = __SELECTED_VIDEO_JSON__;
  let sourceMode = "__SOURCE_MODE__";
  let suppressSliderUpdate = false;
  let uiPlaying = false;
  let controlPendingUntil = 0;
  let personPickMode = false;
  let manualDrawMode = false;
  let aiPickMode = false;
  let drawStart = null;
  let toastTimer = null;

  function setVideos(list, preferredVideo = null) {
    allVideos = list || [];
    const el = document.getElementById("selected_video");
    el.innerHTML = "";
    const target = preferredVideo || selectedVideo;
    allVideos.forEach(v => {
      const opt = document.createElement("option");
      opt.value = v; opt.text = v.split("/").pop();
      if (v === target) opt.selected = true;
      el.appendChild(opt);
    });
    if (el.options.length > 0 && !el.value) el.selectedIndex = 0;
    selectedVideo = el.value || "";
    if (el._dd) el._dd.refresh();   // keep the custom dropdown in sync
  }

  // Close every open custom dropdown so only one is ever open at a time.
  function closeAllPopups() {
    document.querySelectorAll(".dd-menu.open").forEach(m => m.classList.remove("open"));
    document.querySelectorAll(".dd.open").forEach(d => d.classList.remove("open"));
  }

  // Progressive enhancement: replace a native <select> with a themed dropdown.
  // The <select> stays in the DOM as the source of truth — we set its value and
  // fire a real "change" event, so every existing listener keeps working.
  function enhanceSelect(sel) {
    if (!sel || sel._dd) return;
    sel.style.display = "none";
    const dd = document.createElement("div"); dd.className = "dd";
    const btn = document.createElement("button"); btn.type = "button"; btn.className = "dd-btn";
    const menu = document.createElement("div"); menu.className = "dd-menu";
    const search = document.createElement("input");
    search.className = "dd-search"; search.placeholder = "Search…"; search.autocomplete = "off";
    const scroll = document.createElement("div"); scroll.className = "dd-scroll";
    menu.appendChild(search); menu.appendChild(scroll);
    sel.parentNode.insertBefore(dd, sel);
    dd.appendChild(btn);
    document.body.appendChild(menu);   // fixed-positioned, so panel overflow can't clip it

    const labelFor = () => { const o = sel.options[sel.selectedIndex]; return o ? o.text : "—"; };
    const renderBtn = () => { btn.textContent = labelFor(); btn.title = labelFor(); };
    function buildMenu() {
      scroll.innerHTML = "";
      Array.from(sel.options).forEach((o, i) => {
        const it = document.createElement("div");
        it.className = "dd-opt" + (i === sel.selectedIndex ? " sel" : "");
        it.textContent = o.text;
        it.addEventListener("click", () => {
          sel.selectedIndex = i;
          sel.dispatchEvent(new Event("change", { bubbles: true }));
          renderBtn(); close();
        });
        scroll.appendChild(it);
      });
      search.style.display = sel.options.length > 6 ? "block" : "none";   // search only for long lists
    }
    function filter(q) {
      q = (q || "").trim().toLowerCase();
      scroll.querySelectorAll(".dd-opt").forEach(it => {
        it.style.display = (!q || it.textContent.toLowerCase().includes(q)) ? "" : "none";
      });
    }
    function place() {
      const r = btn.getBoundingClientRect();
      menu.style.left = r.left + "px";
      menu.style.top = (r.bottom + 4) + "px";
      menu.style.width = r.width + "px";
    }
    function open() {
      closeAllPopups();   // only one dropdown open at a time
      buildMenu(); search.value = ""; filter(""); place();
      menu.classList.add("open"); dd.classList.add("open");
      if (search.style.display !== "none") setTimeout(() => search.focus(), 0);
      const on = scroll.querySelector(".dd-opt.sel"); if (on) on.scrollIntoView({ block: "nearest" });
    }
    function close() { menu.classList.remove("open"); dd.classList.remove("open"); }
    btn.addEventListener("click", (e) => { e.stopPropagation();
      menu.classList.contains("open") ? close() : open(); });
    search.addEventListener("click", (e) => e.stopPropagation());
    search.addEventListener("input", () => filter(search.value));
    document.addEventListener("click", (e) => { if (!menu.contains(e.target) && e.target !== btn) close(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    // Close on page/panel scroll (the button moves), but NOT when scrolling the
    // menu's own option list.
    window.addEventListener("scroll", (e) => { if (!menu.contains(e.target)) close(); }, true);
    window.addEventListener("resize", close);
    sel._dd = { refresh: () => { renderBtn(); if (menu.classList.contains("open")) buildMenu(); } };
    renderBtn();
  }

  function payloadFromInputs() {
    return {
      person_model_path: document.getElementById("person_model_path").value,
      ppe_model_path: document.getElementById("ppe_model_path").value,
      sm_model_path: document.getElementById("sm_model_path").value,
      video_folder: document.getElementById("video_folder").value,
      selected_video: document.getElementById("selected_video").value,
      video_url: document.getElementById("video_url").value,
      source_mode: sourceMode,
      person_conf: Number(document.getElementById("person_conf").value),
      ppe_conf: Number(document.getElementById("ppe_conf").value),
      sm_conf: Number(document.getElementById("sm_conf").value),
      frame_step: Number(document.getElementById("frame_step").value),
      simulate_realtime: document.getElementById("simulate_realtime").value === "true",
      ppe_inside_person: document.getElementById("ppe_inside_person").checked,
      ppe_classes: currentPpeClasses(),
      sm_classes: currentSmClasses(),
    };
  }

  function currentPpeClasses() {
    return Array.from(document.querySelectorAll("#ppe_classes_list .ppe-cls"))
      .filter(cb => cb.checked).map(cb => cb.value);
  }
  function updateAllToggle() {
    const boxes = Array.from(document.querySelectorAll("#ppe_classes_list .ppe-cls"));
    const total = boxes.length, on = boxes.filter(cb => cb.checked).length;
    const all = document.getElementById("ppe_cls_all");
    all.checked = total > 0 && on === total;
    all.indeterminate = on > 0 && on < total;
  }
  function buildClassList(names, selectedSet) {
    const list = document.getElementById("ppe_classes_list");
    if (!names || names.length === 0) { list.innerHTML = '<div class="cls-empty">No classes</div>'; updateAllToggle(); rebuildAiSections(); return; }
    list.innerHTML = names.map(n => {
      const checked = (selectedSet === null || selectedSet.has(n)) ? " checked" : "";
      const safe = n.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const val = n.replace(/"/g, "&quot;");
      return '<label class="cls-row"><input type="checkbox" class="ppe-cls" value="' + val + '"' + checked + ' />' + safe + '</label>';
    }).join("");
    updateAllToggle();
    rebuildAiSections();
  }
  async function reloadPpeClasses(modelPath) {
    try {
      const res = await fetch("/api/ppe_classes?path=" + encodeURIComponent(modelPath));
      const data = await res.json();
      buildClassList(data.classes || [], null);
    } catch (e) { buildClassList([], null); }
  }

  // ---- SM model class picker (mirror of the PPE one) ----
  function currentSmClasses() {
    return Array.from(document.querySelectorAll("#sm_classes_list .sm-cls"))
      .filter(cb => cb.checked).map(cb => cb.value);
  }
  function updateSmAllToggle() {
    const boxes = Array.from(document.querySelectorAll("#sm_classes_list .sm-cls"));
    const total = boxes.length, on = boxes.filter(cb => cb.checked).length;
    const all = document.getElementById("sm_cls_all");
    all.checked = total > 0 && on === total;
    all.indeterminate = on > 0 && on < total;
  }
  function buildSmClassList(names, selectedSet) {
    const list = document.getElementById("sm_classes_list");
    if (!names || names.length === 0) { list.innerHTML = '<div class="cls-empty">No classes</div>'; updateSmAllToggle(); return; }
    list.innerHTML = names.map(n => {
      const checked = (selectedSet === null || selectedSet.has(n)) ? " checked" : "";
      const safe = n.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const val = n.replace(/"/g, "&quot;");
      return '<label class="cls-row"><input type="checkbox" class="sm-cls" value="' + val + '"' + checked + ' />' + safe + '</label>';
    }).join("");
    updateSmAllToggle();
  }
  async function reloadSmClasses(modelPath) {
    try {
      const res = await fetch("/api/ppe_classes?path=" + encodeURIComponent(modelPath));
      const data = await res.json();
      buildSmClassList(data.classes || [], null);
    } catch (e) { buildSmClassList([], null); }
  }

  // Class cards only exist while their model is actually selected.
  function updateClassCardVisibility() {
    const NONE = "__none__";
    document.getElementById("classes_card").style.display =
      document.getElementById("ppe_model_path").value === NONE ? "none" : "";
    document.getElementById("sm_classes_card").style.display =
      document.getElementById("sm_model_path").value === NONE ? "none" : "";
  }

  async function saveConfig(refreshVideos = false) {
    const res = await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payloadFromInputs()) });
    const data = await res.json();
    if (refreshVideos) setVideos(data.videos || [], data.selected_video || "");
    selectedVideo = data.selected_video || selectedVideo;
  }

  function control(action) {
    fetch("/api/control", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) }).catch(() => {});
  }
  function togglePlayPause() {
    uiPlaying = !uiPlaying;
    renderPlay(uiPlaying);
    reflectPlayState(uiPlaying);
    controlPendingUntil = Date.now() + 900;
    control(uiPlaying ? "start" : "pause");
  }
  function seekBy(delta, unit) {
    fetch("/api/seek", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: "relative", delta, unit }) }).catch(() => {});
  }
  async function seekToSlider() {
    const value = Number(document.getElementById("seek_slider").value);
    await fetch("/api/seek", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: "absolute", frame: value }) });
  }
  // Frame-number jump box: type a frame, Enter seeks straight to it.
  document.getElementById("frame_jump").addEventListener("keydown", (e) => {
    if (e.key !== "Enter") return;
    e.preventDefault();
    const el = e.target;
    let frame = Math.floor(Number(el.value));
    if (!isFinite(frame) || el.value.trim() === "") return;
    const max = Number(document.getElementById("seek_slider").max) || 0;
    frame = Math.max(0, Math.min(frame, max));
    fetch("/api/seek", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ mode: "absolute", frame }) }).catch(() => {});
    el.value = "";
    el.blur();
  });

  const ICON_PLAY = '<svg viewBox="0 0 24 24" width="22" height="22"><path d="M8 5.5v13a1 1 0 0 0 1.54.84l10-6.5a1 1 0 0 0 0-1.68l-10-6.5A1 1 0 0 0 8 5.5z" fill="currentColor"/></svg>';
  const ICON_PAUSE = '<svg viewBox="0 0 24 24" width="22" height="22"><rect x="6.5" y="5" width="4" height="14" rx="1.2" fill="currentColor"/><rect x="13.5" y="5" width="4" height="14" rx="1.2" fill="currentColor"/></svg>';
  const playBtn = document.getElementById("play_pause_btn");
  function renderPlay(playing) {
    playBtn.innerHTML = playing ? ICON_PAUSE : ICON_PLAY;
    playBtn.classList.toggle("playing", !!playing);
    const fp = document.getElementById("fs_play");
    if (fp) { fp.innerHTML = playing ? ICON_PAUSE : ICON_PLAY; fp.classList.toggle("playing", !!playing); }
  }
  function reflectPlayState(playing) {
    const pill = document.getElementById("pill_state");
    pill.textContent = playing ? "▸ Playing" : "❚❚ Paused";
    pill.classList.toggle("playing", !!playing);
    document.getElementById("live_dot").classList.toggle("on", !!playing);
  }

  function formatTime(totalSeconds) {
    if (!isFinite(totalSeconds) || totalSeconds < 0) totalSeconds = 0;
    const s = Math.floor(totalSeconds % 60), m = Math.floor(totalSeconds / 60);
    return m + ":" + String(s).padStart(2, "0");
  }

  async function refreshStatus() {
    const res = await fetch("/api/status");
    const s = await res.json();
    const slider = document.getElementById("seek_slider");
    const lastFrame = Math.max((s.total_frames || 1) - 1, 0);
    slider.max = String(lastFrame);
    if (!suppressSliderUpdate) slider.value = String(Math.max(s.frame_idx || 0, 0));
    if (Date.now() > controlPendingUntil) { uiPlaying = !!s.playing; renderPlay(uiPlaying); reflectPlayState(uiPlaying); }

    const fps = s.fps || 25;
    document.getElementById("time_cur").textContent = formatTime(Math.max(s.frame_idx, 0) / fps);
    document.getElementById("time_total").textContent = formatTime(lastFrame / fps);

    const cur = Math.max(s.frame_idx || 0, 0);
    const span = Math.max(lastFrame, 1);
    const step = Math.max(s.frame_step || 1, 1);
    updateSeekFill();  // fill tracks the slider thumb (incl. while dragging)
    const ranges = s.loaded_ranges || [];
    document.getElementById("seek_buffers").innerHTML = ranges.map(r => {
      const a = r[0], b = r[1];
      const left = Math.min(100, (a / span) * 100);
      const width = Math.max(0.4, Math.min(100, ((b - a + step) / span) * 100));
      return '<div class="seek-seg" style="left:' + left + '%;width:' + width + '%"></div>';
    }).join("");

    let aheadFrames = 0;
    for (const r of ranges) { const a = r[0], b = r[1]; if (a <= cur + step && cur <= b) { aheadFrames = Math.max(0, b - cur); break; } }
    const pill = document.getElementById("buffer_pill");
    const totalLoaded = s.loaded_count || 0;
    if (aheadFrames >= 2 * step) { pill.textContent = "● " + (aheadFrames / fps).toFixed(1) + "s buffered"; pill.classList.add("ready"); }
    else if (totalLoaded > 0) { pill.textContent = "○ loading… " + totalLoaded + " cached"; pill.classList.remove("ready"); }
    else { pill.textContent = "○ loading…"; pill.classList.remove("ready"); }

    const note = document.getElementById("video_action_note");
    if (personPickMode) { note.style.display = "block"; note.textContent = (s.available_person_count || 0) === 0 ? "No person in the frame." : "Click a person in the video to crop them."; }
    else if (manualDrawMode) { note.style.display = "block"; note.textContent = "Drag a box on the video, then release to save that region."; }
    else { note.style.display = "none"; note.textContent = ""; }

    const srcPill = document.getElementById("pill_source");
    srcPill.textContent = (s.source_is_url ? "🔗 " : "📁 ") + (s.source_label || "no source");
    srcPill.classList.toggle("playing", !!s.source_is_url);

    // Download-progress ring on the play button while a URL is being fetched.
    const dl = s.download;
    const ring = document.getElementById("dl_ring");
    const dlPct = document.getElementById("dl_pct");
    const playWrap = document.querySelector(".play-wrap");
    // Visibility is driven purely by the .downloading class (see CSS) — the SVG
    // ring can't be toggled via the .hidden property, and must never overlay the
    // play button's click target.
    if (dl && dl.status === "downloading") {
      const p = Math.max(0, Math.min(1, dl.progress || 0));
      document.getElementById("dl_ring_fg").style.strokeDashoffset = String(195 * (1 - p));
      dlPct.textContent = Math.round(p * 100) + "%";
      playWrap.classList.add("downloading");
    } else {
      playWrap.classList.remove("downloading");
    }
    document.getElementById("pill_frame").textContent = "frame " + Math.max(s.frame_idx, 0) + " / " + lastFrame;
    document.getElementById("pill_persons").textContent = (s.available_person_count || 0) + " persons";
    const errPill = document.getElementById("pill_error");
    if (s.last_error) { errPill.hidden = false; errPill.textContent = s.last_error; } else { errPill.hidden = true; errPill.textContent = ""; }
  }

  function setToolActive(id, on) { const b = document.getElementById(id); if (b) b.classList.toggle("active", !!on); }

  function togglePersonCrop() {
    if (manualDrawMode) closeActionModes();
    document.getElementById("video_stream").classList.add("pick-mode");
    personPickMode = true;
    document.getElementById("crop_cancel_btn").style.display = "flex";
    setToolActive("crop_person_btn", true);
  }
  function closeActionModes() {
    personPickMode = false; manualDrawMode = false; drawStart = null;
    document.getElementById("video_stream").classList.remove("pick-mode");
    document.getElementById("draw_overlay").style.display = "none";
    document.getElementById("draw_box").style.display = "none";
    document.getElementById("crop_cancel_btn").style.display = "none";
    setToolActive("crop_person_btn", false);
    setToolActive("crop_manual_btn", false);
    const note = document.getElementById("video_action_note"); note.style.display = "none"; note.textContent = "";
  }
  function closePersonCrop() { closeActionModes(); }
  function toggleManualMode() {
    if (personPickMode) closeActionModes();
    manualDrawMode = true; drawStart = null;
    document.getElementById("crop_cancel_btn").style.display = "flex";
    document.getElementById("draw_overlay").style.display = "block";
    setToolActive("crop_manual_btn", true);
  }

  function showToast(message, isError = false) {
    const toast = document.getElementById("save_toast");
    toast.textContent = message; toast.classList.toggle("error", !!isError); toast.classList.add("show");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), isError ? 2400 : 1200);
  }
  async function requestCrop(payload) {
    const res = await fetch("/api/crop", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    const data = await res.json();
    if (res.ok && data.ok) { showToast("Saved"); return true; }
    showToast(data.error || "Save failed", true); return false;
  }
  async function cropRequest(type, personIndex = null) { const p = { type }; if (personIndex !== null) p.person_index = personIndex; await requestCrop(p); await refreshStatus(); }
  async function cropPersonByPoint(xRatio, yRatio) { const ok = await requestCrop({ type: "person_point", x_ratio: xRatio, y_ratio: yRatio }); await refreshStatus(); return ok; }
  async function cropFrame() { await cropRequest("frame"); }
  async function cropBackground() { await cropRequest("background"); }

  let exportPollTimer = null;
  async function downloadInferenceVideo() {
    const btn = document.getElementById("download_video_btn");
    const res = await fetch("/api/export", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || !data.ok) { showToast(data.error || "Could not start export.", true); return; }
    btn.disabled = true; showToast("Rendering inference video…");
    if (exportPollTimer) clearInterval(exportPollTimer);
    exportPollTimer = setInterval(pollExportStatus, 700);
  }
  async function pollExportStatus() {
    const btn = document.getElementById("download_video_btn");
    let s;
    try { const res = await fetch("/api/export/status"); s = await res.json(); } catch (e) { return; }
    if (s.error) { clearInterval(exportPollTimer); exportPollTimer = null; btn.disabled = false; btn.innerHTML = "<span>⬇ Download video</span>"; showToast(s.error, true); return; }
    if (s.running) { const pct = s.total ? Math.floor((s.progress / s.total) * 100) : 0; btn.innerHTML = "<span>Rendering " + pct + "%</span>"; return; }
    if (s.done) { clearInterval(exportPollTimer); exportPollTimer = null; btn.disabled = false; btn.innerHTML = "<span>⬇ Download video</span>"; showToast("Video ready — downloading…"); window.location = "/download/video"; }
  }

  const videoStream = document.getElementById("video_stream");
  const drawOverlay = document.getElementById("draw_overlay");
  const drawBox = document.getElementById("draw_box");
  function clamp01(v) { return Math.max(0, Math.min(1, v)); }
  function imageContentRect() {
    const rect = videoStream.getBoundingClientRect();
    const nw = videoStream.naturalWidth, nh = videoStream.naturalHeight;
    if (!nw || !nh || rect.width <= 0 || rect.height <= 0) return rect;
    const scale = Math.min(rect.width / nw, rect.height / nh);
    const dw = nw * scale, dh = nh * scale;
    return { left: rect.left + (rect.width - dw) / 2, top: rect.top + (rect.height - dh) / 2, width: dw, height: dh };
  }
  function evtToImageRatio(evt) {
    const c = imageContentRect();
    if (c.width <= 0 || c.height <= 0) return null;
    return { x: clamp01((evt.clientX - c.left) / c.width), y: clamp01((evt.clientY - c.top) / c.height) };
  }
  videoStream.addEventListener("click", async (evt) => {
    const p = evtToImageRatio(evt);
    if (!p) return;
    if (aiPickMode) {
      const x = p.x, y = p.y;
      cancelAiPick();
      await runAiAnalyze(x, y);
      return;
    }
    if (!personPickMode) return;
    const ok = await cropPersonByPoint(p.x, p.y);
    if (ok) closeActionModes();
  });
  function updateDrawBoxDisplay(box) {
    if (!drawStart || !box) return;
    const overlayRect = drawOverlay.getBoundingClientRect();
    const c = imageContentRect();
    const offX = c.left - overlayRect.left;
    const offY = c.top - overlayRect.top;
    const x1 = offX + Math.min(box.x1, box.x2) * c.width;
    const y1 = offY + Math.min(box.y1, box.y2) * c.height;
    const x2 = offX + Math.max(box.x1, box.x2) * c.width;
    const y2 = offY + Math.max(box.y1, box.y2) * c.height;
    drawBox.style.display = "block";
    drawBox.style.left = x1 + "px";
    drawBox.style.top = y1 + "px";
    drawBox.style.width = Math.max(0, x2 - x1) + "px";
    drawBox.style.height = Math.max(0, y2 - y1) + "px";
  }
  drawOverlay.addEventListener("mousedown", (evt) => {
    if (!manualDrawMode) return;
    const p = evtToImageRatio(evt);
    if (!p) return;
    drawStart = { x: p.x, y: p.y };
    updateDrawBoxDisplay({ x1: p.x, y1: p.y, x2: p.x, y2: p.y });
  });
  drawOverlay.addEventListener("mousemove", (evt) => {
    if (!manualDrawMode || !drawStart) return;
    const p = evtToImageRatio(evt);
    if (!p) return;
    updateDrawBoxDisplay({ x1: drawStart.x, y1: drawStart.y, x2: p.x, y2: p.y });
  });
  drawOverlay.addEventListener("mouseup", async (evt) => {
    if (!manualDrawMode || !drawStart) return;
    const p = evtToImageRatio(evt);
    if (!p) return;
    const box = { x1: drawStart.x, y1: drawStart.y, x2: p.x, y2: p.y };
    const ok = await requestCrop({ type: "manual_box", x1_ratio: box.x1, y1_ratio: box.y1, x2_ratio: box.x2, y2_ratio: box.y2 });
    drawStart = null;
    drawBox.style.display = "none";
    await refreshStatus();
    if (ok) closeActionModes();
  });
  drawOverlay.addEventListener("mouseleave", () => {
    if (!manualDrawMode || !drawStart) return;
    drawStart = null;
    drawBox.style.display = "none";
  });

  document.addEventListener("keydown", async (evt) => {
    if (evt.ctrlKey || evt.metaKey || evt.altKey) return;
    const tag = (evt.target && evt.target.tagName ? evt.target.tagName.toLowerCase() : "");
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    const key = evt.key.toLowerCase();
    if (evt.key === "Escape") {
      if (aiPickMode) { evt.preventDefault(); cancelAiPick(); }
      else if (personPickMode || manualDrawMode) { evt.preventDefault(); closeActionModes(); }
      return;
    }
    if (evt.key === " " || evt.code === "Space") {
      if (tag === "button") return;
      evt.preventDefault();
      await togglePlayPause();
    } else if (key === "p") {
      evt.preventDefault();
      if (manualDrawMode) { closeActionModes(); togglePersonCrop(); }
      else if (personPickMode) { closeActionModes(); }
      else { togglePersonCrop(); }
    } else if (key === "b") { evt.preventDefault(); await cropBackground(); }
    else if (key === "f") { evt.preventDefault(); await cropFrame(); }
    else if (key === "m") { evt.preventDefault(); if (manualDrawMode) closeActionModes(); else toggleManualMode(); }
    else if (evt.key === "ArrowLeft") { evt.preventDefault(); await seekBy(-1, "frame"); }
    else if (evt.key === "ArrowRight") { evt.preventDefault(); await seekBy(1, "frame"); }
  });

  function toggleRightPanel() {
    const layout = document.getElementById("main_layout");
    const btn = document.getElementById("right_toggle_btn");
    const hidden = layout.classList.toggle("right-hidden");
    btn.textContent = hidden ? "Show panel" : "Hide panel";
  }

  function toggleCapture() {
    const tb = document.getElementById("capture_toolbar");
    const btn = document.getElementById("capture_toggle");
    const collapsed = tb.classList.toggle("collapsed");
    btn.textContent = collapsed ? "+" : "–";
    btn.title = collapsed ? "Show capture tools" : "Hide capture tools";
  }
  window.toggleCapture = toggleCapture;

  // Collapsible right-panel cards.
  function toggleCard(id) {
    const c = document.getElementById(id);
    const collapsed = c.classList.toggle("collapsed");
    const b = c.querySelector(".card-min");
    if (b) { b.textContent = collapsed ? "+" : "–"; b.title = collapsed ? "Expand" : "Collapse"; }
  }
  window.toggleCard = toggleCard;

  // Ask AI: one prompt section per selected PPE class. Send runs each selected
  // class's prompt on the clicked person crop and shows a result per class.
  // aiSaved: persisted, reusable prompts — the only thing that prefills a new run.
  // aiWorking: in-memory edits for the current session; used when running, but
  // NOT prefilled on reload unless you Save them.
  const AI_DEFAULTS = __AI_DEFAULTS__;   // shipped per-class default prompts
  let aiSaved = {};
  try { aiSaved = JSON.parse(localStorage.getItem("aiSavedPrompts") || "{}") || {}; } catch (e) { aiSaved = {}; }
  function persistSaved() { try { localStorage.setItem("aiSavedPrompts", JSON.stringify(aiSaved)); } catch (e) {} }
  let aiWorking = {};          // class name -> current (maybe-unsaved) text
  let aiSectionResults = {};   // class name -> its result element
  // What the box shows / runs: your live edit, else your saved prompt, else the shipped default.
  const promptFor = (cls) => (cls in aiWorking) ? aiWorking[cls]
    : (cls in aiSaved) ? aiSaved[cls] : (AI_DEFAULTS[cls] || "");
  // Baseline for the dirty highlight: your saved prompt, else the shipped default.
  const promptBaseline = (cls) => (cls in aiSaved) ? aiSaved[cls] : (AI_DEFAULTS[cls] || "");

  function rebuildAiSections() {
    const classes = currentPpeClasses();   // classes checked under Models
    const box = document.getElementById("ai_sections");
    if (!box) return;
    box.innerHTML = ""; aiSectionResults = {};
    if (!classes.length) {
      box.innerHTML = '<div class="ai-empty">No PPE classes selected. Pick classes in the <b>Classes</b> dropdown under Models to add prompt sections here.</div>';
      return;
    }
    classes.forEach(cls => {
      const sec = document.createElement("div"); sec.className = "ai-sec";
      const head = document.createElement("div"); head.className = "ai-sec-head";
      const label = document.createElement("span"); label.className = "ai-sec-label"; label.textContent = cls;
      const saveBtn = document.createElement("button"); saveBtn.type = "button"; saveBtn.className = "ai-sec-save"; saveBtn.textContent = "Save";
      head.appendChild(label); head.appendChild(saveBtn);
      const ta = document.createElement("textarea"); ta.className = "ai-sec-prompt"; ta.rows = 2;
      ta.placeholder = "Prompt for " + cls + "…"; ta.value = promptFor(cls);
      const res = document.createElement("div"); res.className = "ai-result ai-sec-result"; res.hidden = true;
      // Highlight Save when the box differs from the current baseline (your saved
      // prompt, or the shipped default if you haven't saved one).
      function refreshSaveState() {
        const dirty = ta.value.trim() !== promptBaseline(cls).trim() && ta.value.trim().length > 0;
        saveBtn.classList.toggle("dirty", dirty);
      }
      ta.addEventListener("input", () => { aiWorking[cls] = ta.value; refreshSaveState(); });
      saveBtn.addEventListener("click", () => {
        aiSaved[cls] = ta.value; aiWorking[cls] = ta.value; persistSaved();
        saveBtn.textContent = "Saved ✓"; saveBtn.classList.remove("dirty");
        setTimeout(() => { saveBtn.textContent = "Save"; refreshSaveState(); }, 1200);
      });
      sec.appendChild(head); sec.appendChild(ta); sec.appendChild(res);
      box.appendChild(sec);
      aiSectionResults[cls] = res;
      refreshSaveState();
    });
  }
  window.rebuildAiSections = rebuildAiSections;

  function startAiPick() {
    const active = currentPpeClasses().filter(c => promptFor(c).trim());
    if (!active.length) { showToast("Write a prompt for at least one selected class.", true); return; }
    if (personPickMode || manualDrawMode) closeActionModes();
    if (uiPlaying) togglePlayPause();   // stabilise the frame
    aiPickMode = true;
    document.getElementById("video_stream").classList.add("pick-mode");
    const btn = document.getElementById("ai_send_btn");
    btn.classList.add("picking"); btn.textContent = "Click a person… (Esc to cancel)";
  }
  window.startAiPick = startAiPick;

  function cancelAiPick() {
    aiPickMode = false;
    document.getElementById("video_stream").classList.remove("pick-mode");
    const btn = document.getElementById("ai_send_btn");
    btn.classList.remove("picking"); btn.textContent = "Send";
  }

  async function runAiAnalyze(xRatio, yRatio) {
    const classes = currentPpeClasses().filter(c => promptFor(c).trim());
    await Promise.all(classes.map(async cls => {
      const res = aiSectionResults[cls];
      if (!res) return;
      res.hidden = false; res.classList.remove("err"); res.classList.add("loading");
      res.textContent = "Analyzing…";
      try {
        const r = await fetch("/api/ai_analyze", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: promptFor(cls), x_ratio: xRatio, y_ratio: yRatio }),
        });
        const d = await r.json();
        res.classList.remove("loading");
        if (r.ok && d.ok) { res.textContent = d.result || "(empty response)"; }
        else { res.classList.add("err"); res.textContent = d.error || "Request failed."; }
      } catch (e) {
        res.classList.remove("loading"); res.classList.add("err"); res.textContent = "Network error.";
      }
    }));
  }

  // Fullscreen video with overlay transport controls.
  function toggleFullscreen() {
    const el = document.getElementById("video_wrap");
    const fsEl = document.fullscreenElement || document.webkitFullscreenElement;
    try {
      if (!fsEl) {
        (el.requestFullscreen || el.webkitRequestFullscreen).call(el);
      } else {
        (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      }
    } catch (e) {}
  }
  window.toggleFullscreen = toggleFullscreen;

  let fsIdleTimer = null;
  function fsActivity() {
    const el = document.getElementById("video_wrap");
    el.classList.remove("fs-idle");
    clearTimeout(fsIdleTimer);
    fsIdleTimer = setTimeout(() => {
      if (document.fullscreenElement) el.classList.add("fs-idle");
    }, 2500);
  }
  document.addEventListener("fullscreenchange", () => {
    const el = document.getElementById("video_wrap");
    const fs = !!document.fullscreenElement;
    const btn = document.getElementById("fs_btn");
    if (btn) { btn.textContent = fs ? "🗗" : "⛶"; btn.title = fs ? "Exit fullscreen" : "Fullscreen"; }
    if (fs) { fsActivity(); el.addEventListener("mousemove", fsActivity); }
    else { el.classList.remove("fs-idle"); clearTimeout(fsIdleTimer); el.removeEventListener("mousemove", fsActivity); }
  });
  document.getElementById("video_stream").addEventListener("dblclick", () => {
    if (personPickMode || manualDrawMode) return;   // don't hijack crop/draw modes
    toggleFullscreen();
  });

  // Source accordion: open one section, collapse the other. The open section is
  // the active source; a collapsed one is inert (its input is ignored server-side).
  function applySourceMode() {
    document.querySelectorAll("#source_accordion .src-sec").forEach(sec => {
      sec.classList.toggle("open", sec.dataset.mode === sourceMode);
    });
  }
  async function setSourceMode(mode, save = true) {
    if (mode !== "file" && mode !== "url") mode = "file";
    sourceMode = mode;
    applySourceMode();
    if (save) {
      await saveConfig(false);
      if (mode === "url") document.getElementById("video_url").focus();
    }
  }
  window.setSourceMode = setSourceMode;

  // One-click: make the URL the source, save it, and start streaming.
  async function playUrl() {
    const box = document.getElementById("video_url");
    const url = box.value.trim();
    if (!url) { showToast("Paste a video URL first.", true); box.focus(); return; }
    sourceMode = "url";
    applySourceMode();
    await saveConfig(false);          // selected_video becomes the URL; capture reloads
    uiPlaying = true;
    renderPlay(true);
    reflectPlayState(true);
    controlPendingUntil = Date.now() + 900;
    control("start");
    showToast("Downloading video…");
  }
  window.playUrl = playUrl;

  applySourceMode();

  setVideos(allVideos, selectedVideo);

  // Themed dropdowns for every native select.
  ["person_model_path", "ppe_model_path", "sm_model_path", "selected_video", "simulate_realtime"]
    .forEach(id => enhanceSelect(document.getElementById(id)));

  // Click anywhere on a card header (not just the tiny button) to collapse/expand.
  document.querySelectorAll(".card-head").forEach(h => {
    h.addEventListener("click", () => { const c = h.closest(".card"); if (c && c.id) toggleCard(c.id); });
  });

  const ids = ["person_model_path", "video_folder", "selected_video", "video_url", "person_conf", "ppe_conf", "sm_conf", "frame_step", "simulate_realtime", "ppe_inside_person"];
  ids.forEach((id) => {
    const el = document.getElementById(id);
    const eventName = (id.includes("conf") || id === "frame_step") ? "input" : "change";
    el.addEventListener(eventName, async () => {
      if (id === "video_folder") { await saveConfig(true); } else { await saveConfig(false); }
    });
  });

  document.getElementById("ppe_model_path").addEventListener("change", async (e) => {
    updateClassCardVisibility();
    await reloadPpeClasses(e.target.value);
    await saveConfig(false);
  });
  document.getElementById("sm_model_path").addEventListener("change", async (e) => {
    updateClassCardVisibility();
    await reloadSmClasses(e.target.value);
    await saveConfig(false);
  });

  // PPE Classes card wiring (2-column checkbox list).
  document.getElementById("ppe_cls_all").addEventListener("change", async (e) => {
    document.querySelectorAll("#ppe_classes_list .ppe-cls").forEach(cb => { cb.checked = e.target.checked; });
    updateAllToggle();
    rebuildAiSections();
    await saveConfig(false);
  });
  document.getElementById("ppe_classes_list").addEventListener("change", async (e) => {
    if (!e.target.classList.contains("ppe-cls")) return;
    updateAllToggle();
    rebuildAiSections();
    await saveConfig(false);
  });

  // SM Classes card wiring (same shape as the PPE one).
  document.getElementById("sm_cls_all").addEventListener("change", async (e) => {
    document.querySelectorAll("#sm_classes_list .sm-cls").forEach(cb => { cb.checked = e.target.checked; });
    updateSmAllToggle();
    await saveConfig(false);
  });
  document.getElementById("sm_classes_list").addEventListener("change", async (e) => {
    if (!e.target.classList.contains("sm-cls")) return;
    updateSmAllToggle();
    await saveConfig(false);
  });
  updateAllToggle();
  updateSmAllToggle();
  updateClassCardVisibility();
  rebuildAiSections();

  // Keep the played-fill glued to the slider thumb — from the value, so it moves
  // in lock-step with the dot while you drag (not just when the server catches up).
  function updateSeekFill() {
    const sl = document.getElementById("seek_slider");
    const max = Math.max(Number(sl.max) || 1, 1);
    const val = Math.min(Math.max(Number(sl.value) || 0, 0), max);
    document.getElementById("seek_played").style.width = ((val / max) * 100) + "%";
  }
  const seekSlider = document.getElementById("seek_slider");
  seekSlider.addEventListener("input", () => { suppressSliderUpdate = true; updateSeekFill(); });
  seekSlider.addEventListener("change", async () => { await seekToSlider(); updateSeekFill(); suppressSliderUpdate = false; });

  renderPlay(false);
  reflectPlayState(false);
  setInterval(refreshStatus, 300);
  refreshStatus();
</script>
</body>
</html>
"""
    replacements = {
        "__APP_TITLE__": html.escape(APP_TITLE),
        "__PERSON_OPTIONS__": person_model_options_html,
        "__PPE_OPTIONS__": ppe_model_options_html,
        "__SM_OPTIONS__": sm_model_options_html,
        "__PPE_CLASSES__": ppe_classes_html,
        "__SM_CLASSES__": sm_classes_html,
        "__PPE_INSIDE_CHECKED__": "checked" if ppe_inside_val else "",
        "__VIDEO_FOLDER__": html.escape(str(video_folder_val), quote=True),
        "__VIDEO_URL__": html.escape(str(video_url_val), quote=True),
        "__SOURCE_MODE__": html.escape(str(source_mode_val), quote=True),
        "__PERSON_CONF__": str(person_conf_val),
        "__PPE_CONF__": str(ppe_conf_val),
        "__SM_CONF__": str(sm_conf_val),
        "__FRAME_STEP__": str(frame_step_val),
        "__RT_ENABLED_SEL__": "selected" if simulate_realtime_val else "",
        "__RT_DISABLED_SEL__": "selected" if not simulate_realtime_val else "",
        "__AI_DEFAULTS__": json.dumps(PPE_DEFAULT_PROMPTS),
        "__VIDEOS_JSON__": json.dumps(videos),
        "__SELECTED_VIDEO_JSON__": json.dumps(selected_video_val),
        # Palette + shared chrome, straight from the suite design system.
        "__THEME__": suite_theme.stylesheet(),
        "__THEME_SCRIPT__": suite_theme.THEME_SCRIPT,
        "__THEME_JS__": suite_theme.THEME_JS,
        "__THEME_BUTTON__": suite_theme.THEME_BUTTON,
    }
    page = template
    for _k, _v in replacements.items():
        page = page.replace(_k, _v)
    return Response(page, mimetype="text/html")


@app.route("/home")
def _suite_home():
    """The "Suite" button links to /home. Mounted under run.py that URL reaches
    the suite shell, so this route only fires when this app runs standalone —
    where the nearest thing to a suite home is this app's own landing page."""
    return redirect("/")


@app.post("/api/config")
def api_config():
    payload = request.get_json(silent=True) or {}
    with state_lock:
        old_video = state["selected_video"]
        old_cfg_key = compute_cfg_key()
        for key in (
            "person_model_path",
            "ppe_model_path",
            "sm_model_path",
            "video_folder",
            "selected_video",
            "video_url",
            "source_mode",
            "person_conf",
            "ppe_conf",
            "sm_conf",
            "frame_step",
            "simulate_realtime",
            "ppe_inside_person",
            "ppe_classes",
            "sm_classes",
        ):
            if key in payload:
                state[key] = payload[key]

        # The active source is decided by source_mode; the inactive input is
        # ignored. In "url" mode the link is downloaded to a temp file and played
        # from disk — selected_video is that local path once the download is ready
        # (empty while it downloads). "file" mode uses the dropdown.
        state["source_mode"] = "url" if state.get("source_mode") == "url" else "file"
        state["video_url"] = str(state.get("video_url") or "").strip()
        if state["source_mode"] == "url":
            if state["video_url"]:
                start_url_download(state["video_url"])
                state["selected_video"] = url_download_path(state["video_url"]) or ""
            else:
                state["selected_video"] = ""

        # ppe_classes / sm_classes: None means "all"; else class-name strings.
        for _ck in ("ppe_classes", "sm_classes"):
            if state[_ck] is not None:
                if isinstance(state[_ck], list):
                    state[_ck] = [str(c) for c in state[_ck]]
                else:
                    state[_ck] = None

        state["person_conf"] = max(0.0, min(1.0, float(state["person_conf"])))
        state["ppe_conf"] = max(0.0, min(1.0, float(state["ppe_conf"])))
        state["sm_conf"] = max(0.0, min(1.0, float(state["sm_conf"])))
        state["frame_step"] = max(1, int(state["frame_step"]))

        all_pt = discover_pt_models([str(MODELS_DIR)])
        packages = discover_model_packages(MODELS_DIR)
        package_paths = [p for _, p in packages]
        if packages:
            allowed_pkg = set(package_paths)
            allowed_pkg.add(NONE_MODEL_VALUE)
            if state["ppe_model_path"] not in allowed_pkg:
                state["ppe_model_path"] = package_paths[0]
            if state["sm_model_path"] not in allowed_pkg:
                state["sm_model_path"] = package_paths[0]
        else:
            state["ppe_model_path"] = NONE_MODEL_VALUE
            state["sm_model_path"] = NONE_MODEL_VALUE
        if all_pt:
            allowed_pt = set(all_pt)
            allowed_pt.add(NONE_MODEL_VALUE)
            if state["person_model_path"] not in allowed_pt:
                state["person_model_path"] = all_pt[0]
        else:
            state["person_model_path"] = NONE_MODEL_VALUE

        videos = list_videos(str(state["video_folder"]))
        if state["source_mode"] == "file" and state["selected_video"] not in videos:
            state["selected_video"] = videos[0] if videos else ""
        video_changed = old_video != state["selected_video"]
        if video_changed:
            state["playing"] = False  # switching videos pauses playback
        cfg_changed = compute_cfg_key() != old_cfg_key
        cur_idx = runtime["frame_idx"]
        save_settings()
        selected_video = state["selected_video"]

    # Switching videos: drop the annotation cache so the worker reloads the new one.
    if video_changed:
        release_capture()
    elif cfg_changed and cur_idx >= 0:
        # Model/conf/step changed: re-annotate + refresh the current frame now.
        seek_to_frame(cur_idx)
    return jsonify({"ok": True, "videos": videos, "selected_video": selected_video})


@app.post("/api/control")
def api_control():
    payload = request.get_json(silent=True) or {}
    action = payload.get("action", "").lower()

    if action in ("start", "pause", "toggle"):
        with state_lock:
            if action == "start":
                want = True
            elif action == "pause":
                want = False
            else:
                want = not state["playing"]
            state["playing"] = want
        if want:
            opened = ensure_capture_open()
            if not opened:
                with state_lock:
                    state["playing"] = False
    elif action == "stop":
        with state_lock:
            state["playing"] = False
        release_capture()  # clears cache; worker re-anchors to the start
        with state_lock:
            runtime["frame_idx"] = -1
            runtime["last_jpg"] = None
            runtime["displayed_count"] = 0
            runtime["last_error"] = ""

    with state_lock:
        save_settings()
    return jsonify({"ok": True})


@app.post("/api/seek")
def api_seek():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "relative")).lower()

    with state_lock:
        has_video = bool(state["selected_video"])
        current_idx = runtime["frame_idx"]
        fps = float(runtime["fps"]) if runtime["fps"] > 0 else 25.0
    if not has_video:
        return jsonify({"ok": False, "error": "Choose a valid video."}), 400
    if current_idx < 0:
        current_idx = 0

    if mode == "absolute":
        target_idx = int(payload.get("frame", current_idx))
    else:
        delta = int(payload.get("delta", 0))
        unit = str(payload.get("unit", "frame")).lower()
        if unit == "sec":
            delta = int(round(delta * fps))
        target_idx = current_idx + delta

    ok = seek_to_frame(target_idx)
    with state_lock:
        err = runtime["last_error"]
        frame_idx = runtime["frame_idx"]
    if not ok:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, "frame_idx": frame_idx})


@app.post("/api/crop")
def api_crop():
    payload = request.get_json(silent=True) or {}
    crop_type = str(payload.get("type", "")).lower()
    reseek_idx = None  # manual_box re-renders the current frame AFTER releasing state_lock

    # Fetch the raw (unannotated) pixels for the displayed frame on demand.
    with state_lock:
        video = state["selected_video"]
        frame_idx = runtime["frame_idx"]
    if frame_idx < 0 or not video:
        return jsonify({"ok": False, "error": "No frame loaded yet."}), 400
    raw = read_raw_frame(video, frame_idx)
    if raw is None:
        return jsonify({"ok": False, "error": "Could not read the current frame."}), 400

    with state_lock:
        runtime["last_raw_frame"] = raw  # crop helpers read this
        try:
            if crop_type == "person":
                person_index = int(payload.get("person_index", 1))
                save_path = save_person_crop(person_index)
            elif crop_type == "person_point":
                x_ratio = float(payload.get("x_ratio", 0.0))
                y_ratio = float(payload.get("y_ratio", 0.0))
                save_path = save_person_crop_from_point(x_ratio, y_ratio)
            elif crop_type == "manual_box":
                x1_ratio = float(payload.get("x1_ratio", 0.0))
                y1_ratio = float(payload.get("y1_ratio", 0.0))
                x2_ratio = float(payload.get("x2_ratio", 0.0))
                y2_ratio = float(payload.get("y2_ratio", 0.0))
                class_name = str(payload.get("class_name") or "manual")
                folder = sanitize_class_name(class_name)
                box, frame_idx = add_manual_box_for_current_frame(
                    x1_ratio, y1_ratio, x2_ratio, y2_ratio, class_name=class_name
                )
                save_path = save_person_crop_from_box(box, suffix=folder, subdir=folder)
                reseek_idx = frame_idx
            elif crop_type == "frame":
                save_path = save_frame_image(runtime["last_raw_frame"], "SM")
            elif crop_type == "background":
                save_path = save_frame_image(runtime["last_raw_frame"], "background")
            else:
                return jsonify({"ok": False, "error": "Invalid crop type."}), 400
        except Exception as exc:
            runtime["last_action"] = f"Crop failed: {exc}"
            return jsonify({"ok": False, "error": str(exc)}), 400

        runtime["last_action"] = f"Saved {crop_type} crop: {save_path}"

    # Re-render the current frame so the new manual box shows immediately.
    # Done outside state_lock because seek_to_frame acquires its own locks.
    if reseek_idx is not None:
        seek_to_frame(reseek_idx)
    return jsonify({"ok": True, "path": save_path})


def _run_export(video_path: str, cfg_base: dict, manual_by_frame: dict):
    """Background worker: re-run inference over the whole video and write an mp4."""
    cap = cv2.VideoCapture(video_path)
    writer = None
    out_path = ""
    try:
        if not cap.isOpened():
            raise RuntimeError("Failed to open the selected video for export.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 1e-6 else 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        with export_lock:
            export_state["total"] = total

        out_dir = paths.EXPORTS_ROOT
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(video_path).stem
        out_path = str(out_dir / f"{stem}_inference.mp4")
        download_name = f"{stem}_inference.mp4"

        frame_idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            cfg = dict(cfg_base)
            cfg["manual_boxes"] = list(manual_by_frame.get(frame_idx, []))
            annotated, _boxes, error = annotate_frame(frame, cfg)
            if error:
                raise RuntimeError(error)
            if annotated is None:
                annotated = frame

            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("Could not create the output video writer.")
            writer.write(annotated)

            with export_lock:
                export_state["progress"] = frame_idx + 1

        if writer is None:
            raise RuntimeError("No frames were read from the video.")

        with export_lock:
            export_state["running"] = False
            export_state["done"] = True
            export_state["error"] = ""
            export_state["output_path"] = out_path
            export_state["download_name"] = download_name
    except Exception as exc:
        with export_lock:
            export_state["running"] = False
            export_state["done"] = False
            export_state["error"] = str(exc)
            export_state["output_path"] = ""
            export_state["download_name"] = ""
    finally:
        if writer is not None:
            writer.release()
        cap.release()


@app.post("/api/ai_analyze")
def api_ai_analyze():
    """Crop the clicked person on the current frame and ask the Azure vision model."""
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"ok": False, "error": "Enter a prompt first."}), 400
    try:
        x_ratio = float(payload.get("x_ratio", -1))
        y_ratio = float(payload.get("y_ratio", -1))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Invalid click position."}), 400

    with state_lock:
        video = state["selected_video"]
        frame_idx = runtime["frame_idx"]
    if frame_idx < 0 or not video:
        return jsonify({"ok": False, "error": "No frame loaded — pause on a frame first."}), 400
    raw = read_raw_frame(video, frame_idx)
    if raw is None:
        return jsonify({"ok": False, "error": "Could not read the current frame."}), 400
    with state_lock:
        runtime["last_raw_frame"] = raw  # get_person_crop_from_point reads this

    try:
        crop = get_person_crop_from_point(x_ratio, y_ratio)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    try:
        result = call_foundry(crop, prompt)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "ignore")[:400]
        except Exception:
            pass
        return jsonify({"ok": False, "error": f"Model error ({exc.code}). {detail}"}), 502
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Request failed: {exc}"}), 502

    return jsonify({"ok": True, "result": result or "(empty response)"})


@app.post("/api/export")
def api_export():
    with state_lock:
        video_path = state["selected_video"]
        cfg_base = snapshot_inference_cfg(-1)
        cfg_base.pop("manual_boxes", None)
        manual_by_frame = {
            idx: list(boxes)
            for idx, boxes in runtime["manual_boxes_by_frame"].items()
        }

    if not video_path or not Path(video_path).is_file():
        return jsonify({"ok": False, "error": "Choose a valid video first."}), 400

    with export_lock:
        if export_state["running"]:
            return jsonify({"ok": False, "error": "An export is already running."}), 409
        export_state.update(
            {
                "running": True,
                "done": False,
                "error": "",
                "progress": 0,
                "total": 0,
                "output_path": "",
                "download_name": "",
            }
        )

    threading.Thread(
        target=_run_export,
        args=(video_path, cfg_base, manual_by_frame),
        daemon=True,
    ).start()
    return jsonify({"ok": True})


@app.get("/api/export/status")
def api_export_status():
    with export_lock:
        return jsonify(dict(export_state))


@app.get("/download/video")
def download_video():
    with export_lock:
        out_path = export_state["output_path"]
        download_name = export_state["download_name"] or "inference.mp4"
    if not out_path or not Path(out_path).is_file():
        return jsonify({"ok": False, "error": "No exported video available."}), 404
    return send_file(
        out_path,
        mimetype="video/mp4",
        as_attachment=True,
        download_name=download_name,
    )


@app.get("/api/options")
def api_options():
    """JSON model/video options + current state, for the React control panel."""
    with state_lock:
        person_models = [
            {"name": Path(p).name, "path": p}
            for p in discover_pt_models([str(MODELS_DIR)])
        ]
        packages = [
            {"name": folder, "path": pt}
            for folder, pt in discover_model_packages(MODELS_DIR)
        ]
        videos = [
            {"name": Path(v).name, "path": v}
            for v in list_videos(str(state["video_folder"]))
        ]
        return jsonify(
            {
                "none_value": NONE_MODEL_VALUE,
                "person_models": person_models,
                "packages": packages,
                "videos": videos,
                "models_dir": str(MODELS_DIR),
                "state": {
                    "person_model_path": state["person_model_path"],
                    "ppe_model_path": state["ppe_model_path"],
                    "sm_model_path": state["sm_model_path"],
                    "video_folder": state["video_folder"],
                    "selected_video": state["selected_video"],
                    "video_url": state["video_url"],
                    "source_mode": state["source_mode"],
                    "person_conf": state["person_conf"],
                    "ppe_conf": state["ppe_conf"],
                    "sm_conf": state["sm_conf"],
                    "frame_step": state["frame_step"],
                    "simulate_realtime": state["simulate_realtime"],
                    "ppe_inside_person": state["ppe_inside_person"],
                    "ppe_classes": state["ppe_classes"],
                    "sm_classes": state["sm_classes"],
                    "playing": state["playing"],
                },
            }
        )


@app.get("/api/ppe_classes")
def api_ppe_classes():
    """Class names for a given PPE model path, to populate the class picker."""
    path = request.args.get("path", "")
    return jsonify({"classes": get_ppe_class_names(path)})


@app.get("/api/status")
def api_status():
    with state_lock:
        step = max(int(state["frame_step"]), 1)
        mode = state["source_mode"]
        url = state["video_url"]
        sv = state["selected_video"] or ""
    dl = None
    if mode == "url" and url:
        host = urlparse(url).netloc or "url"
        with download_lock:
            dstatus = download_state["status"] if download_state["url"] == url else "idle"
            dprog = download_state["progress"]
            derror = download_state["error"]
        if dstatus == "downloading":
            source_label = f"downloading · {host}  {int(dprog * 100)}%"
        elif dstatus == "ready":
            source_label = "url · " + host
        elif dstatus == "error":
            source_label = "url failed · " + host
        else:
            source_label = "url · " + host
        is_url = True
        dl = {"status": dstatus, "progress": dprog, "error": derror}
    elif sv:
        source_label = Path(sv).name
        is_url = False
    else:
        source_label = "no source"
        is_url = False
    with state_lock:
        payload = {
            "playing": state["playing"],
            "frame_idx": runtime["frame_idx"],
            "total_frames": runtime["total_frames"],
            "fps": float(runtime["fps"]) if runtime["fps"] > 0 else 25.0,
            "displayed_count": runtime["displayed_count"],
            "last_error": runtime["last_error"],
            "last_action": runtime["last_action"],
            "available_person_count": len(runtime["last_person_boxes"]),
            "frame_step": step,
            "source_label": source_label,
            "source_is_url": is_url,
            "download": dl,
        }
    # Cache coverage as contiguous [start, end] ranges (separate lock, not nested).
    with cache_lock:
        ranges = cached_ranges(step)
        loaded_count = len(frame_cache)
    payload["loaded_ranges"] = ranges
    payload["loaded_count"] = loaded_count
    return jsonify(payload)


@app.get("/stream.mjpg")
def stream_mjpg():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _open_browser_when_ready(url: str, delay_sec: float = 0.8) -> None:
    # The npm launcher sets NO_BROWSER so it can open a single landing page
    # instead of every app popping its own tab.
    if os.environ.get("NO_BROWSER"):
        return

    def _open() -> None:
        time.sleep(delay_sec)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


if __name__ == "__main__":
    load_settings()
    configure_quiet_logging()
    # Background producer that decodes + annotates frames ahead of playback.
    threading.Thread(target=prerender_worker, daemon=True).start()
    # 8500 is only a preference — move up if something already holds it.
    port = ports.resolve(int(os.environ.get("PORT", "8500")))
    print(f"Inference Web App on http://127.0.0.1:{port}")
    _open_browser_when_ready(f"http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
