#!/usr/bin/env python3
"""
Best-effort logging of crops and annotations to Convex.

Every tool calls ``record_crop`` / ``record_annotation`` right after it writes a
file. Those calls are cheap and NON-BLOCKING: the metadata is dropped on a queue
and a single background thread posts it to Convex. The design rules:

- **Never break a tool.** If Convex is down, unconfigured, slow, or erroring, the
  crop/annotation is still saved to disk; we just lose (or drop) the DB row. No
  exception ever propagates back to the caller.
- **Off the request path.** Handlers return immediately; the network call to
  Convex happens on the worker thread, so playback/annotation never waits on it.
- **Metadata only.** Filenames, frames, YOLO coordinates, labels — no image
  bytes ever leave the machine.

Enabled only when both CONVEX_URL and CONVEX_SHARED_SECRET are set; otherwise
every call here is a no-op with zero overhead.
"""
import atexit
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import convex_client

# Bounded so a Convex outage can never grow memory without limit — old events are
# dropped once it fills (best-effort, not a durable log).
_QUEUE_MAX = 5000
_q = None
_worker = None
_lock = threading.Lock()
_dropped = 0
_sent = 0
_failed = 0


def enabled():
    """True when there is somewhere to log to and a secret to authorise it."""
    return bool(convex_client.convex_url() and convex_client.shared_secret())


def _ensure_worker():
    """Start the single poster thread on first use (only if logging is enabled)."""
    global _q, _worker
    if _worker is not None:
        return True
    if not enabled():
        return False
    with _lock:
        if _worker is None:
            _q = queue.Queue(maxsize=_QUEUE_MAX)
            _worker = threading.Thread(target=_run, name="convex-logger", daemon=True)
            _worker.start()
    return True


def _run():
    global _sent, _failed
    from core import cameras
    client = convex_client.get_client()
    if client is None:
        return
    while True:
        fn, args = _q.get()
        try:
            # Tag crops with their camera here (off the request path) so the
            # registry lookup never slows down a crop save.
            if fn == "crops:record" and not args.get("camera"):
                company, site, camera = cameras.resolve(args.get("video", ""))
                if camera:
                    args["company"], args["site"], args["camera"] = company, site, camera
            client.mutation(fn, args)
            _sent += 1
        except Exception:
            # Swallow everything — a logging failure must never surface. Count it
            # so /api/convex/status can show something's wrong without noise.
            _failed += 1
        finally:
            _q.task_done()


def _enqueue(fn, args):
    global _dropped
    if not _ensure_worker():
        return
    try:
        _q.put_nowait((fn, args))
    except queue.Full:
        _dropped += 1  # drop oldest-behaviour: just drop this one; disk still has it


def stats():
    return {"enabled": enabled(), "queued": _q.qsize() if _q else 0,
            "sent": _sent, "failed": _failed, "dropped": _dropped}


def record_crop(filename, video, frame, cx, cy, w, h, source, conf=None):
    """Log one saved crop. Coordinates are YOLO (normalised to the frame)."""
    if not enabled():
        return
    args = {
        "secret": convex_client.shared_secret(),
        "filename": os.path.basename(str(filename)),
        "video": str(video),
        "frame": int(frame),
        "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h),
        "source": str(source),
        "savedAt": int(time.time() * 1000),
    }
    if conf is not None:
        args["conf"] = float(conf)
    _enqueue("crops:record", args)


def record_annotation(image, boxes, crop=None, video=None, frame=None):
    """Log one saved annotation. ``boxes`` is a list of dicts with YOLO
    cls/cx/cy/w/h (and optional className) — the app's own box format."""
    if not enabled():
        return
    clean_boxes = []
    for b in boxes:
        try:
            row = {
                "cls": int(b.get("cls", 0)),
                "cx": float(b["cx"]), "cy": float(b["cy"]),
                "w": float(b["w"]), "h": float(b["h"]),
            }
        except (KeyError, TypeError, ValueError):
            continue
        name = b.get("className") or b.get("name")
        if name:
            row["className"] = str(name)
        clean_boxes.append(row)
    args = {
        "secret": convex_client.shared_secret(),
        "image": os.path.basename(str(image)),
        "boxes": clean_boxes,
        "savedAt": int(time.time() * 1000),
    }
    if crop:
        args["crop"] = os.path.basename(str(crop))
    if video is not None:
        args["video"] = str(video)
    if frame is not None:
        args["frame"] = int(frame)
    _enqueue("annotations:record", args)


@atexit.register
def _drain(timeout=2.0):
    """On shutdown, give the queue a moment to flush — best-effort, never blocks
    for long."""
    if _q is None:
        return
    deadline = time.time() + timeout
    while not _q.empty() and time.time() < deadline:
        time.sleep(0.05)
