#!/usr/bin/env python3
"""
Bulk-import a folder of YOLO label files into the database.

Point it at a dataset root (or a bare ``labels/`` folder) and it reads every
``<stem>.txt``, turns each into an annotation row, and — when the file is named
with the suite's crop scheme (``<video>_<frame>_<cx-cy-w-h>``) — also derives the
crop's frame position and camera so the point shows up on the heatmap. Class ids
are mapped to names via the folder's ``labels.txt`` / ``classes.txt``.

Runs on a background thread with a polled status, so a large folder doesn't block
the request. Best-effort per file: a malformed line is skipped, not fatal.
"""
import os
import threading
import time

import cameras
import convex_client
import cropnames

_BATCH = 100  # rows per bulk mutation — keeps each Convex transaction small

_lock = threading.Lock()
_state = {
    "running": False, "done": False, "error": "",
    "total": 0, "processed": 0, "crops": 0, "annotations": 0,
    "path": "", "message": "",
}


def status():
    with _lock:
        return dict(_state)


def _set(**kw):
    with _lock:
        _state.update(kw)


def _find_labels_dir(path):
    """Resolve the folder that actually holds the .txt files."""
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return None, None
    sub = os.path.join(path, "labels")
    if os.path.isdir(sub):
        return sub, path           # dataset root given
    return path, os.path.dirname(path)  # a bare labels/ folder given


def _read_class_names(labels_dir, dataset_dir):
    """Class names (index = class id) from labels.txt / classes.txt, searched in
    the labels folder, the dataset root, then its parent."""
    seen = []
    for d in (labels_dir, dataset_dir, os.path.dirname(labels_dir)):
        if not d:
            continue
        for fname in ("labels.txt", "classes.txt"):
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                try:
                    with open(p, encoding="utf-8") as fh:
                        names = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
                    if names:
                        return names
                except OSError:
                    pass
    return seen


def _parse_label_file(path):
    """YOLO rows → list of (cls, cx, cy, w, h). Skips malformed lines."""
    boxes = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls = int(float(parts[0]))
                    cx, cy, w, h = (float(x) for x in parts[1:5])
                except ValueError:
                    continue
                boxes.append((cls, cx, cy, w, h))
    except OSError:
        pass
    return boxes


def _flush(client, secret, crop_batch, ann_batch):
    """Send one batch of crops + annotations; return (crops, anns) counts sent."""
    c = a = 0
    if crop_batch:
        client.mutation("crops:bulkRecord", {"secret": secret, "items": crop_batch})
        c = len(crop_batch)
    if ann_batch:
        client.mutation("annotations:bulkRecord", {"secret": secret, "items": ann_batch})
        a = len(ann_batch)
    return c, a


def _run(path):
    client = convex_client.get_client()
    secret = convex_client.shared_secret()
    if client is None or not secret:
        _set(running=False, done=True, error="Convex is not configured (CONVEX_URL / CONVEX_SHARED_SECRET).")
        return

    labels_dir, dataset_dir = _find_labels_dir(path)
    if not labels_dir:
        _set(running=False, done=True, error=f"Not a folder: {path}")
        return

    files = sorted(f for f in os.listdir(labels_dir) if f.lower().endswith(".txt")
                   and f.lower() not in ("labels.txt", "classes.txt"))
    names = _read_class_names(labels_dir, dataset_dir)
    _set(total=len(files), message=(f"{len(files)} label files"
         + (f", {len(names)} classes" if names else ", no class names found")))

    crop_batch, ann_batch = [], []
    sent_c = sent_a = 0
    now = int(time.time() * 1000)

    for i, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]
        boxes_raw = _parse_label_file(os.path.join(labels_dir, fname))
        # Build the annotation boxes (with class names when we have them).
        boxes = []
        for cls, cx, cy, w, h in boxes_raw:
            b = {"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h}
            if 0 <= cls < len(names):
                b["className"] = names[cls]
            boxes.append(b)

        image = stem + ".jpg"  # crops are saved as .jpg; this is the join key
        ann = {"image": image, "boxes": boxes, "savedAt": now}

        # If the name is one of our crops, recover the crop's frame position +
        # camera so the heatmap has a point to plot for these classes.
        link = cropnames.parse_crop_name(image)
        if link:
            ann["crop"] = image
            ann["video"] = link["video"]
            ann["frame"] = link["frame"]
            company, site, camera = cameras.resolve(link["video"])
            crop = {
                "filename": image, "video": link["video"], "frame": link["frame"],
                "cx": link["cx"], "cy": link["cy"], "w": link["w"], "h": link["h"],
                "source": "label_import", "savedAt": now,
            }
            if camera:
                crop["company"], crop["site"], crop["camera"] = company, site, camera
            crop_batch.append(crop)

        ann_batch.append(ann)

        if len(ann_batch) >= _BATCH or len(crop_batch) >= _BATCH:
            try:
                c, a = _flush(client, secret, crop_batch, ann_batch)
                sent_c += c; sent_a += a
            except Exception as exc:
                _set(running=False, done=True, error=f"Import failed at {fname}: {exc}",
                     processed=i, crops=sent_c, annotations=sent_a)
                return
            crop_batch, ann_batch = [], []
        _set(processed=i + 1, crops=sent_c + len(crop_batch), annotations=sent_a + len(ann_batch))

    try:
        c, a = _flush(client, secret, crop_batch, ann_batch)
        sent_c += c; sent_a += a
    except Exception as exc:
        _set(running=False, done=True, error=f"Import failed on final batch: {exc}",
             crops=sent_c, annotations=sent_a)
        return

    cameras.refresh()  # any freshly-tagged cameras show up in filters
    _set(running=False, done=True, processed=len(files), crops=sent_c, annotations=sent_a,
         message=f"Imported {sent_a} annotations ({sent_c} with crop positions) from {len(files)} files.")


def start(path):
    """Kick off an import. Returns an error string, or None if it started."""
    with _lock:
        if _state["running"]:
            return "An import is already running."
        _state.update({"running": True, "done": False, "error": "", "total": 0,
                       "processed": 0, "crops": 0, "annotations": 0,
                       "path": str(path), "message": "Starting…"})
    threading.Thread(target=_run, args=(path,), daemon=True).start()
    return None
