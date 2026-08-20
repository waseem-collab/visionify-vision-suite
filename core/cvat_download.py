#!/usr/bin/env python3
"""
Package one CVAT task as a ready-to-download training ZIP.

The CVAT Download card: pick a project and task, and this builds a zip shaped
like the team's dataset folders (e.g. Downloads/single_model_fall-os_task_40):

    <task_name>/
      data.yaml               nc + names (from the export's obj.names)
      train/
        images/<frame>.jpg    the task's frames
        labels/<frame>.txt    YOLO labels
      review/                 empty review-tool state, ready for the reviewer
        reviewed.json  metadata.json  tags/

Exports "YOLO 1.1" WITH images via the same authenticated REST flow as
cvat_sync, restructures it, zips it under paths.CVAT_DL_ROOT and hands the
path to /api/cvatdl/file for the browser download. One job at a time,
polled status like the other background jobs.
"""
import json
import os
import re
import shutil
import tempfile
import threading
import zipfile

from core import cvat_sync, paths

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_lock = threading.Lock()
_state = {
    "running": False, "done": False, "error": "",
    "task": "", "images": 0, "labels": 0, "classes": 0,
    "zip": "", "zip_name": "", "size_mb": 0.0, "message": "",
}


def status():
    with _lock:
        return dict(_state)


def _set(**kw):
    with _lock:
        _state.update(kw)


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()) or "task"


def _read_names(extract_dir):
    """Class names from the export's obj.names (index = class id)."""
    p = os.path.join(extract_dir, "obj.names")
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh.read().splitlines() if ln.strip()]


def _run(task_id, task_name):
    safe = _safe_name(task_name)
    tmp = tempfile.mkdtemp(prefix="cvat_dl_")
    try:
        _set(message="Exporting from CVAT (with images)…")
        session, base_url = cvat_sync._session()
        raw_zip = os.path.join(tmp, "export.zip")
        cvat_sync._export_labels(session, base_url, int(task_id), raw_zip,
                                 save_images=True)

        _set(message="Unpacking export…")
        extract = os.path.join(tmp, "x")
        with zipfile.ZipFile(raw_zip) as zf:
            zf.extractall(extract)
        os.remove(raw_zip)

        # Restructure into the team's dataset layout.
        _set(message="Restructuring into train/images + train/labels…")
        root = os.path.join(tmp, safe)
        img_dir = os.path.join(root, "train", "images")
        lbl_dir = os.path.join(root, "train", "labels")
        os.makedirs(img_dir)
        os.makedirs(lbl_dir)
        n_img = n_lbl = 0
        for dirpath, _dirs, files in os.walk(extract):
            base = os.path.basename(dirpath)
            if not (base.startswith("obj_") and base.endswith("_data")):
                continue
            for f in files:
                src = os.path.join(dirpath, f)
                if f.lower().endswith(".txt"):
                    shutil.move(src, os.path.join(lbl_dir, f))
                    n_lbl += 1
                elif f.lower().endswith(_IMG_EXTS):
                    shutil.move(src, os.path.join(img_dir, f))
                    n_img += 1

        names = _read_names(extract)
        with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
            fh.write("train: train/images\nval: train/images\ntest: train/images\n\n")
            fh.write(f"nc: {len(names)}\n")
            fh.write("names: [" + ", ".join(json.dumps(n) for n in names) + "]\n")

        # Review-tool state: everything from CVAT is already annotated, so every
        # image arrives marked reviewed ("train/<stem>"), like the reference
        # datasets. Tags start empty.
        review = os.path.join(root, "review")
        os.makedirs(os.path.join(review, "tags"))
        reviewed = sorted(f"train/{os.path.splitext(f)[0]}" for f in os.listdir(img_dir))
        with open(os.path.join(review, "reviewed.json"), "w", encoding="utf-8") as fh:
            json.dump({"reviewed": reviewed}, fh, indent=2)
        with open(os.path.join(review, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({}, fh)

        _set(images=n_img, labels=n_lbl, classes=len(names),
             message=f"Zipping {n_img} image(s) + {n_lbl} label file(s)…")
        paths.CVAT_DL_ROOT.mkdir(parents=True, exist_ok=True)
        final = paths.CVAT_DL_ROOT / f"{safe}.zip"
        part = paths.CVAT_DL_ROOT / f"{safe}.zip.part"
        # Write to .part and rename only when complete — a job killed mid-write
        # (server reload, crash) can never leave a truncated .zip behind.
        with zipfile.ZipFile(part, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _dirs, files in os.walk(root):
                for f in files:
                    full = os.path.join(dirpath, f)
                    zf.write(full, os.path.relpath(full, tmp))  # keeps <task>/ root
            # empty dirs (tags/) need explicit entries
            zf.writestr(f"{safe}/review/tags/.keep", "")
        os.replace(part, final)

        size_mb = round(final.stat().st_size / 1e6, 1)
        _set(running=False, done=True, zip=str(final), zip_name=f"{safe}.zip",
             size_mb=size_mb,
             message=f"Ready: {safe}.zip — {n_img} images, {n_lbl} labels, "
                     f"{len(names)} classes ({size_mb} MB)")
    except Exception as exc:
        _set(running=False, done=True, error=str(exc), message=f"Failed: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def start(task_id, task_name):
    """Kick off packaging. Returns an error string, or None if started."""
    if not task_id:
        return "No task selected."
    # Sweep leftovers from any previously interrupted job.
    try:
        for p in paths.CVAT_DL_ROOT.glob("*.zip.part"):
            p.unlink()
    except OSError:
        pass
    with _lock:
        if _state["running"]:
            return "A download is already being prepared."
        _state.update({"running": True, "done": False, "error": "",
                       "task": str(task_name), "images": 0, "labels": 0,
                       "classes": 0, "zip": "", "zip_name": "", "size_mb": 0.0,
                       "message": "Starting…"})
    threading.Thread(target=_run, args=(task_id, task_name), daemon=True).start()
    return None
