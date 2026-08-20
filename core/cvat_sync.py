#!/usr/bin/env python3
"""
Pull annotations for CVAT tasks into the database, on demand.

The heatmap's project/task filters list what exists in CVAT; selecting one
calls ``start(project, tasks)`` here, which — per task — exports a labels-only
"YOLO 1.1" dataset from CVAT (no images), unpacks it, and feeds it through the
label importer so every annotation (and, for crop-scheme filenames, its heatmap
position) lands in Convex stamped with the CVAT project/task names.

Runs on a background thread with a polled status, mirroring label_import.
Best-effort per task: one task failing doesn't stop the rest.
"""
import os
import shutil
import tempfile
import threading
import time
import zipfile

from core import convex_client, label_import

_lock = threading.Lock()
_state = {
    "running": False, "done": False, "error": "",
    "total": 0, "processed": 0, "annotations": 0, "crops": 0,
    "project": "", "current": "", "message": "", "failed": [],
}


def status():
    with _lock:
        return dict(_state)


def _set(**kw):
    with _lock:
        _state.update(kw)


def _env(key):
    convex_client.convex_url()  # side-effect: loads .env into os.environ
    return (os.environ.get(key) or "").strip()


def _session():
    """Authenticated requests session (org-scoped) — same REST approach as the
    annotation studio; more reliable across CVAT versions than the SDK export."""
    import requests
    url = _env("CVAT_URL") or _env("CVAT_HOST")
    user, pw = _env("CVAT_USERNAME"), _env("CVAT_PASSWORD")
    org = _env("CVAT_ORG_SLUG") or "visionify"
    if not (url and user and pw):
        raise RuntimeError("CVAT_URL / CVAT_USERNAME / CVAT_PASSWORD missing in .env")
    url = url.rstrip("/")
    s = requests.Session()
    if org:
        s.headers.update({"X-Organization": org})
    s.headers.update({"Referer": url})
    r = s.post(f"{url}/api/auth/login",
               json={"username": user, "password": pw},
               headers={"Content-Type": "application/json"})
    if r.status_code not in (200, 201):
        raise RuntimeError(f"CVAT login failed: {r.status_code}")
    csrf = s.cookies.get("csrftoken")
    if csrf:
        s.headers.update({"X-CSRFToken": csrf})
    return s, url


def _export_labels(session, base_url, task_id, zip_path, save_images=False):
    """Export a task as 'YOLO 1.1' and download the zip. By default labels
    only; ``save_images=True`` bundles the frames too (the download card)."""
    r = session.post(f"{base_url}/api/tasks/{task_id}/dataset/export",
                     params={"format": "YOLO 1.1",
                             "save_images": "true" if save_images else "false",
                             "location": "local"})
    if r.status_code not in (200, 201, 202):
        raise RuntimeError(f"export init failed {r.status_code}: {r.text[:160]}")
    rq_id = r.json().get("rq_id")
    if not rq_id:
        raise RuntimeError(f"no rq_id from export: {r.text[:160]}")
    waited = 0
    while waited < 1800:   # image exports can be large
        st = session.get(f"{base_url}/api/requests/{rq_id}").json()
        s = (st.get("status") or "").lower()
        if s == "finished":
            url = st.get("result_url")
            if not url:
                raise RuntimeError("export finished but no result_url")
            if not url.startswith("http"):
                url = f"{base_url}{url}"
            resp = session.get(url, stream=True)
            if resp.status_code != 200:
                raise RuntimeError(f"download failed {resp.status_code}")
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
            return
        if s == "failed":
            raise RuntimeError(f"export failed: {st.get('message')}")
        time.sleep(2)
        waited += 2
    raise RuntimeError("export timed out")


def _labels_dir(extract_dir):
    """The folder holding the .txt files inside a YOLO 1.1 export
    (obj_train_data / obj_<subset>_data), or the root as a fallback."""
    for name in sorted(os.listdir(extract_dir)):
        full = os.path.join(extract_dir, name)
        if os.path.isdir(full) and name.startswith("obj_") and name.endswith("_data"):
            return full
    return extract_dir


def _keep_list(labels_dir):
    """Image filenames present in this export (the importer's join key is
    ``<stem>.jpg``) — everything else stamped with the task is stale."""
    return [os.path.splitext(f)[0] + ".jpg" for f in os.listdir(labels_dir)
            if f.lower().endswith(".txt") and f.lower() not in ("labels.txt", "classes.txt")]


def _prune_task(project, task_name, keep):
    """Mirror-delete rows this task no longer contains. Returns rows removed."""
    if len(keep) > 8000:   # stay inside Convex's array-argument limit
        return 0
    client = convex_client.get_client()
    if client is None:
        return 0
    res = client.mutation("crops:pruneCvatTask", {
        "secret": convex_client.shared_secret(),
        "project": project, "task": task_name, "keep": keep,
    })
    return int(res.get("crops", 0)) + int(res.get("annotations", 0))


def _import_folder(path, project, task_name):
    """Run one folder through the label importer and wait for it, returning
    (annotations, crops, error)."""
    while label_import.status()["running"]:
        time.sleep(0.2)
    err = label_import.start(path, project, task_name)
    if err:
        return 0, 0, err
    while True:
        st = label_import.status()
        if st["done"] or not st["running"]:
            return st["annotations"], st["crops"], st["error"]
        time.sleep(0.2)


def _run(project, tasks):
    try:
        session, base_url = _session()
    except Exception as exc:
        _set(running=False, done=True, error=str(exc))
        return
    total_ann = total_crops = total_pruned = 0
    failed = []
    for i, t in enumerate(tasks):
        name = str(t.get("name") or t.get("id"))
        _set(processed=i, current=name,
             message=f"({i + 1}/{len(tasks)}) pulling annotations for {name}…")
        tmp = tempfile.mkdtemp(prefix="cvat_sync_")
        zip_path = os.path.join(tmp, "labels.zip")
        try:
            _export_labels(session, base_url, int(t["id"]), zip_path)
            extract = os.path.join(tmp, "x")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract)
            labels = _labels_dir(extract)
            ann, crops, err = _import_folder(labels, project, name)
            if err:
                failed.append(f"{name}: {err}")
            else:
                # A sync is a mirror: rows this task no longer contains go away.
                try:
                    total_pruned += _prune_task(project, name, _keep_list(labels))
                except Exception:
                    pass  # best-effort — never fail the sync over pruning
            total_ann += ann
            total_crops += crops
        except Exception as exc:
            failed.append(f"{name}: {exc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    msg = f"Synced {total_ann} annotations ({total_crops} heatmap points) from {len(tasks)} task(s)."
    if total_pruned:
        msg += f" Removed {total_pruned} stale row(s)."
    if failed:
        msg += f" {len(failed)} failed."
    _set(running=False, done=True, processed=len(tasks), annotations=total_ann,
         crops=total_crops, failed=failed, current="", message=msg)


def start(project, tasks):
    """Sync the given CVAT tasks ([{id, name}]) under ``project``.
    Returns an error string, or None if the job started."""
    tasks = [t for t in (tasks or []) if t.get("id")]
    if not tasks:
        return "No tasks to sync."
    with _lock:
        if _state["running"]:
            return "A CVAT sync is already running."
        _state.update({"running": True, "done": False, "error": "", "failed": [],
                       "total": len(tasks), "processed": 0, "annotations": 0,
                       "crops": 0, "project": str(project), "current": "",
                       "message": "Starting…"})
    threading.Thread(target=_run, args=(str(project), tasks), daemon=True).start()
    return None
