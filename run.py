#!/usr/bin/env python3
"""
Vision Suite — one server, three workspaces.

    /             landing page
    /annotate/    Annotation Studio          (apps/annotation_app.py)
    /webapp/      Inference Web App PPE+SM   (apps/web_app.py)
    /crop/        Crop Tools                 (apps/crop_balancer_app.py)

Each tool is a mature Flask app in its own right — thousands of lines, its own
routes, threads and background jobs. Rather than rewrite them into one app, we
compose them with Werkzeug's application dispatcher (the standard Flask
"Application Dispatching" pattern): every app keeps its own state and routes,
but now lives under a URL prefix on a single port, sharing one model pool and
one .env.

The one wrinkle is that all three emit *root-absolute* URLs in their HTML/JS
(fetch("/api/status"), <img src="/stream.mjpg">, href="/balancer"). Those would
resolve against the site root and miss the mounted sub-app, so a narrow
after_request pass rewrites the known prefixes to include the mount point.
request.script_root supplies that mount point, so one hook serves every app.

Run:  python3 run.py        (or: npm run dev)
"""
import atexit
import fcntl
import html
import os
import sys
import threading
import time
from pathlib import Path

from flask import Flask, Response, redirect, request
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from core import auth, convex_client, paths, ports, theme

paths.ensure_dirs()
# Load .env into os.environ NOW: auth reads AUTH_SECRET from the environment,
# and leaving it to convex_client's lazy loader means a freshly (re)started
# server can 401 valid session cookies until some Convex call happens first.
convex_client.convex_url()

# Importing the tools pulls in torch/ultralytics/opencv, so it takes a moment.
from apps import annotation_app, crop_balancer_app, web_app

LANDING_TEMPLATE = paths.ROOT / "templates" / "landing.html"
HOST = os.environ.get("HOST", "0.0.0.0")
# PORT is a *preference*, not a promise: if it is taken (a stale server, another
# tool, a second copy of the suite) we serve on the next port that is free.
# SUITE_PORT carries the resolved choice to the reloader's child processes so the
# URL stays put across restarts. We announce a moved port from the banner below
# (once we know we're actually serving), not here — so a launch that ends up
# refused by the single-instance guard doesn't first claim a port it won't use.
_PREFERRED_PORT = int(os.environ.get("PORT", "8000"))
PORT = ports.resolve(_PREFERRED_PORT, HOST, env_var="SUITE_PORT", announce=False)

# A token unique to this process. When the dev launcher restarts the server on a
# file change, the token changes and the injected live-reload script refreshes
# every open page automatically — so edits show up without a manual reload.
START_TOKEN = str(time.time())
LIVERELOAD_SNIPPET = (
    "<script>(function(){var t=null;function c(){"
    "fetch('/__livereload__',{cache:'no-store'}).then(function(r){return r.text();})"
    ".then(function(id){if(t===null){t=id;}else if(id!==t){location.reload();}})"
    ".catch(function(){});}setInterval(c,1000);c();})();</script>"
)


def _with_livereload(body):
    """Insert the live-reload poller just before </body> (or append it)."""
    if "</body>" in body:
        return body.replace("</body>", LIVERELOAD_SNIPPET + "</body>", 1)
    return body + LIVERELOAD_SNIPPET


# Quoted absolute-URL prefixes the sub-apps emit in their HTML/JS. Each entry is
# a string that, when found in an HTML response, gets the app's mount prefix
# injected right after the opening quote. Listing them explicitly (rather than
# blindly rewriting every "/…") keeps us from touching unrelated strings — the
# crop tools' "/path/to/video.mp4" placeholder, say. The union is safe to apply
# to all three: a prefix an app never emits simply never matches.
URL_LITERAL_PREFIXES = (
    '"/api/', "'/api/",                  # every fetch()/img.src API call, all three apps
    '"/stream.mjpg', "'/stream.mjpg",    # web_app MJPEG <img> stream
    '"/download/', "'/download/",        # web_app "download video"
    '"/balancer', "'/balancer",          # crop tools nav links
    '"/disagreement', "'/disagreement",
    '"/review', "'/review",
    '"/demo', "'/demo",                  # annotation studio's class-wheel gallery
    'href="/"', "href='/'",
)


def _rewrite_absolute_urls(app):
    """Register an after_request hook that prefixes the sub-app's own absolute
    URLs with its mount point, so links/fetches resolve to the mounted app."""

    @app.after_request
    def _prefix_urls(resp):
        root = request.script_root  # e.g. "/crop" or "/webapp"
        # Only HTML pages carry these link/fetch literals. Skip JSON (real file
        # paths live there), images, and the multipart MJPEG stream — rewriting
        # or buffering those would corrupt them or hang the stream.
        if not root or resp.mimetype != "text/html" or resp.direct_passthrough:
            return resp
        body = resp.get_data(as_text=True)
        for literal in URL_LITERAL_PREFIXES:
            if literal in body:
                quote = literal[len("href=")] if literal.startswith("href=") else literal[0]
                # Insert the mount root right after the opening quote.
                cut = literal.index(quote) + 1
                body = body.replace(literal, literal[:cut] + root + literal[cut:])
        resp.set_data(_with_livereload(body))
        return resp

    return app


SUB_APPS = (annotation_app.app, web_app.app, crop_balancer_app.app)


def _register_hooks():
    """Cheap: attach the URL-rewrite + live-reload after_request hooks. Safe to
    run in every process (the reloader's watcher parent and the serving child
    both import this module)."""
    for sub in SUB_APPS:
        _rewrite_absolute_urls(sub)


def _start_workers():
    """Heavy: load settings and start background threads. Runs ONLY in the
    process that actually serves requests — never in the reloader's watcher
    parent — so models and threads aren't spun up twice."""
    web_app.load_settings()
    web_app.configure_quiet_logging()
    threading.Thread(target=web_app.prerender_worker, daemon=True).start()


# --------------------------------------------------------------------------- #
# The shell: landing page + the handful of routes shared by every tool
# --------------------------------------------------------------------------- #
shell = Flask("vision_suite")


def _render_landing():
    email = auth.current_email() or ""
    admin_link = ('<a class="admin" href="/admin">Manage users</a>'
                  if auth.is_admin(email) else "")
    user_menu = (f'<span class="usermenu"><span class="who">{html.escape(email)}</span>'
                 f'{admin_link}<a href="/logout">Log out</a></span>') if email else ""
    page = LANDING_TEMPLATE.read_text(encoding="utf-8")
    return (page
            .replace("__USER_MENU__", user_menu)
            .replace("__THEME__", theme.stylesheet())
            .replace("__THEME_SCRIPT__", theme.THEME_SCRIPT)
            .replace("__THEME_JS__", theme.THEME_JS)
            .replace("__THEME_BUTTON__", theme.THEME_BUTTON))


@shell.get("/")
def home():
    resp = Response(_render_landing(), mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@shell.get("/home")
def home_redirect():
    # Sub-apps link here to get back to the picker. They can't just link to "/":
    # the URL rewrite above turns a root-absolute "/" into their own mount point.
    return redirect("/")


@shell.get("/api/convex/status")
def convex_status():
    """Read-only connection check: is Convex configured and reachable? Calls no
    Convex function, so it reads and writes nothing on the deployment."""
    url = convex_client.convex_url()
    if not convex_client.is_configured():
        return {"configured": False, "url": "", "reachable": False, "authed": False,
                "detail": "CONVEX_URL not set in .env"}
    ok, detail = convex_client.ping()
    from core import db_log
    return {"configured": True, "url": url, "reachable": ok,
            "authed": bool(convex_client.auth_token()),
            "logging": db_log.stats(), "detail": detail}


# --------------------------------------------------------------------------- #
# Heatmap: cascading filters from the camera registry + a density grid built
# from the logged crops. All read-only Convex queries (no secret needed).
# --------------------------------------------------------------------------- #
def _convex_query(name, args):
    client = convex_client.get_client()
    if client is None:
        return None
    return client.query(name, args)


@shell.get("/api/heatmap/cameras")
def heatmap_cameras():
    """The full camera registry — the page builds the company→site→camera
    cascade from this."""
    try:
        return {"cameras": _convex_query("cameras:all", {}) or []}
    except Exception as exc:
        return {"cameras": [], "error": str(exc)}


@shell.get("/api/heatmap/taskprojects")
def heatmap_taskprojects():
    """Distinct CVAT (project, task) pairs seen across crops — the heatmap's
    project/task filter options."""
    try:
        return {"pairs": _convex_query("crops:taskProjects", {}) or []}
    except Exception as exc:
        return {"pairs": [], "error": str(exc)}


@shell.post("/api/heatmap/cvat/sync")
def heatmap_cvat_sync():
    """Pull annotations for the given CVAT tasks into the database."""
    from core import cvat_sync
    payload = request.get_json(silent=True) or {}
    project = str(payload.get("project", "")).strip()
    tasks = payload.get("tasks")
    if not project or not isinstance(tasks, list):
        return {"ok": False, "error": "project and tasks are required"}, 400
    err = cvat_sync.start(project, tasks)
    if err:
        return {"ok": False, "error": err}, 409
    return {"ok": True, "count": len(tasks)}


@shell.get("/api/heatmap/cvat/sync/status")
def heatmap_cvat_sync_status():
    from core import cvat_sync
    return cvat_sync.status()


@shell.get("/api/heatmap/cvat/cameras")
def heatmap_cvat_cameras():
    """Cameras actually present in the crops of a CVAT project/task."""
    args = {k: request.args[k] for k in ("project", "task") if request.args.get(k)}
    try:
        return {"cameras": _convex_query("crops:camerasFor", args) or []}
    except Exception as exc:
        return {"cameras": [], "error": str(exc)}


@shell.get("/api/heatmap/classes")
def heatmap_classes():
    """Class labels present in the annotations matching the given filters —
    the dropdown only offers what's actually available."""
    args = {k: request.args[k] for k in ("company", "site", "camera", "project", "task")
            if request.args.get(k)}
    try:
        return {"classes": _convex_query("annotations:classesFor", args) or []}
    except Exception as exc:
        return {"classes": [], "error": str(exc)}


@shell.get("/api/heatmap/data")
def heatmap_data():
    """Bin the filtered crops into a density grid (16:9). Returns the grid of
    counts, its max, and the total point count."""
    filters = {k: request.args[k] for k in ("company", "site", "camera", "className",
                                            "project", "task")
               if request.args.get(k)}
    cols = max(8, min(int(request.args.get("cols", 40)), 80))
    rows = max(5, round(cols * 9 / 16))
    grid = [[0] * cols for _ in range(rows)]
    peak = 0
    count = 0
    try:
        res = _convex_query("crops:heatmap", filters)
    except Exception as exc:
        return {"cols": cols, "rows": rows, "cells": grid, "max": 0, "count": 0, "error": str(exc)}
    if res:
        count = int(res.get("count", 0))
        for p in res.get("points", []):
            gx = min(max(int(p["cx"] * cols), 0), cols - 1)
            gy = min(max(int(p["cy"] * rows), 0), rows - 1)
            grid[gy][gx] += 1
            if grid[gy][gx] > peak:
                peak = grid[gy][gx]
    return {"cols": cols, "rows": rows, "cells": grid, "max": peak, "count": count}


@shell.get("/api/heatmap/unknown")
def heatmap_unknown():
    """Cameras found in crops that aren't in the registry — drives the prompt."""
    try:
        return {"cameras": _convex_query("crops:unknownCameras", {}) or []}
    except Exception as exc:
        return {"cameras": [], "error": str(exc)}


@shell.post("/api/heatmap/register")
def heatmap_register():
    """Add a discovered camera to the registry and back-fill its crops."""
    client = convex_client.get_client()
    if client is None:
        return {"ok": False, "error": "Convex not configured"}, 400
    payload = request.get_json(silent=True) or {}
    company = str(payload.get("company", "")).strip()
    site = str(payload.get("site", "")).strip()
    camera = str(payload.get("camera", "")).strip()
    alias = str(payload.get("alias", "")).strip()
    if not (company and site and camera):
        return {"ok": False, "error": "company, site and camera are all required"}, 400
    try:
        args = {
            "secret": convex_client.shared_secret(),
            "company": company, "site": site, "camera": camera,
        }
        if alias and alias != camera:
            args["alias"] = alias  # renamed at registration — keep matching its videos
        res = client.mutation("cameras:register", args)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 502
    from core import cameras as _cams
    _cams.refresh()  # so newly-arriving crops from this camera auto-tag
    return {"ok": True, "tagged": (res or {}).get("tagged", 0)}


@shell.post("/api/heatmap/import")
def heatmap_import_start():
    """Start importing a folder of YOLO label files into the database."""
    from core import label_import
    payload = request.get_json(silent=True) or {}
    path = str(payload.get("path", "")).strip()
    project = str(payload.get("project", "")).strip()
    task = str(payload.get("task", "")).strip()
    if not path:
        return {"ok": False, "error": "A folder path is required."}, 400
    if not (project and task):
        return {"ok": False, "error": "Select the CVAT project and task this data belongs to."}, 400
    err = label_import.start(path, project, task)
    if err:
        return {"ok": False, "error": err}, 409
    return {"ok": True}


@shell.get("/api/heatmap/import/status")
def heatmap_import_status():
    from core import label_import
    return label_import.status()


@shell.get("/heatmap")
def heatmap_page():
    page = (paths.ROOT / "templates" / "heatmap.html").read_text(encoding="utf-8")
    page = (page
            .replace("__THEME__", theme.stylesheet())
            .replace("__THEME_SCRIPT__", theme.THEME_SCRIPT)
            .replace("__THEME_JS__", theme.THEME_JS)
            .replace("__THEME_BUTTON__", theme.THEME_BUTTON))
    resp = Response(page, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@shell.get("/cameras")
def cameras_page():
    page = (paths.ROOT / "templates" / "cameras.html").read_text(encoding="utf-8")
    page = (page
            .replace("__THEME__", theme.stylesheet())
            .replace("__THEME_SCRIPT__", theme.THEME_SCRIPT)
            .replace("__THEME_JS__", theme.THEME_JS)
            .replace("__THEME_BUTTON__", theme.THEME_BUTTON))
    resp = Response(page, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


def _camera_mutation(name, args):
    """Run a registry mutation and refresh the resolver cache."""
    client = convex_client.get_client()
    if client is None:
        return None, ({"ok": False, "error": "Convex not configured"}, 400)
    try:
        res = client.mutation(name, {"secret": convex_client.shared_secret(), **args})
    except Exception as exc:
        return None, ({"ok": False, "error": str(exc)}, 502)
    from core import cameras as _cams
    _cams.refresh()
    return res, None


@shell.post("/api/cameras/update")
def cameras_update():
    """Edit a registry camera (rename / company / site / thumbnail / alias)."""
    payload = request.get_json(silent=True) or {}
    camera = str(payload.get("camera", "")).strip()
    if not camera:
        return {"ok": False, "error": "camera is required"}, 400
    args = {"camera": camera}
    for k in ("newName", "company", "site", "thumbnail", "alias"):
        if k in payload:
            args[k] = str(payload[k]).strip()
    merge = payload.get("mergeFrom")
    if isinstance(merge, list):
        merge = [str(m).strip() for m in merge if str(m).strip()]
        if merge:
            args["mergeFrom"] = merge
    res, err = _camera_mutation("cameras:update", args)
    if err:
        return err
    return {"ok": True, "retagged": (res or {}).get("retagged", 0)}


@shell.post("/api/cameras/delete")
def cameras_delete():
    """Remove a camera from the registry (its crops are untagged, not deleted)."""
    payload = request.get_json(silent=True) or {}
    camera = str(payload.get("camera", "")).strip()
    if not camera:
        return {"ok": False, "error": "camera is required"}, 400
    res, err = _camera_mutation("cameras:remove", args={"camera": camera})
    if err:
        return err
    return {"ok": True, "untagged": (res or {}).get("untagged", 0)}


@shell.get("/frames")
def frames_page():
    page = (paths.ROOT / "templates" / "frames.html").read_text(encoding="utf-8")
    page = (page
            .replace("__THEME__", theme.stylesheet())
            .replace("__THEME_SCRIPT__", theme.THEME_SCRIPT)
            .replace("__THEME_JS__", theme.THEME_JS)
            .replace("__THEME_BUTTON__", theme.THEME_BUTTON))
    resp = Response(page, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


_FRAMES_UPLOAD = paths.STATE_DIR / "frame_extract_upload.csv"


@shell.post("/api/frames/upload")
def frames_upload():
    """Receive the dropped CSV, stash it, and return its headers."""
    from core import frame_extract
    f = request.files.get("file")
    if f is None:
        return {"ok": False, "error": "No file received."}, 400
    if not (f.filename or "").lower().endswith(".csv"):
        return {"ok": False, "error": "Please drop a .csv file."}, 400
    raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    headers = frame_extract.read_headers(text)
    if not headers:
        return {"ok": False, "error": "Could not read any headers from the CSV."}, 400
    _FRAMES_UPLOAD.parent.mkdir(parents=True, exist_ok=True)
    _FRAMES_UPLOAD.write_text(text, encoding="utf-8")
    rows = max(0, text.count("\n") - 1)
    return {"ok": True, "headers": headers, "rows": rows,
            "name": os.path.basename(f.filename or "upload.csv")}


@shell.post("/api/frames/start")
def frames_start():
    """Refresh SAS URLs in the chosen column and extract the chosen frame."""
    from core import frame_extract
    payload = request.get_json(silent=True) or {}
    column = str(payload.get("column", "")).strip()
    if not column:
        return {"ok": False, "error": "Pick the column that holds the video URLs."}, 400
    batch = os.path.splitext(str(payload.get("name", "batch")))[0]
    err = frame_extract.start(_FRAMES_UPLOAD, column,
                              payload.get("frames", payload.get("frame", "24")), batch,
                              mode=str(payload.get("mode", "frames")),
                              model_path=str(payload.get("model", "")))
    if err:
        return {"ok": False, "error": err}, 409
    return {"ok": True}


@shell.get("/api/frames/models")
def frames_models():
    """The model pool, for the person-crops model picker."""
    from core import frame_extract
    return {"models": frame_extract.list_models()}


@shell.post("/api/frames/stop")
def frames_stop():
    """Stop a running extraction (in-flight grabs finish, the rest cancel)."""
    from core import frame_extract
    err = frame_extract.stop()
    if err:
        return {"ok": False, "error": err}, 409
    return {"ok": True}


@shell.get("/api/frames/status")
def frames_status():
    from core import frame_extract
    return frame_extract.status()


@shell.get("/download")
def download_page():
    page = (paths.ROOT / "templates" / "download.html").read_text(encoding="utf-8")
    page = (page
            .replace("__THEME__", theme.stylesheet())
            .replace("__THEME_SCRIPT__", theme.THEME_SCRIPT)
            .replace("__THEME_JS__", theme.THEME_JS)
            .replace("__THEME_BUTTON__", theme.THEME_BUTTON))
    resp = Response(page, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@shell.post("/api/cvatdl/start")
def cvatdl_start():
    """Package a CVAT task (images + YOLO labels + data.yaml) as a zip."""
    from core import cvat_download
    payload = request.get_json(silent=True) or {}
    err = cvat_download.start(payload.get("task_id"),
                              str(payload.get("task_name", "task")))
    if err:
        return {"ok": False, "error": err}, 409
    return {"ok": True}


@shell.get("/api/cvatdl/status")
def cvatdl_status():
    from core import cvat_download
    return cvat_download.status()


@shell.get("/api/cvatdl/list")
def cvatdl_list():
    """Already-built zips, newest first — downloads survive server restarts."""
    out = []
    if paths.CVAT_DL_ROOT.is_dir():
        for p in paths.CVAT_DL_ROOT.glob("*.zip"):
            st = p.stat()
            out.append({"name": p.name, "size_mb": round(st.st_size / 1e6, 1),
                        "mtime": int(st.st_mtime)})
    out.sort(key=lambda z: z["mtime"], reverse=True)
    return {"zips": out}


@shell.get("/api/cvatdl/file")
def cvatdl_file():
    """Serve a built zip as a browser download. Stateless: any zip in
    CVAT_DL_ROOT can be fetched by name, so a server reload between building
    and clicking Download can never break the button."""
    from flask import send_file
    name = os.path.basename(str(request.args.get("name", "")).strip())
    if not name:
        from core import cvat_download
        name = cvat_download.status().get("zip_name") or ""
    zip_path = paths.CVAT_DL_ROOT / name
    if not (name.endswith(".zip") and zip_path.is_file()):
        return {"ok": False, "error": "No finished zip to download."}, 404
    return send_file(zip_path, as_attachment=True, download_name=name)


# --------------------------------------------------------------------------- #
# Event Review — raw vs inference (core/event_review.py does the work)
# --------------------------------------------------------------------------- #
@shell.get("/review")
def review_page():
    page = (paths.ROOT / "templates" / "review.html").read_text(encoding="utf-8")
    page = (page
            .replace("__THEME__", theme.stylesheet())
            .replace("__THEME_SCRIPT__", theme.THEME_SCRIPT)
            .replace("__THEME_JS__", theme.THEME_JS)
            .replace("__THEME_BUTTON__", theme.THEME_BUTTON))
    resp = Response(page, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


def _ev():
    from core import event_review
    return event_review


def _ev_guard(fn):
    """Run one event-review handler, mapping ApiError to (json, code)."""
    try:
        return fn()
    except Exception as exc:
        ev = _ev()
        if isinstance(exc, ev.ApiError):
            return {"detail": exc.message}, exc.code
        raise


@shell.post("/api/review/upload")
def review_upload():
    def go():
        f = request.files.get("file")
        if f is None:
            raise _ev().ApiError(400, "No file received.")
        return _ev().create_session(f.filename, f.read())
    return _ev_guard(go)


@shell.post("/api/review/columns")
def review_columns():
    def go():
        ev = _ev()
        p = request.get_json(silent=True) or {}
        s = ev.get_session(p.get("session_id"))
        raw_col, inf_col = p.get("raw_col"), p.get("inference_col")
        for c in (raw_col, inf_col):
            if c is None or not (0 <= int(c) < len(s.headers)):
                raise ev.ApiError(400, "Pick a valid raw and inference column.")
        s.raw_col, s.inf_col = int(raw_col), int(inf_col)
        return {"ok": True, "raw_col": s.raw_col, "inference_col": s.inf_col}
    return _ev_guard(go)


@shell.post("/api/review/refresh")
def review_refresh():
    def go():
        ev = _ev()
        s = ev.get_session((request.get_json(silent=True) or {}).get("session_id"))
        ev.start_refresh(s)
        with s.lock:
            return dict(s.refresh_state)
    return _ev_guard(go)


@shell.get("/api/review/refresh/status")
def review_refresh_status():
    def go():
        s = _ev().get_session(request.args.get("session_id"))
        with s.lock:
            return dict(s.refresh_state)
    return _ev_guard(go)


@shell.post("/api/review/frames/start")
def review_frames_start():
    def go():
        ev = _ev()
        s = ev.get_session((request.get_json(silent=True) or {}).get("session_id"))
        ev.start_frames(s)
        with s.lock:
            return dict(s.frame_state)
    return _ev_guard(go)


@shell.post("/api/review/frames/stop")
def review_frames_stop():
    def go():
        s = _ev().get_session((request.get_json(silent=True) or {}).get("session_id"))
        s.abort.set()
        return {"ok": True}
    return _ev_guard(go)


@shell.get("/api/review/frames/status")
def review_frames_status():
    def go():
        s = _ev().get_session(request.args.get("session_id"))
        with s.lock:
            return dict(s.frame_state)
    return _ev_guard(go)


@shell.get("/api/review/events")
def review_events():
    def go():
        ev = _ev()
        s = ev.get_session(request.args.get("session_id"))
        return ev.events_payload(s,
                                 offset=int(request.args.get("offset", 0)),
                                 limit=int(request.args.get("limit", 100000)))
    return _ev_guard(go)


@shell.get("/api/review/event/<int:idx>/raw")
def review_event_raw(idx):
    def go():
        ev = _ev()
        s = ev.get_session(request.args.get("session_id"))
        if not (0 <= idx < len(s.rows)):
            raise ev.ApiError(404, "no such event")
        row = s.rows[idx]
        return {"idx": idx, "fields": {h: row[i] for i, h in enumerate(s.headers)}}
    return _ev_guard(go)


@shell.get("/api/review/frame/<int:idx>/<kind>")
def review_frame(idx, kind):
    def go():
        from flask import send_file
        ev = _ev()
        s = ev.get_session(request.args.get("session_id"))
        if kind not in ("raw", "inference"):
            raise ev.ApiError(400, "kind must be raw or inference")
        p = s.frame_path(idx, kind)
        if not (p.exists() and p.stat().st_size > 0):
            raise ev.ApiError(404, "frame not extracted yet")
        resp = send_file(p, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    return _ev_guard(go)


@shell.post("/api/review/video/prepare")
def review_video_prepare():
    def go():
        ev = _ev()
        p = request.get_json(silent=True) or {}
        s = ev.get_session(p.get("session_id"))
        idx = int(p["idx"])
        started = {}
        for kind in (p.get("kinds") or ["raw", "inference"]):
            if kind not in ("raw", "inference"):
                continue
            key = f"{idx}_{kind}"
            with s.lock:
                st = s.video_status.get(key, {})
            if st.get("state") == "downloading":
                started[kind] = st
                continue
            if s.video_path(idx, kind).exists():
                started[kind] = {"state": "ready", "pct": 100}
                with s.lock:
                    s.video_status[key] = started[kind]
                continue
            with s.lock:
                s.video_status[key] = {"state": "queued", "pct": 0}
            s.video_pool.submit(ev.download_video, s, idx, kind)
            started[kind] = {"state": "queued", "pct": 0}
        return started
    return _ev_guard(go)


@shell.get("/api/review/video/status")
def review_video_status():
    def go():
        s = _ev().get_session(request.args.get("session_id"))
        idx = int(request.args.get("idx", -1))
        out = {}
        for kind in ("raw", "inference"):
            if s.video_path(idx, kind).exists():
                out[kind] = {"state": "ready", "pct": 100}
            else:
                with s.lock:
                    out[kind] = s.video_status.get(f"{idx}_{kind}",
                                                   {"state": "none", "pct": 0})
        return out
    return _ev_guard(go)


@shell.get("/api/review/video/file/<int:idx>/<kind>")
def review_video_file(idx, kind):
    def go():
        from flask import send_file
        ev = _ev()
        s = ev.get_session(request.args.get("session_id"))
        p = s.video_path(idx, kind)
        if kind not in ("raw", "inference") or not p.exists():
            raise ev.ApiError(404, "video not downloaded yet")
        return send_file(p, mimetype="video/mp4", conditional=True)
    return _ev_guard(go)


@shell.get("/api/review/tags")
def review_tags_get():
    def go():
        s = _ev().get_session(request.args.get("session_id"))
        with s.lock:
            return {"tags": {str(k): v for k, v in s.tags.items()},
                    "tag_counts": s.tag_counts(), "tagged_events": len(s.tags)}
    return _ev_guard(go)


@shell.post("/api/review/tags")
def review_tags_set():
    def go():
        ev = _ev()
        p = request.get_json(silent=True) or {}
        s = ev.get_session(p.get("session_id"))
        return ev.set_tags(s, int(p.get("idx", -1)), p)
    return _ev_guard(go)


@shell.post("/api/review/tags/delete")
def review_tags_delete():
    def go():
        ev = _ev()
        p = request.get_json(silent=True) or {}
        s = ev.get_session(p.get("session_id"))
        return ev.delete_tag(s, p.get("name", ""))
    return _ev_guard(go)


@shell.get("/api/review/download/tagged")
def review_download_tagged():
    def go():
        from flask import send_file
        ev = _ev()
        s = ev.get_session(request.args.get("session_id"))
        import json as _json
        filters = None
        raw = request.args.get("filters")
        if raw:
            try:
                parsed = _json.loads(raw)
                if isinstance(parsed, list):
                    filters = [str(x) for x in parsed]
            except ValueError:
                pass
        out = ev.write_tagged_csv(s, filters)
        return send_file(out, mimetype="text/csv", as_attachment=True,
                         download_name=out.name)
    return _ev_guard(go)


@shell.get("/api/review/download/refreshed")
def review_download_refreshed():
    def go():
        from flask import send_file
        ev = _ev()
        s = ev.get_session(request.args.get("session_id"))
        out = s.dir / f"{os.path.splitext(s.filename)[0]}_refreshed.csv"
        if not out.exists():
            raise ev.ApiError(404, "Run the SAS refresh first.")
        return send_file(out, mimetype="text/csv", as_attachment=True,
                         download_name=out.name)
    return _ev_guard(go)


@shell.get("/api/review/state")
def review_state():
    return _ev().state_payload()


# --------------------------------------------------------------------------- #
# Auth: login (password or Google), logout, and the admin allowlist page.
# --------------------------------------------------------------------------- #
def _render_template(name, **subs):
    page = (paths.ROOT / "templates" / name).read_text(encoding="utf-8")
    page = (page
            .replace("__THEME__", theme.stylesheet())
            .replace("__THEME_SCRIPT__", theme.THEME_SCRIPT)
            .replace("__THEME_JS__", theme.THEME_JS)
            .replace("__THEME_BUTTON__", theme.THEME_BUTTON))
    for k, v in subs.items():
        page = page.replace(k, v)
    return page


def _login_page(error=""):
    err_html = f'<div class="err">{error}</div>' if error else ""
    return _render_template("login.html", __ERROR__=err_html)


@shell.get("/login")
def login_get():
    if auth.current_email():
        return redirect("/")
    return Response(_login_page(), mimetype="text/html")


@shell.post("/login")
def login_post():
    email = request.form.get("email", "")
    password = request.form.get("password", "")
    if auth.check_password(email, password):
        return auth.set_session(redirect("/"), email.strip().lower())
    return Response(_login_page("Wrong email or password."), mimetype="text/html", status=401)


@shell.get("/logout")
def logout():
    return auth.clear_session(redirect("/login"))


def _require_admin():
    email = auth.current_email()
    return email if auth.is_admin(email) else None


@shell.get("/admin")
def admin_get():
    if not _require_admin():
        return redirect("/")
    msg = {"added": ('<div class="msg ok">User saved. They can sign in with that email and password.</div>'),
           "removed": ('<div class="msg ok">User removed.</div>'),
           "bad": ('<div class="msg err">Enter a valid email and a password.</div>'),
           "err": ('<div class="msg err">Couldn\'t reach the database. Try again.</div>'),
           }.get(request.args.get("m", ""), "")
    try:
        users = auth.list_users()
    except Exception:
        users = []
    admin = html.escape(auth.admin_email())
    rows = [f'<tr><td>{admin}<span class="tag">admin</span></td><td class="act"></td></tr>']
    for e in users:
        esc = html.escape(e)
        rows.append(
            f'<tr><td>{esc}</td><td class="act">'
            f'<form class="inline" method="post" action="/admin/users/remove">'
            f'<input type="hidden" name="email" value="{esc}">'
            f'<button class="rm" type="submit">Remove</button></form></td></tr>')
    if not users:
        rows.append('<tr><td colspan="2"><div class="empty">No other users yet. Add an email above.</div></td></tr>')
    return Response(_render_template("admin.html", __MSG__=msg, __ROWS__="".join(rows)),
                    mimetype="text/html")


@shell.post("/admin/users/add")
def admin_add():
    if not _require_admin():
        return redirect("/")
    email = request.form.get("email", "")
    password = request.form.get("password", "")
    try:
        ok = auth.add_user(email, password, by=auth.current_email())
    except Exception:
        return redirect("/admin?m=err")
    return redirect("/admin?m=" + ("added" if ok else "bad"))


@shell.post("/admin/users/remove")
def admin_remove():
    if not _require_admin():
        return redirect("/")
    try:
        auth.remove_user(request.form.get("email", ""))
    except Exception:
        return redirect("/admin?m=err")
    return redirect("/admin?m=removed")


@shell.get("/__livereload__")
def livereload_token():
    return START_TOKEN, 200, {"Content-Type": "text/plain"}


@shell.after_request
def _shell_livereload(resp):
    if resp.mimetype == "text/html" and not resp.direct_passthrough:
        resp.set_data(_with_livereload(resp.get_data(as_text=True)))
    return resp


def build_application():
    _register_hooks()
    # Gate every app: the shell and all three tools require a valid session. They
    # share the AUTH_SECRET, so each verifies the same cookie independently.
    for app in (shell, annotation_app.app, web_app.app, crop_balancer_app.app):
        app.before_request(auth.gate)
    return DispatcherMiddleware(shell, {
        "/annotate": annotation_app.app,
        "/webapp": web_app.app,
        "/crop": crop_balancer_app.app,
    })


application = build_application()


def _open_browser_when_ready(url, delay_sec=1.0):
    if os.environ.get("NO_BROWSER"):
        return
    import webbrowser

    def _open():
        time.sleep(delay_sec)
        webbrowser.open(url)

    threading.Thread(target=_open, daemon=True).start()


_LOCK_PATH = paths.STATE_DIR / "suite.lock"
_lock_fd = None  # kept open for the process lifetime so the lock stays held


def _enforce_single_instance():
    """Refuse to start a second copy of the suite.

    Each suite process loads torch + three YOLO models (~4 GB) and, with no GPU
    on this box, runs them on the CPU. A second copy started by accident — easy
    now that the port finder means a re-launch no longer fails on a busy port —
    can exhaust RAM and freeze the whole machine. So the first instance takes an
    flock; a later one finds it held, points the user at the running server, and
    bows out. flock frees itself when the holder dies, so there is no stale lock
    to clean up (works even through the os._exit in _hard_exit).
    """
    global _lock_fd
    _lock_fd = open(_LOCK_PATH, "a+")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lock_fd.seek(0)
        where = _lock_fd.read().strip()
        print("\n  The Vision Suite is already running"
              + (f" ({where})" if where else "") + ".")
        print("  Open that one, or stop it first (Ctrl-C in its terminal) "
              "before starting another.\n")
        raise SystemExit(1)
    # Won the lock — record who/where for the next launch's message.
    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(f"pid {os.getpid()} · http://127.0.0.1:{PORT}")
    _lock_fd.flush()


def _hard_exit(code):
    """Leave without letting the interpreter finalize.

    The prerender worker is a daemon thread that spends its life inside OpenCV
    and torch. Daemon threads aren't joined at shutdown, so finalizing the
    interpreter out from under a native call aborts the process — "terminate
    called without an active exception" — which under the reloader kills the
    server instead of restarting it, and drops the port with it.

    So we run the atexit handlers ourselves (web_app registers one to delete a
    downloaded video), flush our own output, and then exit the hard way, leaving
    the native runtimes alone.
    """
    atexit._run_exitfuncs()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code if isinstance(code, int) else 0)


if __name__ == "__main__":
    # Hold the single-instance lock in the long-lived top-level process. Under the
    # reloader that is the watcher parent (WERKZEUG_RUN_MAIN unset); the restarted
    # child inherits the running server and must NOT re-check, so it skips this.
    if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        _enforce_single_instance()

    # Auto-reload on any .py (or the landing template) change — the "npm run dev"
    # behaviour. NO_RELOAD=1 disables it (e.g. for production).
    use_reload = os.environ.get("NO_RELOAD") != "1"
    # With the reloader this script runs in two processes: a watcher parent and
    # the serving child (marked by WERKZEUG_RUN_MAIN). Do the heavy startup and
    # the banner only in the child so nothing loads twice.
    serving = (not use_reload) or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if serving:
        _start_workers()
        if PORT != _PREFERRED_PORT:
            print(f"\n  Port {_PREFERRED_PORT} is in use — using {PORT} instead.")
        url = f"http://127.0.0.1:{PORT}"
        print(f"\n  Vision Suite on {url}   (auto-reloads on edits)")
        print(f"    Landing             {url}/")
        print(f"    Annotation Studio   {url}/annotate/")
        print(f"    Inference Web App   {url}/webapp/")
        print(f"    Crop Tools          {url}/crop/\n")
        if convex_client.is_configured():
            ok, detail = convex_client.ping()
            token_note = "auth token set" if convex_client.auth_token() else "no auth token"
            print(f"  Convex: {convex_client.convex_url()} "
                  f"({'reachable' if ok else 'UNREACHABLE — ' + detail}, {token_note})\n")
        else:
            print("  Convex: not configured (set CONVEX_URL in .env)\n")
        _open_browser_when_ready(url)
    # The reloader signals "restart me" by raising SystemExit(3) on the main
    # thread; Ctrl-C arrives as KeyboardInterrupt. Either way we leave through
    # _hard_exit so the exit code still reaches the watcher parent (3 = restart)
    # without tripping the shutdown abort described there.
    try:
        run_simple(
            HOST, PORT, application,
            threaded=True,
            use_reloader=use_reload,
            use_debugger=False,
            extra_files=[str(LANDING_TEMPLATE), str(paths.ROOT / "templates" / "heatmap.html")],
            # The reloader watches *.zip by default (zipimport support), so a
            # tool writing a zip under data/ would restart the server and kill
            # the very job writing it. Runtime output must never trigger reloads.
            exclude_patterns=[str(paths.DATA_DIR / "*"), str(paths.STATE_DIR / "*")],
        )
        _hard_exit(0)
    except SystemExit as exc:
        _hard_exit(exc.code)
    except KeyboardInterrupt:
        _hard_exit(0)
