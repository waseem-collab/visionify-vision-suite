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

import auth
import convex_client
import paths
import ports
import theme

paths.ensure_dirs()

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
    import db_log
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


@shell.get("/api/heatmap/classes")
def heatmap_classes():
    """Distinct annotation class labels, for the class filter."""
    try:
        return {"classes": _convex_query("annotations:classes", {}) or []}
    except Exception as exc:
        return {"classes": [], "error": str(exc)}


@shell.get("/api/heatmap/data")
def heatmap_data():
    """Bin the filtered crops into a density grid (16:9). Returns the grid of
    counts, its max, and the total point count."""
    filters = {k: request.args[k] for k in ("company", "site", "camera", "className")
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
    if not (company and site and camera):
        return {"ok": False, "error": "company, site and camera are all required"}, 400
    try:
        res = client.mutation("cameras:register", {
            "secret": convex_client.shared_secret(),
            "company": company, "site": site, "camera": camera,
        })
    except Exception as exc:
        return {"ok": False, "error": str(exc)}, 502
    import cameras as _cams
    _cams.refresh()  # so newly-arriving crops from this camera auto-tag
    return {"ok": True, "tagged": (res or {}).get("tagged", 0)}


@shell.post("/api/heatmap/import")
def heatmap_import_start():
    """Start importing a folder of YOLO label files into the database."""
    import label_import
    path = str((request.get_json(silent=True) or {}).get("path", "")).strip()
    if not path:
        return {"ok": False, "error": "A folder path is required."}, 400
    err = label_import.start(path)
    if err:
        return {"ok": False, "error": err}, 409
    return {"ok": True}


@shell.get("/api/heatmap/import/status")
def heatmap_import_status():
    import label_import
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
        )
        _hard_exit(0)
    except SystemExit as exc:
        _hard_exit(exc.code)
    except KeyboardInterrupt:
        _hard_exit(0)
