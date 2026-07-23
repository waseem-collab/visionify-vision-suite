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
import os
import threading
import time
from pathlib import Path

from flask import Flask, Response, redirect, request
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

import paths
import theme

paths.ensure_dirs()

# Importing the tools pulls in torch/ultralytics/opencv, so it takes a moment.
from apps import annotation_app, crop_balancer_app, web_app

LANDING_TEMPLATE = paths.ROOT / "templates" / "landing.html"
PORT = int(os.environ.get("PORT", "8000"))

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
    page = LANDING_TEMPLATE.read_text(encoding="utf-8")
    return (page
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


if __name__ == "__main__":
    # Auto-reload on any .py (or the landing template) change — the "npm run dev"
    # behaviour. NO_RELOAD=1 disables it (e.g. for production).
    use_reload = os.environ.get("NO_RELOAD") != "1"
    # With the reloader this script runs in two processes: a watcher parent and
    # the serving child (marked by WERKZEUG_RUN_MAIN). Do the heavy startup and
    # the banner only in the child so nothing loads twice.
    serving = (not use_reload) or os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if serving:
        _start_workers()
        url = f"http://127.0.0.1:{PORT}"
        print(f"\n  Vision Suite on {url}   (auto-reloads on edits)")
        print(f"    Landing             {url}/")
        print(f"    Annotation Studio   {url}/annotate/")
        print(f"    Inference Web App   {url}/webapp/")
        print(f"    Crop Tools          {url}/crop/\n")
        _open_browser_when_ready(url)
    run_simple(
        os.environ.get("HOST", "0.0.0.0"), PORT, application,
        threaded=True,
        use_reloader=use_reload,
        use_debugger=False,
        extra_files=[str(LANDING_TEMPLATE)],
    )
