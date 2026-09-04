#!/usr/bin/env python3
"""
WSGI entrypoint for the whole Vision Suite.

Serve THIS (``wsgi:application`` or ``run:application``) — never ``apps/*.py``.
Those modules each expose their own ``app``, but they are the individual tools;
serving one gives you just that tool, with no cross-tool routing.
``application`` here is the composed app: the landing page plus all three tools,
mounted under /annotate, /webapp and /crop.

IMPORTANT — run it as ONE process/worker. The suite keeps in-process state (the
model cache, the annotated-frame cache, the prerender thread, the single-instance
lock), so multiple workers would each load the models (~4 GB) and duplicate that
state. With gunicorn:

    gunicorn --workers 1 --threads 8 --timeout 120 wsgi:application

Importing this module only exposes the callable — it does NOT start the web
app's background prerender worker (frames then render on demand, which is fine;
just slightly less smooth). The intended way to run it, with the prerender
worker, auto-reload and automatic free-port selection, is simply:

    python3 run.py        (or:  npm run dev)
"""
from run import application  # the composed WSGI app (landing + tools)

__all__ = ["application"]
