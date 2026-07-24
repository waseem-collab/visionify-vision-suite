#!/usr/bin/env python3
"""
Convex connection for the Vision Suite.

Wires up the Convex Python client so the app can call Convex functions. It
deliberately does NOT define a schema, create tables, or write any data — it
only establishes (and can verify) the connection and hands back a ready client.
Add Convex functions and a schema later, when you want to start reading/writing;
nothing here commits you to a data model.

Configure it in .env:

    CONVEX_URL=https://your-deployment.convex.cloud

Then, once you have Convex functions deployed:

    from convex_client import get_client
    client = get_client()
    result = client.query("yourModule:yourFunction", {"arg": 1})
"""
import os
import socket
import threading
from pathlib import Path
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent
_client = None
_lock = threading.Lock()


def _load_env():
    """Populate CONVEX_URL from the repo-root .env if the app hasn't already.

    The running apps load .env themselves, but this keeps the module usable on
    its own (e.g. a quick connectivity check from a REPL).
    """
    if os.environ.get("CONVEX_URL"):
        return
    env = _ROOT / ".env"
    if not env.exists():
        return
    try:
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except OSError:
        pass


def convex_url():
    """The configured deployment URL, or "" if unset."""
    _load_env()
    return (os.environ.get("CONVEX_URL") or "").strip().rstrip("/")


def auth_token():
    """The auth token used to call authenticated functions, or "" if unset."""
    _load_env()
    return (os.environ.get("CONVEX_AUTH_TOKEN") or "").strip()


def shared_secret():
    """The server-identity secret sent with write calls (see convex/lib/auth.ts)."""
    _load_env()
    return (os.environ.get("CONVEX_SHARED_SECRET") or "").strip()


def is_configured():
    return bool(convex_url())


def get_client():
    """Return a shared ConvexClient, or None if CONVEX_URL is unset.

    The client connects lazily on the first function call, so building it here
    touches no network and creates nothing on the deployment. If CONVEX_AUTH_TOKEN
    is set, it is applied so authenticated functions can be called.
    """
    global _client
    url = convex_url()
    if not url:
        return None
    with _lock:
        if _client is None:
            from convex import ConvexClient
            client = ConvexClient(url)
            token = auth_token()
            if token:
                client.set_auth(token)
            _client = client
        return _client


def ping(timeout=4.0):
    """Confirm the deployment host is reachable, without calling any function.

    A plain TLS-port connect to the deployment host: it proves the URL resolves
    and the server is accepting connections, but reads and writes nothing (there
    are no functions to call yet). Returns (ok, detail).
    """
    url = convex_url()
    if not url:
        return False, "CONVEX_URL not set"
    host = urlparse(url).hostname
    if not host:
        return False, f"bad CONVEX_URL: {url!r}"
    port = urlparse(url).port or (443 if url.startswith("https") else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"{host}:{port} reachable"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
