#!/usr/bin/env python3
"""
Login + access control for the Vision Suite.

One admin (email + password, from .env) plus a local allowlist of emails who may
sign in with Google. Everything else is denied. A signed cookie carries the
session across the shell and all three mounted tools (they share the same
AUTH_SECRET, so any of them can verify it).

Design notes:
- The allowlist lives in a local JSON file, NOT Convex, so login keeps working
  even if Convex is down or unconfigured.
- The allowlist is re-checked on every request, so removing someone takes effect
  immediately (their existing session stops working), not just at next login.
- Cookies are HttpOnly + SameSite=Lax. There's no Secure flag because the suite
  runs over plain HTTP on a LAN — fine for that, but don't expose it to the
  public internet without TLS in front.
"""
import os
import threading
import time
from urllib.parse import urlencode

from flask import redirect, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash

import convex_client

COOKIE = "vs_auth"
STATE_COOKIE = "vs_oauth_state"
MAX_AGE = 7 * 24 * 3600  # a week
_lock = threading.Lock()


def _env(key, default=""):
    return (os.environ.get(key) or default).strip()


def _secret():
    # A weak fallback keeps dev from crashing, but .env always sets a real one.
    return _env("AUTH_SECRET") or "insecure-dev-secret-set-AUTH_SECRET"


def _serializer(salt):
    return URLSafeTimedSerializer(_secret(), salt=salt)


def admin_email():
    return _env("AUTH_ADMIN_EMAIL").lower()


def google_client_id():
    return _env("GOOGLE_CLIENT_ID")


def google_client_secret():
    return _env("GOOGLE_CLIENT_SECRET")


def google_configured():
    return bool(google_client_id() and google_client_secret())


def redirect_uri():
    base = _env("AUTH_REDIRECT_BASE") or "http://localhost:8000"
    return base.rstrip("/") + "/auth/google/callback"


# --------------------------------------------------------------------------- #
# Allowlist (stored in Convex, cached in memory)
# --------------------------------------------------------------------------- #
# The gate re-checks the allowlist on every request, so we cache it briefly to
# avoid a Convex round-trip each time. On a Convex error we keep the last known
# good set, so a transient outage doesn't lock people out. The admin is always
# allowed regardless, so a full outage still lets the admin in.
_ALLOW_TTL = 20.0  # seconds
_allow = {"emails": set(), "at": 0.0, "ok": False}


def _fetch_allowed():
    client = convex_client.get_client()
    if client is None:
        raise RuntimeError("Convex not configured")
    rows = client.query("users:list", {"secret": convex_client.shared_secret()})
    return {str(e).strip().lower() for e in (rows or []) if str(e).strip()}


def _allowed_emails(force=False):
    now = time.monotonic()
    with _lock:
        fresh = _allow["ok"] and (now - _allow["at"] < _ALLOW_TTL)
        if fresh and not force:
            return set(_allow["emails"])
    try:
        emails = _fetch_allowed()
    except Exception:
        with _lock:
            return set(_allow["emails"])  # keep last known good on failure
    with _lock:
        _allow.update(emails=emails, at=now, ok=True)
        return set(emails)


def _invalidate():
    with _lock:
        _allow["at"] = 0.0


def list_users():
    """Allowed emails (excluding the always-allowed admin), sorted."""
    return sorted(_allowed_emails(force=True))


def add_user(email, by=None):
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return False
    client = convex_client.get_client()
    if client is None:
        return False
    client.mutation("users:add", {
        "secret": convex_client.shared_secret(), "email": email, "addedBy": (by or "")})
    _invalidate()
    return True


def remove_user(email):
    email = (email or "").strip().lower()
    client = convex_client.get_client()
    if client is None:
        return
    client.mutation("users:remove", {"secret": convex_client.shared_secret(), "email": email})
    _invalidate()


def is_admin(email):
    return bool(email) and email.lower() == admin_email()


def is_allowed(email):
    if not email:
        return False
    email = email.lower()
    return email == admin_email() or email in _allowed_emails()


# --------------------------------------------------------------------------- #
# Session cookie
# --------------------------------------------------------------------------- #
def make_session_value(email):
    return _serializer("vs-session").dumps({"email": email.lower()})


def _read_email():
    tok = request.cookies.get(COOKIE)
    if not tok:
        return None
    try:
        data = _serializer("vs-session").loads(tok, max_age=MAX_AGE)
        return (data or {}).get("email")
    except (BadSignature, SignatureExpired, Exception):
        return None


def current_email():
    """The logged-in email IF the cookie is valid AND still allowed, else None."""
    email = _read_email()
    return email if (email and is_allowed(email)) else None


def set_session(resp, email):
    resp.set_cookie(COOKIE, make_session_value(email), max_age=MAX_AGE,
                    httponly=True, samesite="Lax", path="/")
    return resp


def clear_session(resp):
    resp.delete_cookie(COOKIE, path="/")
    return resp


# --------------------------------------------------------------------------- #
# Password login (admin only)
# --------------------------------------------------------------------------- #
def check_admin_password(email, password):
    if (email or "").strip().lower() != admin_email():
        return False
    hashed = _env("AUTH_ADMIN_PASSWORD_HASH")
    return bool(hashed) and check_password_hash(hashed, password or "")


# --------------------------------------------------------------------------- #
# Google OAuth (authorization-code flow, verified via tokeninfo — no extra deps)
# --------------------------------------------------------------------------- #
def google_auth_url(state):
    params = {
        "client_id": google_client_id(),
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)


def make_state():
    return _serializer("vs-oauth").dumps({"n": os.urandom(8).hex()})


def valid_state(state):
    if not state:
        return False
    try:
        _serializer("vs-oauth").loads(state, max_age=600)  # 10 min
        return True
    except (BadSignature, SignatureExpired, Exception):
        return False


def google_email_from_code(code):
    """Exchange the auth code and return the verified email (or raise)."""
    import requests
    tok = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": google_client_id(),
            "client_secret": google_client_secret(),
            "redirect_uri": redirect_uri(),
            "grant_type": "authorization_code",
        },
        timeout=15,
    )
    tok.raise_for_status()
    id_token = tok.json().get("id_token")
    if not id_token:
        raise ValueError("no id_token in Google response")
    info = requests.get(
        "https://oauth2.googleapis.com/tokeninfo", params={"id_token": id_token}, timeout=15
    )
    info.raise_for_status()
    claims = info.json()
    if claims.get("aud") != google_client_id():
        raise ValueError("token audience mismatch")
    if str(claims.get("email_verified")).lower() != "true":
        raise ValueError("Google account email is not verified")
    email = (claims.get("email") or "").lower()
    if not email:
        raise ValueError("no email in Google token")
    return email


# --------------------------------------------------------------------------- #
# The gate — a before_request hook registered on every app
# --------------------------------------------------------------------------- #
def gate():
    """Return a response to block the request, or None to let it through.

    Public paths (login, the OAuth dance, logout, live-reload) are always
    allowed; everything else needs a valid, still-allowed session.
    """
    if request.method == "OPTIONS":
        return None  # let CORS preflight through
    p = request.path or "/"
    if p == "/login" or p == "/logout" or p.startswith("/auth/") or p == "/__livereload__":
        return None
    if current_email():
        return None
    # Not authenticated (or no longer allowed).
    accept = request.headers.get("Accept", "")
    wants_json = p.startswith("/api/") or "application/json" in accept or p.endswith(".mjpg")
    if wants_json:
        return ("Unauthorized", 401)
    return redirect("/login")
