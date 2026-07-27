#!/usr/bin/env python3
"""
Login + access control for the Vision Suite.

Everyone signs in with an email and password. One admin comes from .env; the
admin adds other users on the /admin page, setting each person's password at that
time. Passwords are hashed in this backend (werkzeug) — the plaintext never
reaches Convex, only the hash is stored. A signed cookie carries the session
across the shell and all three mounted tools (they share the same AUTH_SECRET,
so any of them can verify it).

Design notes:
- The allowlist + password hashes live in Convex, cached briefly here and
  re-checked per request, so removing someone takes effect immediately.
- The admin is configured in .env and always allowed, so a Convex blip can't
  lock the admin out.
- Cookies are HttpOnly + SameSite=Lax. There's no Secure flag because the suite
  runs over plain HTTP on a LAN — fine for that, but don't expose it to the
  public internet without TLS in front.
"""
import os
import threading
import time

from flask import redirect, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.security import check_password_hash, generate_password_hash

from core import convex_client

COOKIE = "vs_auth"
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


def add_user(email, password, by=None):
    """Add (or update) a user with a password. Re-adding an existing email just
    updates their password. The password is hashed here; only the hash is stored."""
    email = (email or "").strip().lower()
    if not email or "@" not in email or not (password or "").strip():
        return False
    client = convex_client.get_client()
    if client is None:
        return False
    client.mutation("users:add", {
        "secret": convex_client.shared_secret(),
        "email": email,
        "passwordHash": generate_password_hash(password),
        "addedBy": (by or ""),
    })
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
# Password login (admin from .env; everyone else from Convex)
# --------------------------------------------------------------------------- #
def _user_hash(email):
    """The stored password hash for a non-admin user, or None."""
    client = convex_client.get_client()
    if client is None:
        return None
    try:
        rec = client.query("users:get", {"secret": convex_client.shared_secret(), "email": email})
    except Exception:
        return None
    return (rec or {}).get("passwordHash")


def check_password(email, password):
    """True if this email + password is a valid login (admin or an added user)."""
    email = (email or "").strip().lower()
    if not email or not (password or ""):
        return False
    if email == admin_email():
        hashed = _env("AUTH_ADMIN_PASSWORD_HASH")
        return bool(hashed) and check_password_hash(hashed, password)
    hashed = _user_hash(email)
    return bool(hashed) and check_password_hash(hashed, password)


# --------------------------------------------------------------------------- #
# The gate — a before_request hook registered on every app
# --------------------------------------------------------------------------- #
def gate():
    """Return a response to block the request, or None to let it through.

    Public paths (login, logout, live-reload) are always allowed; everything else
    needs a valid, still-allowed session.
    """
    if request.method == "OPTIONS":
        return None  # let CORS preflight through
    p = request.path or "/"
    if p == "/login" or p == "/logout" or p == "/__livereload__":
        return None
    if current_email():
        return None
    # Not authenticated (or no longer allowed).
    accept = request.headers.get("Accept", "")
    wants_json = p.startswith("/api/") or "application/json" in accept or p.endswith(".mjpg")
    if wants_json:
        return ("Unauthorized", 401)
    return redirect("/login")
