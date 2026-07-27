#!/usr/bin/env python3
"""
Resolve a video name to the camera it came from.

Camera names in the registry are globally unique and appear as the prefix of a
video's filename (e.g. ``RMS-Corridor-4A_20260722-052339`` → camera
``RMS-Corridor-4A``). This loads the registry from Convex once and matches a
video stem against it, returning the camera's (company, site, camera).

Used by the crop logger to tag each crop with its camera, so the heatmap can
filter by company / site / camera.
"""
import re
import threading

from core import convex_client, cropnames

# Trailing date/time on a video stem, e.g. "_20260722-052339" or "_20260706_142252".
_DATE_SUFFIX = re.compile(r"_\d{6,8}([_-]\d{2,6})*$")


def guess_camera_name(video_stem):
    """Camera name guessed from a video stem by stripping a trailing date/time.
    Kept in sync with convex/lib/cameraName.ts."""
    stem = cropnames.clean_video_name(video_stem)
    return _DATE_SUFFIX.sub("", stem) or stem

_lock = threading.Lock()
_registry = None  # list of (match, company, site, camera), sorted longest-match-first


def _load():
    """Fetch the camera registry from Convex once and cache it. Best-effort:
    on any failure the resolver simply matches nothing.

    ``match`` is the video-name prefix to match — the camera's alias when the
    user renamed it at registration, otherwise the camera name itself."""
    global _registry
    if _registry is not None:
        return _registry
    with _lock:
        if _registry is None:
            rows = []
            try:
                client = convex_client.get_client()
                if client is not None:
                    for r in client.query("cameras:all", {}):
                        # The camera answers to its own name, its rename alias,
                        # and every name merged into it.
                        names = {r["camera"]}
                        if r.get("alias"):
                            names.add(r["alias"])
                        names.update(r.get("aliases") or [])
                        for n in names:
                            rows.append((n, r["company"], r["site"], r["camera"]))
            except Exception:
                rows = []
            # Longest match first, so the most specific prefix wins if one
            # camera name happens to be a prefix of another.
            rows.sort(key=lambda t: len(t[0]), reverse=True)
            _registry = rows
    return _registry


def resolve(video_stem):
    """Return (company, site, camera) for a video stem, or (None, None, None).

    Matches when a camera name equals the stem or is a prefix followed by a
    separator (``_``, ``-``, ``.`` or space) — i.e. ``<camera>_<date>``.
    """
    stem = cropnames.clean_video_name(video_stem)
    for match, company, site, camera in _load():
        if stem == match:
            return (company, site, camera)
        if stem.startswith(match) and stem[len(match):len(match) + 1] in ("_", "-", ".", " "):
            return (company, site, camera)
    return (None, None, None)


def refresh():
    """Drop the cache so the next resolve() re-fetches (after a re-seed)."""
    global _registry
    with _lock:
        _registry = None
