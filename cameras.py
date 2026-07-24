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
import threading

import convex_client
import cropnames

_lock = threading.Lock()
_registry = None  # list of (camera, company, site), sorted longest-name-first


def _load():
    """Fetch the camera registry from Convex once and cache it. Best-effort:
    on any failure the resolver simply matches nothing."""
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
                        rows.append((r["camera"], r["company"], r["site"]))
            except Exception:
                rows = []
            # Longest camera name first, so the most specific prefix wins if one
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
    for camera, company, site in _load():
        if stem == camera:
            return (company, site, camera)
        if stem.startswith(camera) and stem[len(camera):len(camera) + 1] in ("_", "-", ".", " "):
            return (company, site, camera)
    return (None, None, None)


def refresh():
    """Drop the cache so the next resolve() re-fetches (after a re-seed)."""
    global _registry
    with _lock:
        _registry = None
