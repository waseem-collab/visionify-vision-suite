#!/usr/bin/env python3
"""
One naming scheme for every person crop the suite writes.

All three producers — the Crop Balancer, the Inference Web App's "Crop person"
button, and the PPE inference path — route through here so a crop is named the
same way no matter which tool made it:

    <videoname>_<framenumber>_<cx>-<cy>-<w>-<h>.jpg

The coordinates are YOLO format: the crop box's centre and size normalised to the
full frame, each in [0, 1] to six decimals — the same convention as a YOLO label
file, so a crop's name lines up with how its box would be written for training.
"""
import os
import re


def clean_video_name(name):
    """A filesystem-safe stem for the source video.

    Accepts a full path, a URL, or an already-bare stem; returns just the name
    with its extension dropped and whitespace/separators collapsed to '_'.
    """
    stem = os.path.splitext(os.path.basename(str(name).rstrip("/\\")))[0]
    stem = re.sub(r"[\s/\\]+", "_", stem.strip())
    return stem or "video"


def yolo_crop_name(video_name, frame_idx, box, frame_w, frame_h, ext=".jpg"):
    """Build ``<video>_<frame:06d>_<cx>-<cy>-<w>-<h><ext>`` for one crop.

    ``box`` is the saved (padded, clamped) region as ``(x1, y1, x2, y2)`` in
    pixels; ``frame_w``/``frame_h`` are the full frame's dimensions, which the box
    is normalised against. Dashes join the four coordinates so the underscores
    stay as the top-level separator between name, frame and coordinates.
    """
    cx, cy, w, h = yolo_coords(box, frame_w, frame_h)
    stem = clean_video_name(video_name)
    return f"{stem}_{int(frame_idx):06d}_{cx:.6f}-{cy:.6f}-{w:.6f}-{h:.6f}{ext}"


def yolo_coords(box, frame_w, frame_h):
    """(x1,y1,x2,y2) pixel box → YOLO (cx, cy, w, h) normalised to the frame.

    The single source of truth for both the filename and the database record, so
    a crop's coordinates in Convex always match the coordinates in its name.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    fw = max(float(frame_w), 1.0)
    fh = max(float(frame_h), 1.0)
    return (
        ((x1 + x2) / 2.0) / fw,
        ((y1 + y2) / 2.0) / fh,
        (x2 - x1) / fw,
        (y2 - y1) / fh,
    )


# <video>_<frame>_<cx>-<cy>-<w>-<h>.<ext>  — the shape yolo_crop_name() produces.
_CROP_RE = re.compile(
    r"^(?P<video>.+)_(?P<frame>\d{6})_"
    r"(?P<cx>[\d.]+)-(?P<cy>[\d.]+)-(?P<w>[\d.]+)-(?P<h>[\d.]+)\.[^.]+$"
)


def parse_crop_name(filename):
    """Pull (video, frame, cx, cy, w, h) back out of a crop filename.

    Returns a dict, or None if the name isn't one we generated — lets the
    annotation logger link an annotated image back to its source crop/frame
    when the name matches, and skip the link cleanly when it doesn't.
    """
    m = _CROP_RE.match(os.path.basename(str(filename)))
    if not m:
        return None
    try:
        return {
            "video": m.group("video"),
            "frame": int(m.group("frame")),
            "cx": float(m.group("cx")), "cy": float(m.group("cy")),
            "w": float(m.group("w")), "h": float(m.group("h")),
        }
    except ValueError:
        return None
