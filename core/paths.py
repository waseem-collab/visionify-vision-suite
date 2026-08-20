#!/usr/bin/env python3
"""
Shared filesystem layout for the Vision Suite.

The three tools (annotation studio, inference web app, crop tools) each used to
live in their own repo and resolve everything relative to their own file. Now
they sit in ``apps/`` and share one set of directories at the repo root, so all
of them are declared here — one place to look, one place to change.

Nothing in here does any work beyond creating the directories, so it is safe to
import from anywhere (including the reloader's watcher process).
"""
import os
from pathlib import Path

# Repo root — the directory holding run.py, models/, data/, PPE/, SM/ …
ROOT = Path(__file__).resolve().parent.parent  # this file lives in core/
# All generated/runtime output lands under one folder, so the root stays clean.
DATA_DIR = Path(os.getenv("DATA_DIR") or (ROOT / "data"))

# --------------------------------------------------------------------------- #
# Shared assets
# --------------------------------------------------------------------------- #
# One model pool for every tool: the annotation studio's auto-annotate and model
# validation, the web app's person/PPE/SM models, and the crop balancer all list
# from here. Override with MODELS_DIR to point at a shared network drive.
MODELS_DIR = Path(os.getenv("MODELS_DIR") or (ROOT / "models"))
# Models uploaded through the validation UI land in a subfolder so they never
# get mixed up with the curated ones.
UPLOAD_MODELS_DIR = MODELS_DIR / "_uploaded"

# --------------------------------------------------------------------------- #
# Annotation studio
# --------------------------------------------------------------------------- #
# CVAT tasks are exported and unpacked here (one folder per task).
IMPORTS_DIR = Path(os.getenv("IMPORTS_DIR") or (DATA_DIR / "imports"))
# Ground-truth pulled from CVAT for model validation / comparison runs.
VAL_DIR = Path(os.getenv("VAL_DIR") or (DATA_DIR / "validation"))

# --------------------------------------------------------------------------- #
# Inference web app
# --------------------------------------------------------------------------- #
CROPS_ROOT = DATA_DIR / "crops"              # person crops saved from the live stream
EXPORTS_ROOT = DATA_DIR / "exports"          # annotated videos rendered by "export"

# --------------------------------------------------------------------------- #
# Crop tools
# --------------------------------------------------------------------------- #
PERSON_CROPS_ROOT = DATA_DIR / "person_crops"        # balanced crop batches
FRAMES_ROOT = DATA_DIR / "extracted_frames"      # Frame Extractor output (per batch)
CVAT_DL_ROOT = DATA_DIR / "cvat_downloads"       # CVAT Download card's packaged zips
REPORTS_ROOT = DATA_DIR / "crop_reports"             # heatmaps + manifests
DISAGREE_ROOT = DATA_DIR / "disagreement_frames"     # teacher/student mining output

# --------------------------------------------------------------------------- #
# Per-user state (all gitignored — remembered form inputs, caches, history)
# --------------------------------------------------------------------------- #
STATE_DIR = ROOT / ".state"

APP_SETTINGS_FILE = STATE_DIR / "web_app_settings.json"
CROP_SETTINGS_FILE = STATE_DIR / "crop_balancer_settings.json"
DISAGREE_SETTINGS_FILE = STATE_DIR / "disagreement_settings.json"
REVIEW_SETTINGS_FILE = STATE_DIR / "review_settings.json"
PPE_PROMPTS_FILE = ROOT / "PPE" / "ppe_prompts.json"

ANNOTATION_STATE_FILE = STATE_DIR / "annotation_app_state.json"
CVAT_CACHE_FILE = STATE_DIR / "cvat_cache.json"
VAL_HIST_FILE = STATE_DIR / "val_history.json"
VAL_LAST_FILE = STATE_DIR / "val_last.json"
CMP_HIST_FILE = STATE_DIR / "cmp_history.json"

# Directories every tool expects to exist before it writes its first file.
RUNTIME_DIRS = (
    MODELS_DIR, UPLOAD_MODELS_DIR, IMPORTS_DIR, VAL_DIR,
    CROPS_ROOT, EXPORTS_ROOT,
    PERSON_CROPS_ROOT, REPORTS_ROOT, DISAGREE_ROOT, FRAMES_ROOT, CVAT_DL_ROOT,
    STATE_DIR,
)


def ensure_dirs():
    """Create every runtime directory. Idempotent; called once at startup."""
    for d in RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)
