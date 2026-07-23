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

# Repo root — the directory holding run.py, models/, imports/, PPE/, SM/ …
ROOT = Path(__file__).resolve().parent

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
IMPORTS_DIR = Path(os.getenv("IMPORTS_DIR") or (ROOT / "imports"))
# Ground-truth pulled from CVAT for model validation / comparison runs.
VAL_DIR = Path(os.getenv("VAL_DIR") or (ROOT / "validation"))

# --------------------------------------------------------------------------- #
# Inference web app
# --------------------------------------------------------------------------- #
CROPS_ROOT = ROOT / "crops"              # person crops saved from the live stream
EXPORTS_ROOT = ROOT / "exports"          # annotated videos rendered by "export"

# --------------------------------------------------------------------------- #
# Crop tools
# --------------------------------------------------------------------------- #
PERSON_CROPS_ROOT = ROOT / "person_crops"        # balanced crop batches
REPORTS_ROOT = ROOT / "crop_reports"             # heatmaps + manifests
DISAGREE_ROOT = ROOT / "disagreement_frames"     # teacher/student mining output

# --------------------------------------------------------------------------- #
# Per-user state (all gitignored — remembered form inputs, caches, history)
# --------------------------------------------------------------------------- #
STATE_DIR = ROOT / ".state"

APP_SETTINGS_FILE = ROOT / "web_app_settings.json"
CROP_SETTINGS_FILE = ROOT / "crop_balancer_settings.json"
DISAGREE_SETTINGS_FILE = ROOT / "disagreement_settings.json"
REVIEW_SETTINGS_FILE = ROOT / "review_settings.json"
PPE_PROMPTS_FILE = ROOT / "ppe_prompts.json"

ANNOTATION_STATE_FILE = STATE_DIR / "annotation_app_state.json"
CVAT_CACHE_FILE = STATE_DIR / "cvat_cache.json"
VAL_HIST_FILE = STATE_DIR / "val_history.json"
VAL_LAST_FILE = STATE_DIR / "val_last.json"
CMP_HIST_FILE = STATE_DIR / "cmp_history.json"

# Directories every tool expects to exist before it writes its first file.
RUNTIME_DIRS = (
    MODELS_DIR, UPLOAD_MODELS_DIR, IMPORTS_DIR, VAL_DIR,
    CROPS_ROOT, EXPORTS_ROOT,
    PERSON_CROPS_ROOT, REPORTS_ROOT, DISAGREE_ROOT,
    STATE_DIR,
)


def ensure_dirs():
    """Create every runtime directory. Idempotent; called once at startup."""
    for d in RUNTIME_DIRS:
        d.mkdir(parents=True, exist_ok=True)
