# Vision Suite

Three computer-vision workspaces served as **one app on one port**, sharing one
model pool, one `.env` and one light/dark theme.

| Workspace | URL | What it does |
|---|---|---|
| **Annotation Studio** | `/annotate/` | YOLO box editor with a radial class picker, CVAT import/upload, auto-annotation, model validation and head-to-head model comparison |
| **Inference Web App** | `/webapp/` | Live person + PPE + SM inference over a local video or a stream URL, per-person PPE status, crop capture, AI crop analysis, annotated-video export |
| **Crop Tools** | `/crop/` | Spatially balanced person-crop extraction, teacher/student disagreement mining, and dataset review & cleanup |

This repo is the merge of two previously separate apps (`annotation_app` and
`inference_web_app`). Every feature from both is here; nothing was dropped.

---

## Quick start

```bash
pip install -r requirements.txt     # or: npm run setup
cp .env.example .env                # fill in CVAT + Azure credentials
npm run dev                         # or: python3 run.py
```

Then open <http://localhost:8000>.

`npm run dev` runs under Werkzeug's auto-reloader: save any `.py` or the landing
template and the server restarts, and every open page reloads itself. Set
`NO_RELOAD=1` to turn that off, `PORT=…` to move the port, `NO_BROWSER=1` to stop
it opening a tab.

Each tool also still runs standalone if you want to work on one in isolation:

```bash
python3 apps/annotation_app.py --dev
python3 apps/web_app.py
python3 apps/crop_balancer_app.py
```

---

## Layout

```
run.py                  entry point — mounts the three apps on one port
paths.py                every directory and state file, in one place
theme.py                the design system: palette, chrome, theme toggle
templates/landing.html  the app picker at /

apps/
  annotation_app.py     Annotation Studio      (mounted at /annotate)
  web_app.py            Inference Web App      (mounted at /webapp)
  crop_balancer_app.py  Crop Tools             (mounted at /crop)

PPE/ppe_inference.py    PPE detection helpers used by the web app
SM/sm_cropper.py        SM cropper helpers used by the web app

models/                 shared model pool — every tool lists from here
  _uploaded/            models dropped into the validation UI land here
sample_dataset/         16 images + labels, so the editor opens with something
imports/                CVAT tasks, unpacked
validation/             ground truth pulled from CVAT for validate/compare
crops/ exports/         inference web app output
person_crops/ crop_reports/ disagreement_frames/   crop tools output
.state/                 caches, run history, remembered session (gitignored)
```

### How the three apps share one port

Each tool is a full Flask app with its own routes, threads and background jobs.
`run.py` composes them with Werkzeug's `DispatcherMiddleware` — the standard
Flask "application dispatching" pattern — so each keeps its own state but lives
under a URL prefix.

The one wrinkle: all three emit *root-absolute* URLs in their HTML and JS
(`fetch("/api/status")`, `<img src="/stream.mjpg">`, `href="/balancer"`), which
would resolve against the site root and miss the mounted app. A narrow
`after_request` hook in `run.py` rewrites the known prefixes to include the mount
point, using `request.script_root`. If you add a new root-absolute URL to a page,
add its prefix to `URL_LITERAL_PREFIXES`.

---

## Theming

`theme.py` is the single source of truth. It defines the palette twice — dark and
light — as CSS custom properties, switched by `data-theme` on `<html>` and
remembered in `localStorage`. Because all three tools now share an origin, one
key keeps the theme in sync as you move between them.

The inference apps were originally built against a different set of variable
names (`--panel`, `--hivis`, …). Rather than rewrite thousands of CSS rules,
`LEGACY_ALIASES` re-expresses those names in terms of the tokens, so their
existing CSS adopts the suite palette — including light mode — untouched.

To restyle everything, edit `TOKENS` in `theme.py`. Nothing else should hardcode
a colour.

---

## Configuration

`.env` at the repo root (see `.env.example`):

| Key | Used by | For |
|---|---|---|
| `CVAT_URL`, `CVAT_USERNAME`, `CVAT_PASSWORD`, `CVAT_ORG_SLUG` | Annotation Studio | importing tasks, uploading annotations, class counts, validation ground truth |
| `AZURE_FOUNDRY_API_ENDPOINT`, `AZURE_FOUNDRY_API_TOKEN`, `AZURE_FOUNDRY_DEPLOYMENT`, `AZURE_FOUNDRY_API_VERSION` | Inference Web App | the "AI analyze" button on a person crop |

Directory overrides (optional): `MODELS_DIR`, `IMPORTS_DIR`, `VAL_DIR`.

Remembered form inputs live in `*_settings.json` at the root and in `.state/` —
all gitignored, all safe to delete (the apps recreate them with defaults).

---

## Adding a feature

- **A new colour or a restyle** → `theme.py`.
- **A new directory or state file** → declare it in `paths.py`, then use it.
- **A new tool** → add `apps/your_app.py` exposing a Flask `app`, mount it in
  `run.py`'s `build_application()`, and add a card to `templates/landing.html`.
- **Inside an existing tool** → the app modules are self-contained; edit in place.
  If you add a root-absolute URL, register its prefix in `URL_LITERAL_PREFIXES`.
