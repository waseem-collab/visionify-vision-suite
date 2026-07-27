# Convex functions

The Convex backend for the Vision Suite — where crops and annotations are logged
so you can later build a heatmap of where people appear.

**Metadata only.** No image bytes are ever stored: just filenames, video, frame
numbers, YOLO coordinates and annotation labels. A crop row is ~0.4 KB, so even
100k crops is ~40 MB — comfortably inside the free tier. The images stay on disk.

## Data model (`schema.ts`)

- **crops** — one row per crop that has been ANNOTATED. Saving a crop from any
  tool writes nothing; the row is created by the annotation event (`source:
  "annotation"`) or a bulk label import (`"label_import"`). `filename` (unique),
  `video`, `frame`, YOLO `cx/cy/w/h`, `savedAt`. The `cx/cy` are what a heatmap
  plots.
- **annotations** — one row per saved annotation. `image` (unique), the drawn
  `boxes` (YOLO cls/cx/cy/w/h + className), and — when the image is a crop we
  made — `crop`/`video`/`frame` linking it back to the source crop and frame.

Both upsert by their unique key, so re-running a video or re-saving an image
updates in place instead of duplicating.

## Functions

- `crops:record` (mutation) — log a crop. Called by the app.
- `crops:forHeatmap` (query) — `{cx,cy,w,h,video,frame}` for every crop, optionally
  filtered to one `video`. This is what a heatmap reads.
- `crops:videos` (query) — distinct videos with crop counts (for a picker).
- `annotations:record` (mutation) — log an annotation.
- `annotations:recent` (query) — latest annotations, for a sanity view.

Writes are gated by a shared secret (see Auth below).

## Activate it (one-time)

From the repo root:

```bash
npm install                                   # gets the convex CLI + npm deps
npx convex dev                                # log in, link the deployment, deploy these functions
npx convex env set CONVEX_SHARED_SECRET <the value from your .env>
```

`npx convex dev` generates `convex/_generated/` (the typed API) and pushes the
functions live. Keep it running while developing, or use `npx convex deploy` for
a one-off push. The app reads `CONVEX_URL` and `CONVEX_SHARED_SECRET` from `.env`;
the deployment needs the **same** secret set via `npx convex env set`.

Once deployed, every crop and annotation the tools save is logged automatically.
Verify with `GET /api/convex/status` (shows `logging.sent` climbing) or in the
Convex dashboard's Data tab.

## Auth — server identity (not Clerk)

This suite is a trusted backend, so it does **not** use Clerk end-user auth
(short-lived per-user browser JWTs don't fit a serverless Python app). Every
write carries `CONVEX_SHARED_SECRET`, which `convex/lib/auth.ts` checks against
the same value set on the deployment. No expiring tokens. `CONVEX_AUTH_TOKEN`
(`set_auth`) stays available for a future token-based caller but is unused.

## Building the heatmap (later)

`crops:forHeatmap` gives you every `(cx, cy)` — bin them into a grid and count to
get a density map, per video or across all. The suite already has heatmap drawing
in the Crop Balancer; this just makes it queryable from the accumulated database
instead of a single run.
