# Putting the suite online (without moving models/videos to the cloud)

The suite is a heavy, stateful, local tool — it can't run on a serverless host
like Vercel (torch + models are ~2.3 GB, far over the 250 MB function limit, and
it needs a long-running process, an open video stream, background jobs, and your
local video files). See the note at the bottom.

Instead, keep it running **on your machine** and expose just its web interface
through a tunnel. The models, videos and inference never leave your computer —
the tunnel only forwards HTTPS web requests to `localhost:8000`. Your existing
login gates who can get in, and the tunnel provides the TLS.

## Quick (temporary URL)

Two terminals:

```bash
npm run dev       # 1) the app (or: python3 run.py) — must be on port 8000
npm run share     # 2) the tunnel — prints a https://xxx.trycloudflare.com URL
```

That URL is public but **changes every time** you restart the tunnel, and it's
best for testing/demos. Requires `cloudflared` (install below).

## Install cloudflared (one time)

```bash
curl -fsSL -o ~/.local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x ~/.local/bin/cloudflared
```

(Make sure `~/.local/bin` is on your `PATH`.)

## Permanent URL (your own domain)

For a stable address you control (e.g. `suite.yourdomain.com`), use a *named*
Cloudflare tunnel — free, but needs a Cloudflare account with a domain:

```bash
cloudflared login                                  # opens a browser, pick your domain
cloudflared tunnel create vision-suite             # creates the tunnel + credentials
cloudflared tunnel route dns vision-suite suite.yourdomain.com
cloudflared tunnel run --url http://localhost:8000 vision-suite
```

After that, `suite.yourdomain.com` always points at whatever machine is running
the tunnel. You can run cloudflared as a background service so it survives
reboots (`cloudflared service install`).

## Security notes

- Access is gated by the login you already have; passwords are hashed, sessions
  are signed. Only add users you trust.
- The tunnel gives you HTTPS end-to-end (client → Cloudflare → localhost).
- There's **no brute-force rate limiting** on `/login` yet. With a strong admin
  password that's low risk, but don't share the URL more widely than needed, and
  consider a Cloudflare Access policy on a named tunnel for a second gate.
- Anyone who reaches the URL can *see* the login page. The trycloudflare URL is
  random/unguessable but not secret once shared.

## Why not Vercel / serverless?

| Serverless (Vercel) wants | This app is |
|---|---|
| functions ≤ 250 MB | torch 1.2 GB + models 1.1 GB |
| stateless, per-request | long-running process, in-memory caches, prerender thread |
| short responses | holds an MJPEG stream, runs multi-minute jobs |
| ephemeral read-only FS | reads/writes local folders and your video files |

Only the Convex-backed **Detection Heatmap** could be split out to a serverless
host (it just reads Convex) — the three tools cannot.
