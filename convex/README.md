# Convex functions

This folder holds the Convex backend for the Vision Suite — the queries and
mutations the Python app calls through `convex_client.py`.

Right now it is an **empty scaffold**: `schema.ts` defines no tables and there
are no functions yet, so nothing exists on the deployment.

## Getting started (when you're ready to add data)

1. Install the npm dependency (declared in the repo's `package.json`):

   ```bash
   npm install
   ```

2. Link this folder to your deployment and start the dev sync. This logs you in
   and generates `convex/_generated/` (the typed API):

   ```bash
   npx convex dev
   ```

   It writes `CONVEX_DEPLOYMENT` / `CONVEX_URL` into `.env.local` for the CLI.
   The Python app reads its own `CONVEX_URL` from `.env` (see `.env.example`).

3. Add tables to `schema.ts` and write functions here, e.g. `crops.ts`:

   ```ts
   import { query } from "./_generated/server";
   export const list = query({ handler: async (ctx) => ctx.db.query("crops").collect() });
   ```

4. Call them from Python:

   ```python
   from convex_client import get_client
   get_client().query("crops:list", {})
   ```

## Auth — server identity (not Clerk)

This suite is a trusted backend tool, so it does **not** use Clerk end-user auth.
Clerk logs users in from a browser and mints short-lived per-user JWTs, which
don't map onto a serverless Python app. Instead, when we add functions they
should authorize the server directly, one of:

- **Shared secret argument** — the app passes a secret (e.g. `CONVEX_SHARED_SECRET`
  from `.env`) as a function argument, and each function checks it before acting.
  Simple and explicit.
- **Convex admin key** — the Python client calls `set_admin_auth(deploy_key)` for
  full-access server calls. Powerful (bypasses function-level checks), so reserve
  it for trusted server contexts and keep `CONVEX_DEPLOY_KEY` secret.

Either way there are no expiring tokens to refresh. `CONVEX_AUTH_TOKEN`
(`set_auth`) remains available for a future token-based caller but is unused
today.

## Deploying

`npx convex dev` pushes changes live while you work. For a one-off or CI deploy
use `npx convex deploy` (set `CONVEX_DEPLOY_KEY` in the environment for a
non-interactive push).
