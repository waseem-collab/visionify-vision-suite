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

## Auth

If your functions require authentication, set `CONVEX_AUTH_TOKEN` in `.env`; the
Python client applies it via `set_auth()` before any call.

## Deploying

`npx convex dev` pushes changes live while you work. For a one-off or CI deploy
use `npx convex deploy` (set `CONVEX_DEPLOY_KEY` in the environment for a
non-interactive push).
