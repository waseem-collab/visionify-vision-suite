#!/usr/bin/env node
// Zero-dependency dev launcher with hot reload.
// Runs the whole suite under Werkzeug's auto-reloader: any edit to the Python
// or the landing template restarts the server, and open pages refresh
// themselves via the injected live-reload poller. No manual restart, ever.
// Run with: npm run dev
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");
const PYTHON = process.env.PYTHON || "python3";
const PORT = process.env.PORT || "8000";

// No URL banner here: run.py may land on a different port if this one is taken,
// so it prints the real address once it knows it.
const child = spawn(PYTHON, [join(ROOT, "run.py")], {
  cwd: ROOT,
  // NO_BROWSER: don't auto-open a tab. PYTHONUNBUFFERED: stream logs immediately.
  env: { ...process.env, NO_BROWSER: "1", PYTHONUNBUFFERED: "1", PORT },
  stdio: "inherit",
});

const shutdown = () => { try { child.kill("SIGTERM"); } catch {} process.exit(0); };
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
child.on("exit", (code) => process.exit(code ?? 0));
