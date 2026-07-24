// Server-identity gate (see ../README.md). The Vision Suite is a trusted
// backend, not a Clerk end user, so writes are authorised by a shared secret it
// sends with each call rather than a per-user JWT.
//
// Set the same value in two places:
//   .env (the app)            CONVEX_SHARED_SECRET=...
//   the deployment env        npx convex env set CONVEX_SHARED_SECRET ...
export function assertSecret(provided: string | undefined): void {
  const expected = process.env.CONVEX_SHARED_SECRET;
  if (!expected) {
    throw new Error(
      "CONVEX_SHARED_SECRET is not set on the deployment — run: npx convex env set CONVEX_SHARED_SECRET <value>"
    );
  }
  if (provided !== expected) {
    throw new Error("Unauthorized: bad or missing secret.");
  }
}
