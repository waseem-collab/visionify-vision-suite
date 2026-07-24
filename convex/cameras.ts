import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { assertSecret } from "./lib/auth";
import { guessCameraName } from "./lib/cameraName";

// Seed (replace) the whole camera registry from cams-data.csv. Idempotent: it
// clears the table and inserts the given list, so re-running just refreshes it.
export const seed = mutation({
  args: {
    secret: v.string(),
    rows: v.array(
      v.object({
        company: v.string(),
        site: v.string(),
        camera: v.string(),
      })
    ),
  },
  handler: async (ctx, { secret, rows }) => {
    assertSecret(secret);
    const existing = await ctx.db.query("cameras").collect();
    for (const doc of existing) await ctx.db.delete(doc._id);
    for (const row of rows) await ctx.db.insert("cameras", row);
    return { inserted: rows.length };
  },
});

// Register one newly-discovered camera and back-fill the crops it already
// produced. `camera` is the name guessed from the video (so future videos from
// it auto-tag); the user supplies which company/site it belongs to.
export const register = mutation({
  args: {
    secret: v.string(),
    company: v.string(),
    site: v.string(),
    camera: v.string(),
  },
  handler: async (ctx, { secret, company, site, camera }) => {
    assertSecret(secret);
    const existing = await ctx.db
      .query("cameras")
      .withIndex("by_camera", (q) => q.eq("camera", camera))
      .unique();
    if (!existing) await ctx.db.insert("cameras", { company, site, camera });
    // Tag every untagged crop whose video maps to this camera.
    const rows = await ctx.db.query("crops").collect();
    let tagged = 0;
    for (const r of rows) {
      if (r.camera) continue;
      if (guessCameraName(r.video) === camera) {
        await ctx.db.patch(r._id, { company, site, camera });
        tagged++;
      }
    }
    return { tagged };
  },
});

// The full registry (small — a few hundred rows). The filter UI builds the
// company → site → camera cascade from this client-side.
export const all = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("cameras").collect();
    return rows
      .map((r) => ({ company: r.company, site: r.site, camera: r.camera }))
      .sort(
        (a, b) =>
          a.company.localeCompare(b.company) ||
          a.site.localeCompare(b.site) ||
          a.camera.localeCompare(b.camera)
      );
  },
});
