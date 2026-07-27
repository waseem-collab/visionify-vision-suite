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
        thumbnail: v.optional(v.string()),
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
    // The name guessed from the video, when the user renamed the camera —
    // stored as the alias so this camera keeps matching those videos.
    alias: v.optional(v.string()),
  },
  handler: async (ctx, { secret, company, site, camera, alias }) => {
    assertSecret(secret);
    const match = alias && alias !== camera ? alias : undefined;
    const existing = await ctx.db
      .query("cameras")
      .withIndex("by_camera", (q) => q.eq("camera", camera))
      .unique();
    if (!existing)
      await ctx.db.insert("cameras", { company, site, camera, ...(match ? { alias: match } : {}) });
    // Tag every untagged crop whose video maps to this camera.
    const rows = await ctx.db.query("crops").collect();
    let tagged = 0;
    for (const r of rows) {
      if (r.camera) continue;
      if (guessCameraName(r.video) === (match || camera)) {
        await ctx.db.patch(r._id, { company, site, camera });
        tagged++;
      }
    }
    return { tagged };
  },
});

// Attach thumbnail URLs to registry rows, matched by camera name. Only patches
// cameras that exist; returns the names it couldn't find so a typo is visible.
export const setThumbnails = mutation({
  args: {
    secret: v.string(),
    items: v.array(v.object({ camera: v.string(), thumbnail: v.string() })),
  },
  handler: async (ctx, { secret, items }) => {
    assertSecret(secret);
    let updated = 0;
    const missing: string[] = [];
    for (const it of items) {
      const rows = await ctx.db
        .query("cameras")
        .withIndex("by_camera", (q) => q.eq("camera", it.camera))
        .collect();
      if (!rows.length) { missing.push(it.camera); continue; }
      for (const row of rows) await ctx.db.patch(row._id, { thumbnail: it.thumbnail });
      updated += rows.length;
    }
    return { updated, missing };
  },
});

// The full registry (small — a few hundred rows). The filter UI builds the
// company → site → camera cascade from this client-side.
export const all = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("cameras").collect();
    return rows
      .map((r) => ({ company: r.company, site: r.site, camera: r.camera, thumbnail: r.thumbnail, alias: r.alias }))
      .sort(
        (a, b) =>
          a.company.localeCompare(b.company) ||
          a.site.localeCompare(b.site) ||
          a.camera.localeCompare(b.camera)
      );
  },
});
