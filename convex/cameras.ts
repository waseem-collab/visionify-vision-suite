import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { assertSecret } from "./lib/auth";

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
