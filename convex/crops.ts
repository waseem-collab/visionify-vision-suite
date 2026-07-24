import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { assertSecret } from "./lib/auth";

// Record one saved crop. Upserts by filename, so re-running the same video (the
// crop names are deterministic) updates rather than duplicates.
export const record = mutation({
  args: {
    secret: v.string(),
    filename: v.string(),
    video: v.string(),
    frame: v.number(),
    cx: v.number(),
    cy: v.number(),
    w: v.number(),
    h: v.number(),
    source: v.string(),
    conf: v.optional(v.number()),
    savedAt: v.number(),
  },
  handler: async (ctx, args) => {
    assertSecret(args.secret);
    const { secret, ...doc } = args;
    const existing = await ctx.db
      .query("crops")
      .withIndex("by_filename", (q) => q.eq("filename", doc.filename))
      .unique();
    if (existing) {
      await ctx.db.patch(existing._id, doc);
      return existing._id;
    }
    return await ctx.db.insert("crops", doc);
  },
});

// Coordinates for a heatmap: every crop's centre + size, optionally for one
// video. Reads nothing but the fields the heatmap needs.
export const forHeatmap = query({
  args: { video: v.optional(v.string()) },
  handler: async (ctx, { video }) => {
    const rows = video
      ? await ctx.db
          .query("crops")
          .withIndex("by_video", (q) => q.eq("video", video))
          .collect()
      : await ctx.db.query("crops").collect();
    return rows.map((r) => ({
      video: r.video,
      frame: r.frame,
      cx: r.cx,
      cy: r.cy,
      w: r.w,
      h: r.h,
    }));
  },
});

// The distinct videos that have crops, with a count each — handy for a picker.
export const videos = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("crops").collect();
    const counts: Record<string, number> = {};
    for (const r of rows) counts[r.video] = (counts[r.video] ?? 0) + 1;
    return Object.entries(counts)
      .map(([video, count]) => ({ video, count }))
      .sort((a, b) => b.count - a.count);
  },
});
