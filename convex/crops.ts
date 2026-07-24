import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { assertSecret } from "./lib/auth";
import { guessCameraName } from "./lib/cameraName";

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
    company: v.optional(v.string()),
    site: v.optional(v.string()),
    camera: v.optional(v.string()),
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

// Upsert many crops in one call (batched by the caller) — used by the label
// importer so a big folder isn't thousands of round-trips.
export const bulkRecord = mutation({
  args: {
    secret: v.string(),
    items: v.array(
      v.object({
        filename: v.string(),
        video: v.string(),
        frame: v.number(),
        cx: v.number(),
        cy: v.number(),
        w: v.number(),
        h: v.number(),
        source: v.string(),
        conf: v.optional(v.number()),
        company: v.optional(v.string()),
        site: v.optional(v.string()),
        camera: v.optional(v.string()),
        savedAt: v.number(),
      })
    ),
  },
  handler: async (ctx, { secret, items }) => {
    assertSecret(secret);
    for (const doc of items) {
      const existing = await ctx.db
        .query("crops")
        .withIndex("by_filename", (q) => q.eq("filename", doc.filename))
        .unique();
      if (existing) await ctx.db.patch(existing._id, doc);
      else await ctx.db.insert("crops", doc);
    }
    return { count: items.length };
  },
});

// Heatmap points for the given filters: every matching crop's (cx, cy).
// company/site/camera narrow by the crop's camera tag; className keeps only
// crops whose saved annotation includes that class. Empty filters = everything.
export const heatmap = query({
  args: {
    company: v.optional(v.string()),
    site: v.optional(v.string()),
    camera: v.optional(v.string()),
    className: v.optional(v.string()),
  },
  handler: async (ctx, { company, site, camera, className }) => {
    let rows = camera
      ? await ctx.db
          .query("crops")
          .withIndex("by_camera", (q) => q.eq("camera", camera))
          .collect()
      : await ctx.db.query("crops").collect();
    if (company) rows = rows.filter((r) => r.company === company);
    if (site) rows = rows.filter((r) => r.site === site);
    if (className) {
      const anns = await ctx.db.query("annotations").collect();
      const withClass = new Set(
        anns
          .filter((a) => a.boxes.some((b) => b.className === className))
          .map((a) => a.crop || a.image)
      );
      rows = rows.filter((r) => withClass.has(r.filename));
    }
    return {
      count: rows.length,
      points: rows.map((r) => ({ cx: r.cx, cy: r.cy })),
    };
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

// Crops that came from a camera NOT in the registry, grouped by the camera name
// guessed from the video. Powers the "new camera detected" prompt.
export const unknownCameras = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("crops").collect();
    const groups: Record<string, { camera: string; count: number; sampleVideo: string }> = {};
    for (const r of rows) {
      if (r.camera) continue; // already tagged to a known camera
      const g = guessCameraName(r.video);
      if (!groups[g]) groups[g] = { camera: g, count: 0, sampleVideo: r.video };
      groups[g].count++;
    }
    return Object.values(groups).sort((a, b) => b.count - a.count);
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
