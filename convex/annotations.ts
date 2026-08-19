import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { assertSecret } from "./lib/auth";

// Record one saved annotation (the YOLO boxes drawn on an image). Upserts by
// image filename, so re-saving the same image overwrites its boxes rather than
// piling up rows. When the image is a crop we produced, `crop`/`video`/`frame`
// link it back to the source crop and frame.
export const record = mutation({
  args: {
    secret: v.string(),
    image: v.string(),
    crop: v.optional(v.string()),
    video: v.optional(v.string()),
    frame: v.optional(v.number()),
    boxes: v.array(
      v.object({
        cls: v.number(),
        cx: v.number(),
        cy: v.number(),
        w: v.number(),
        h: v.number(),
        className: v.optional(v.string()),
      })
    ),
    project: v.optional(v.string()),
    task: v.optional(v.string()),
    savedAt: v.number(),
  },
  handler: async (ctx, args) => {
    assertSecret(args.secret);
    const { secret, ...doc } = args;
    const existing = await ctx.db
      .query("annotations")
      .withIndex("by_image", (q) => q.eq("image", doc.image))
      .unique();
    if (existing) {
      await ctx.db.patch(existing._id, doc);
      return existing._id;
    }
    return await ctx.db.insert("annotations", doc);
  },
});

// Upsert many annotations in one call (batched by the caller) — the bulk path
// for the label-folder importer.
export const bulkRecord = mutation({
  args: {
    secret: v.string(),
    items: v.array(
      v.object({
        image: v.string(),
        crop: v.optional(v.string()),
        video: v.optional(v.string()),
        frame: v.optional(v.number()),
        boxes: v.array(
          v.object({
            cls: v.number(),
            cx: v.number(),
            cy: v.number(),
            w: v.number(),
            h: v.number(),
            className: v.optional(v.string()),
          })
        ),
        project: v.optional(v.string()),
        task: v.optional(v.string()),
        savedAt: v.number(),
      })
    ),
  },
  handler: async (ctx, { secret, items }) => {
    assertSecret(secret);
    for (const doc of items) {
      const existing = await ctx.db
        .query("annotations")
        .withIndex("by_image", (q) => q.eq("image", doc.image))
        .unique();
      if (existing) await ctx.db.patch(existing._id, doc);
      else await ctx.db.insert("annotations", doc);
    }
    return { count: items.length };
  },
});

// Class labels present in the annotations matching the given filters — the
// heatmap's class dropdown only offers what's actually there. company/site/
// camera scope via the linked crops; project/task via the annotation's stamp.
export const classesFor = query({
  args: {
    company: v.optional(v.string()),
    site: v.optional(v.string()),
    camera: v.optional(v.string()),
    project: v.optional(v.string()),
    task: v.optional(v.string()),
  },
  handler: async (ctx, { company, site, camera, project, task }) => {
    let anns = await ctx.db.query("annotations").collect();
    if (project) anns = anns.filter((a) => a.project === project);
    if (task) anns = anns.filter((a) => a.task === task);
    if (company || site || camera) {
      const crops = await ctx.db.query("crops").collect();
      const ok = new Set(
        crops
          .filter((r) =>
            (!company || r.company === company) &&
            (!site || r.site === site) &&
            (!camera || r.camera === camera))
          .map((r) => r.filename)
      );
      anns = anns.filter((a) => ok.has(a.crop || a.image));
    }
    const names = new Set<string>();
    for (const a of anns) {
      for (const b of a.boxes) if (b.className) names.add(b.className);
    }
    return Array.from(names).sort();
  },
});

// The distinct class labels seen across all annotations — populates the
// heatmap's class filter.
export const classes = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("annotations").collect();
    const names = new Set<string>();
    for (const a of rows) {
      for (const b of a.boxes) if (b.className) names.add(b.className);
    }
    return Array.from(names).sort();
  },
});

// Recent annotations (most recent first) — for a quick sanity view.
export const recent = query({
  args: { limit: v.optional(v.number()) },
  handler: async (ctx, { limit }) => {
    const rows = await ctx.db.query("annotations").order("desc").take(limit ?? 50);
    return rows.map((r) => ({
      image: r.image,
      video: r.video,
      frame: r.frame,
      boxCount: r.boxes.length,
      savedAt: r.savedAt,
    }));
  },
});
