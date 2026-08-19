import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { assertSecret } from "./lib/auth";
import { guessCameraName, resolveCamera } from "./lib/cameraName";

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
    project: v.optional(v.string()),
    task: v.optional(v.string()),
    savedAt: v.number(),
  },
  handler: async (ctx, args) => {
    assertSecret(args.secret);
    const { secret, ...doc } = args;
    // The server is the tagging authority: resolve the camera here so a crop
    // from a client with a stale/old registry cache still tags correctly.
    if (!doc.camera) {
      const cam = resolveCamera(doc.video, await ctx.db.query("cameras").collect());
      if (cam) {
        doc.company = cam.company;
        doc.site = cam.site;
        doc.camera = cam.camera;
      }
    }
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
        project: v.optional(v.string()),
        task: v.optional(v.string()),
        savedAt: v.number(),
      })
    ),
  },
  handler: async (ctx, { secret, items }) => {
    assertSecret(secret);
    const registry = await ctx.db.query("cameras").collect();
    for (const doc of items) {
      // Same server-side tagging as crops:record.
      if (!doc.camera) {
        const cam = resolveCamera(doc.video, registry);
        if (cam) {
          doc.company = cam.company;
          doc.site = cam.site;
          doc.camera = cam.camera;
        }
      }
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
    project: v.optional(v.string()),
    task: v.optional(v.string()),
  },
  handler: async (ctx, { company, site, camera, className, project, task }) => {
    let rows = camera
      ? await ctx.db
          .query("crops")
          .withIndex("by_camera", (q) => q.eq("camera", camera))
          .collect()
      : await ctx.db.query("crops").collect();
    if (company) rows = rows.filter((r) => r.company === company);
    if (site) rows = rows.filter((r) => r.site === site);
    if (project) rows = rows.filter((r) => r.project === project);
    if (task) rows = rows.filter((r) => r.task === task);
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
    const registry = await ctx.db.query("cameras").collect();
    const groups: Record<string, { camera: string; count: number; sampleVideo: string }> = {};
    for (const r of rows) {
      if (r.camera) continue; // already tagged to a known camera
      // An untagged crop whose video matches a REGISTERED camera (by name or
      // alias) is a stale-client stray, not a new camera — never prompt for it.
      if (resolveCamera(r.video, registry)) continue;
      const g = guessCameraName(r.video);
      if (!groups[g]) groups[g] = { camera: g, count: 0, sampleVideo: r.video };
      groups[g].count++;
    }
    return Object.values(groups).sort((a, b) => b.count - a.count);
  },
});

// Delete specific crops by filename, plus any annotations linked to them
// (matching by annotation image or its crop link). Precise cleanup for
// mistaken uploads; names with no matching row are reported back.
export const deleteByFilenames = mutation({
  args: { secret: v.string(), filenames: v.array(v.string()) },
  handler: async (ctx, { secret, filenames }) => {
    assertSecret(secret);
    let crops = 0;
    const notFound: string[] = [];
    for (const filename of filenames) {
      const row = await ctx.db
        .query("crops")
        .withIndex("by_filename", (q) => q.eq("filename", filename))
        .unique();
      if (row) { await ctx.db.delete(row._id); crops++; }
      else notFound.push(filename);
    }
    const wanted = new Set(filenames);
    let annotations = 0;
    for (const a of await ctx.db.query("annotations").collect()) {
      if (wanted.has(a.image) || (a.crop && wanted.has(a.crop))) {
        await ctx.db.delete(a._id);
        annotations++;
      }
    }
    return { crops, annotations, notFound };
  },
});

// Delete every crop tagged to one company, plus annotations linked to those
// crops. The camera registry is untouched — only logged crop data goes.
export const deleteByCompany = mutation({
  args: { secret: v.string(), company: v.string() },
  handler: async (ctx, { secret, company }) => {
    assertSecret(secret);
    const rows = await ctx.db.query("crops").collect();
    const doomed = rows.filter((r) => r.company === company);
    const filenames = new Set(doomed.map((r) => r.filename));
    for (const r of doomed) await ctx.db.delete(r._id);
    let annotations = 0;
    if (filenames.size) {
      for (const a of await ctx.db.query("annotations").collect()) {
        if (filenames.has(a.image) || (a.crop && filenames.has(a.crop))) {
          await ctx.db.delete(a._id);
          annotations++;
        }
      }
    }
    return { crops: doomed.length, annotations };
  },
});

// Delete every untagged crop grouped under one guessed camera name — i.e. the
// rows behind a single "new camera detected" entry — plus any annotations
// linked to those crops. Used to discard junk/test uploads instead of
// registering them.
export const deleteUnknownCamera = mutation({
  args: { secret: v.string(), camera: v.string() },
  handler: async (ctx, { secret, camera }) => {
    assertSecret(secret);
    const rows = await ctx.db.query("crops").collect();
    const doomed = rows.filter((r) => !r.camera && guessCameraName(r.video) === camera);
    const filenames = new Set(doomed.map((r) => r.filename));
    for (const r of doomed) await ctx.db.delete(r._id);
    let annotations = 0;
    if (filenames.size) {
      const anns = await ctx.db.query("annotations").collect();
      for (const a of anns) {
        if ((a.crop && filenames.has(a.crop)) || filenames.has(a.image)) {
          await ctx.db.delete(a._id);
          annotations++;
        }
      }
    }
    return { crops: doomed.length, annotations };
  },
});

// Mirror-delete for a CVAT re-sync: rows stamped with this project/task whose
// image is no longer part of the task's export get removed from both tables —
// so deleting an annotation in CVAT deletes it here on the next sync.
export const pruneCvatTask = mutation({
  args: {
    secret: v.string(),
    project: v.string(),
    task: v.string(),
    keep: v.array(v.string()), // image filenames still present in the export
  },
  handler: async (ctx, { secret, project, task, keep }) => {
    assertSecret(secret);
    const keepSet = new Set(keep);
    let crops = 0;
    for (const r of await ctx.db.query("crops").collect()) {
      if (r.project === project && r.task === task && !keepSet.has(r.filename)) {
        await ctx.db.delete(r._id);
        crops++;
      }
    }
    let annotations = 0;
    for (const a of await ctx.db.query("annotations").collect()) {
      if (a.project === project && a.task === task && !keepSet.has(a.image)) {
        await ctx.db.delete(a._id);
        annotations++;
      }
    }
    return { crops, annotations };
  },
});

// Distinct cameras among the crops of one CVAT project/task — powers the
// related-cameras dropdown in the heatmap's CVAT filter mode.
export const camerasFor = query({
  args: { project: v.optional(v.string()), task: v.optional(v.string()) },
  handler: async (ctx, { project, task }) => {
    const rows = await ctx.db.query("crops").collect();
    const out = new Set<string>();
    for (const r of rows) {
      if (project && r.project !== project) continue;
      if (task && r.task !== task) continue;
      if (r.camera) out.add(r.camera);
    }
    return [...out].sort();
  },
});

// The distinct CVAT (project, task) pairs seen across crops — populates the
// heatmap's project/task filters (task options cascade from the project).
export const taskProjects = query({
  args: {},
  handler: async (ctx) => {
    const rows = await ctx.db.query("crops").collect();
    const seen = new Set<string>();
    const pairs: { project: string; task: string }[] = [];
    for (const r of rows) {
      if (!r.project && !r.task) continue;
      const key = `${r.project ?? ""}\u0000${r.task ?? ""}`;
      if (seen.has(key)) continue;
      seen.add(key);
      pairs.push({ project: r.project ?? "", task: r.task ?? "" });
    }
    return pairs.sort((a, b) => a.project.localeCompare(b.project) || a.task.localeCompare(b.task));
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
