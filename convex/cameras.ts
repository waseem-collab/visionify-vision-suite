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
    if (!existing) {
      await ctx.db.insert("cameras", { company, site, camera, ...(match ? { alias: match } : {}) });
    } else if (match && existing.alias !== match && !(existing.aliases ?? []).includes(match)) {
      // Matching a discovered name onto an already-registered camera: the
      // guessed name joins its aliases so those videos keep tagging here.
      await ctx.db.patch(existing._id, { aliases: [...(existing.aliases ?? []), match] });
    }
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

// Edit one registry row, matched by its current camera name. Only the provided
// fields change; an empty thumbnail/alias string clears that field. A rename or
// company/site change propagates to the camera's crops (they denormalise those
// fields), and a rename auto-sets the alias to the old name so the camera keeps
// matching its videos. `mergeFrom` folds other cameras into this one: their
// registry rows are deleted, their crops retagged here, and their names (and
// their own aliases) join this camera's `aliases` so their videos keep
// matching.
export const update = mutation({
  args: {
    secret: v.string(),
    camera: v.string(),
    newName: v.optional(v.string()),
    company: v.optional(v.string()),
    site: v.optional(v.string()),
    thumbnail: v.optional(v.string()),
    alias: v.optional(v.string()),
    aliases: v.optional(v.array(v.string())), // full replacement of the alias list
    mergeFrom: v.optional(v.array(v.string())),
  },
  handler: async (ctx, { secret, camera, newName, company, site, thumbnail, alias, aliases, mergeFrom }) => {
    assertSecret(secret);
    const row = await ctx.db
      .query("cameras")
      .withIndex("by_camera", (q) => q.eq("camera", camera))
      .unique();
    if (!row) throw new Error(`No camera named "${camera}"`);
    const rename = newName && newName !== camera ? newName : undefined;
    if (rename) {
      const clash = await ctx.db
        .query("cameras")
        .withIndex("by_camera", (q) => q.eq("camera", rename))
        .unique();
      if (clash) throw new Error(`A camera named "${rename}" already exists`);
    }
    const finalName = rename || camera;
    const finalCompany = company ?? row.company;
    const finalSite = site ?? row.site;
    await ctx.db.patch(row._id, {
      ...(rename ? { camera: rename } : {}),
      ...(rename && !row.alias && alias === undefined ? { alias: camera } : {}),
      ...(company !== undefined ? { company } : {}),
      ...(site !== undefined ? { site } : {}),
      ...(thumbnail !== undefined ? { thumbnail: thumbnail || undefined } : {}),
      ...(alias !== undefined ? { alias: alias || undefined } : {}),
      ...(aliases !== undefined ? { aliases: aliases.length ? aliases : undefined } : {}),
    });
    // Propagate the change to this camera's crops.
    let retagged = 0;
    if (rename || company !== undefined || site !== undefined) {
      const crops = await ctx.db
        .query("crops")
        .withIndex("by_camera", (q) => q.eq("camera", camera))
        .collect();
      for (const c of crops) {
        await ctx.db.patch(c._id, { camera: finalName, company: finalCompany, site: finalSite });
        retagged++;
      }
    }
    // Merge the listed cameras into this one.
    let merged = 0;
    if (mergeFrom && mergeFrom.length) {
      const aliasSet = new Set<string>(row.aliases ?? []);
      for (const src of mergeFrom) {
        if (src === camera || src === finalName) continue;
        const srow = await ctx.db
          .query("cameras")
          .withIndex("by_camera", (q) => q.eq("camera", src))
          .unique();
        if (!srow) {
          // Not a registry camera — treat the typed name as a raw video prefix:
          // it becomes an alias here and claims any untagged crops it matches.
          // Video stems never contain spaces (they're collapsed to "_" when a
          // crop is named), so normalise typed whitespace the same way.
          const prefix = src.replace(/\s+/g, "_");
          aliasSet.add(prefix);
          const untagged = await ctx.db.query("crops").collect();
          for (const c of untagged) {
            if (c.camera) continue;
            if (guessCameraName(c.video) === prefix) {
              await ctx.db.patch(c._id, { camera: finalName, company: finalCompany, site: finalSite });
              retagged++;
            }
          }
          merged++;
          continue;
        }
        aliasSet.add(srow.camera);
        if (srow.alias) aliasSet.add(srow.alias);
        for (const a of srow.aliases ?? []) aliasSet.add(a);
        await ctx.db.delete(srow._id);
        const crops = await ctx.db
          .query("crops")
          .withIndex("by_camera", (q) => q.eq("camera", src))
          .collect();
        for (const c of crops) {
          await ctx.db.patch(c._id, { camera: finalName, company: finalCompany, site: finalSite });
          retagged++;
        }
        merged++;
      }
      aliasSet.delete(finalName);
      await ctx.db.patch(row._id, { aliases: [...aliasSet] });
    }
    return { ok: true, retagged, merged };
  },
});

// Delete a camera from the registry. Its crops are untagged (not deleted) so
// they reappear under "new cameras" instead of keeping a stale tag.
export const remove = mutation({
  args: { secret: v.string(), camera: v.string() },
  handler: async (ctx, { secret, camera }) => {
    assertSecret(secret);
    const row = await ctx.db
      .query("cameras")
      .withIndex("by_camera", (q) => q.eq("camera", camera))
      .unique();
    if (!row) return { removed: 0, untagged: 0 };
    await ctx.db.delete(row._id);
    let untagged = 0;
    const crops = await ctx.db
      .query("crops")
      .withIndex("by_camera", (q) => q.eq("camera", camera))
      .collect();
    for (const c of crops) {
      await ctx.db.patch(c._id, { camera: undefined, company: undefined, site: undefined });
      untagged++;
    }
    return { removed: 1, untagged };
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
      .map((r) => ({ company: r.company, site: r.site, camera: r.camera, thumbnail: r.thumbnail, alias: r.alias, aliases: r.aliases }))
      .sort(
        (a, b) =>
          a.company.localeCompare(b.company) ||
          a.site.localeCompare(b.site) ||
          a.camera.localeCompare(b.camera)
      );
  },
});
