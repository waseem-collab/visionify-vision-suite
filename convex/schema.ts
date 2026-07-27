import { defineSchema, defineTable } from "convex/server";
import { v } from "convex/values";

// Metadata only — filenames, frames and YOLO coordinates, never image bytes.
// That keeps the whole thing tiny (well inside the free tier) while carrying
// everything a heatmap needs: where people were, per video and frame.
export default defineSchema({
  // Login allowlist — emails allowed to sign in with Google. The admin is NOT
  // stored here (it's configured in .env and always allowed).
  users: defineTable({
    email: v.string(),
    passwordHash: v.optional(v.string()), // set by the admin; hashed in the backend
    addedBy: v.optional(v.string()),
    addedAt: v.number(),
  }).index("by_email", ["email"]),

  // The camera registry, seeded from cams-data.csv. One row per
  // (company, site, camera). Drives the heatmap's cascading filters and lets a
  // crop be tagged to a camera by matching the video name.
  cameras: defineTable({
    company: v.string(),
    site: v.string(),
    camera: v.string(),
    thumbnail: v.optional(v.string()), // hosted image URL of the camera's scene
    // Video-name prefix this camera matches when the user renamed it at
    // registration (e.g. camera "Yard East" with alias "Mayfield" still tags
    // videos named Mayfield_*). Absent when the camera name IS the prefix.
    alias: v.optional(v.string()),
  })
    .index("by_company", ["company"])
    .index("by_company_site", ["company", "site"])
    .index("by_camera", ["camera"]),

  // One row per saved crop, from any tool (inference web app, crop balancer, PPE).
  crops: defineTable({
    filename: v.string(), // e.g. Dyecoats_000120_0.48-0.38-0.20-0.52.jpg (unique key)
    video: v.string(), //    source video stem
    frame: v.number(), //    frame index within that video
    // YOLO box, normalised to the full frame — this is what the heatmap plots.
    cx: v.number(),
    cy: v.number(),
    w: v.number(),
    h: v.number(),
    source: v.string(), //   "inference_webapp" | "crop_balancer" | "ppe"
    conf: v.optional(v.number()), // detector confidence, when known
    // Camera this crop came from, resolved from the video name against the
    // registry. Absent when the video doesn't match a known camera.
    company: v.optional(v.string()),
    site: v.optional(v.string()),
    camera: v.optional(v.string()),
    savedAt: v.number(), //  epoch ms
  })
    .index("by_filename", ["filename"])
    .index("by_video", ["video"])
    .index("by_camera", ["camera"]),

  // One row per saved annotation. Linked to a crop by filename when the annotated
  // image is one we produced (crop === filename); video/frame are denormalised
  // from the crop name so a heatmap can be built from annotations alone too.
  annotations: defineTable({
    image: v.string(), //    the image file that was annotated (unique key)
    crop: v.optional(v.string()), // matching crop filename, if the image is a crop
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
    savedAt: v.number(),
  })
    .index("by_image", ["image"])
    .index("by_crop", ["crop"])
    .index("by_video", ["video"]),
});
