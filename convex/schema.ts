import { defineSchema } from "convex/server";

// Intentionally empty — no tables defined yet.
//
// Add tables here when you're ready to model data, e.g.:
//
//   import { defineSchema, defineTable } from "convex/server";
//   import { v } from "convex/values";
//
//   export default defineSchema({
//     crops: defineTable({
//       video: v.string(),
//       frame: v.number(),
//       box: v.array(v.number()),   // [cx, cy, w, h] in YOLO format
//     }),
//   });
//
// Until then this defines the schema as having no tables, which creates nothing
// on the deployment.
export default defineSchema({});
