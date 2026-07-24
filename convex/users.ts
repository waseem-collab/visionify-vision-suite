import { mutation, query } from "./_generated/server";
import { v } from "convex/values";
import { assertSecret } from "./lib/auth";

// The login allowlist — emails permitted to sign in with Google. All three are
// gated by the shared secret, since the auth backend (which holds it) is the
// only caller. The admin email is never here; it's configured in .env.

export const list = query({
  args: { secret: v.string() },
  handler: async (ctx, { secret }) => {
    assertSecret(secret);
    const rows = await ctx.db.query("users").collect();
    return rows.map((r) => r.email);
  },
});

export const add = mutation({
  args: { secret: v.string(), email: v.string(), addedBy: v.optional(v.string()) },
  handler: async (ctx, { secret, email, addedBy }) => {
    assertSecret(secret);
    const e = email.trim().toLowerCase();
    const existing = await ctx.db
      .query("users")
      .withIndex("by_email", (q) => q.eq("email", e))
      .unique();
    if (existing) return existing._id; // already allowed — no-op
    return await ctx.db.insert("users", { email: e, addedBy: addedBy || "", addedAt: Date.now() });
  },
});

export const remove = mutation({
  args: { secret: v.string(), email: v.string() },
  handler: async (ctx, { secret, email }) => {
    assertSecret(secret);
    const e = email.trim().toLowerCase();
    const existing = await ctx.db
      .query("users")
      .withIndex("by_email", (q) => q.eq("email", e))
      .unique();
    if (existing) await ctx.db.delete(existing._id);
    return { removed: existing ? 1 : 0 };
  },
});
