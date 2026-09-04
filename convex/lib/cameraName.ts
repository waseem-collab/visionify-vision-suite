// Guess a camera name from a video stem by stripping a trailing date/time, e.g.
//   RMS-Corridor-4A_20260722-052339 -> RMS-Corridor-4A
//   Dyecoats_20260706_142252        -> Dyecoats
//   Baging-Area4_20260724           -> Baging-Area4
// Used to group crops from an unknown camera (many videos, one camera) so the
// "new camera" prompt shows one entry per camera, not one per video file.
// Kept in sync with the Python version in cameras.py.
export function guessCameraName(videoStem: string): string {
  const stem = videoStem || "";
  const stripped = stem.replace(/_(?:\d{6,8}(?:[_-]\d{2,6})*|\d{4}(?:-\d{2}){2}(?:[-_]\d{2}){0,3})$/, "");
  return stripped || stem;
}

// True when a video stem belongs to `name`: equal, or `name` is a prefix
// followed by a separator (mirrors cameras.py resolve()).
export function matchesPrefix(stem: string, name: string): boolean {
  if (!name) return false;
  if (stem === name) return true;
  if (!stem.startsWith(name)) return false;
  return ["_", "-", ".", " "].includes(stem[name.length]);
}

// Resolve a video stem against the registry rows (camera name, alias, and all
// merged aliases). Longest match wins so the most specific camera claims the
// video. Returns the matching row or null.
type CameraRow = {
  camera: string;
  company: string;
  site: string;
  alias?: string;
  aliases?: string[];
};
export function resolveCamera<T extends CameraRow>(stem: string, rows: T[]): T | null {
  let best: T | null = null;
  let bestLen = -1;
  for (const row of rows) {
    const names = [row.camera, row.alias ?? "", ...(row.aliases ?? [])];
    for (const name of names) {
      if (name.length > bestLen && matchesPrefix(stem, name)) {
        best = row;
        bestLen = name.length;
      }
    }
  }
  return best;
}
