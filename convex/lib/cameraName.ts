// Guess a camera name from a video stem by stripping a trailing date/time, e.g.
//   RMS-Corridor-4A_20260722-052339 -> RMS-Corridor-4A
//   Dyecoats_20260706_142252        -> Dyecoats
//   Baging-Area4_20260724           -> Baging-Area4
// Used to group crops from an unknown camera (many videos, one camera) so the
// "new camera" prompt shows one entry per camera, not one per video file.
// Kept in sync with the Python version in cameras.py.
export function guessCameraName(videoStem: string): string {
  const stem = videoStem || "";
  const stripped = stem.replace(/_\d{6,8}([_-]\d{2,6})*$/, "");
  return stripped || stem;
}
