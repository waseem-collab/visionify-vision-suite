#!/usr/bin/env python3
"""
run_ppe_video.py

Usage:
    python3 run_ppe_video.py
    python3 run_ppe_video.py --conf 0.5
    python3 run_ppe_video.py --output output.mp4
    python3 run_ppe_video.py --conf 0.6 --output annotated.mp4
"""

import os
import json
import argparse
import datetime
import cv2
import sys
import time
from ultralytics import YOLO

# Repo root on the path so this stays runnable standalone (python3 PPE/ppe_inference.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import cropnames, paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights come from the suite-wide model pool, shared with every other tool.
DEFAULT_MODEL_PATH = str(
    paths.MODELS_DIR / "pharma-ppe-pt-1.0.0" / "pharma-ppe-pt" / "pharma-ppe-detection.pt"
)
DEFAULT_PERSON_MODEL = str(paths.MODELS_DIR / "yolo26n.pt")
PERSON_CLASS_ID = 0
HISTORY_FILE = str(paths.STATE_DIR / "ppe_saved_video_paths.json")
CENTROID_LOG_FILE = str(paths.STATE_DIR / "ppe_crop_centroids.jsonl")
WINDOW_NAME = "PPE Detection"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
PERSON_CROPS_DIR = str(paths.CROPS_ROOT / "ppe_person_crops")
BACKGROUND_CROPS_DIR = str(paths.CROPS_ROOT / "ppe_background")
PERSON_CROP_PADDING = 10


# ----------------------------- CLI ----------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="PPE Detection Video Inference (YOLO + OpenCV)")
    parser.add_argument("--conf", type=float, default=0.4, help="PPE confidence threshold (default: 0.4)")
    parser.add_argument(
        "--person-conf",
        type=float,
        default=0.4,
        help="Person detection confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--model",
        "--ppe-model",
        dest="ppe_model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to PPE YOLO .pt model",
    )
    parser.add_argument(
        "--person-model",
        type=str,
        default=DEFAULT_PERSON_MODEL,
        help="Path/name for person detector model (default: yolo11m.pt)",
    )
    parser.add_argument("--output", type=str, default="", help="Optional output MP4 path")
    return parser.parse_args()


# ----------------------------- Source History ----------------------------- #
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def list_videos_in_folder(folder_path):
    videos = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if not os.path.isfile(full_path):
            continue
        if name.lower().endswith(VIDEO_EXTENSIONS):
            videos.append(full_path)
    videos.sort()
    return videos


def choose_folder_from_history():
    history = load_history()

    while True:
        print("\nSelect a folder path:")
        for idx, src in enumerate(history):
            print(f"{idx + 1}. {src}")
        print("N. Enter new folder")
        print("C. Clear saved folders")

        choice = input("Choice: ").strip()

        if choice.lower() == "c":
            history = []
            save_history(history)
            print("Saved folder history cleared.")
            continue

        if choice.lower() == "n" or not choice.isdigit():
            new_src = input("Enter local folder path: ").strip()
            new_src = os.path.abspath(os.path.expanduser(new_src))

            if not os.path.isdir(new_src):
                print("Error: Folder does not exist.")
                continue

            if new_src not in history:
                history.append(new_src)
                save_history(history)

            return new_src

        idx = int(choice) - 1
        if 0 <= idx < len(history):
            selected = os.path.abspath(os.path.expanduser(history[idx]))
            if not os.path.isdir(selected):
                print("Saved folder no longer exists. Please choose another folder.")
                continue
            return selected

        print("Invalid selection.")


# ----------------------------- Video Handling ----------------------------- #
def can_open_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return False
    ok, _ = cap.read()
    cap.release()
    return ok


def open_video(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        sys.exit(1)

    return cap


def choose_video_source():
    while True:
        folder = choose_folder_from_history()
        videos = list_videos_in_folder(folder)

        if not videos:
            print(f"No supported video files found in: {folder}")
            print(f"Supported extensions: {', '.join(VIDEO_EXTENSIONS)}")
            continue

        print(f"\nVideos in folder: {folder}")
        for idx, video in enumerate(videos, start=1):
            print(f"{idx}. {os.path.basename(video)}")
        print("B. Back to folder selection")

        choice = input("Select video number: ").strip()
        if choice.lower() == "b":
            continue
        if not choice.isdigit():
            print("Invalid selection.")
            continue

        idx = int(choice) - 1
        if 0 <= idx < len(videos):
            selected_video = videos[idx]
            if not can_open_video(selected_video):
                print(
                    "Selected video cannot be opened (possibly corrupted/incomplete, "
                    "e.g. missing moov atom). Please choose another file."
                )
                continue
            return selected_video
        print("Invalid selection.")


def is_path_like(model_ref):
    return (
        "/" in model_ref
        or "\\" in model_ref
        or model_ref.startswith(".")
        or model_ref.startswith("~")
    )


def load_person_model(model_ref):
    if os.path.exists(model_ref):
        print(f"Using person model: {model_ref}")
        return YOLO(model_ref)

    if is_path_like(model_ref):
        print(f"Person model path not found: {model_ref}")
        print("Tip: pass a model name like 'yolov11m.pt' to auto-download it.")
        sys.exit(1)

    print(f"Person model '{model_ref}' not found locally. Downloading...")
    try:
        model = YOLO(model_ref)
        print(f"Person model ready: {model_ref}")
        return model
    except Exception as exc:
        print(f"Error: Failed to download/load person model '{model_ref}'.")
        print(f"Details: {exc}")
        sys.exit(1)


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def apply_saved_masks(frame, frame_idx, saved_person_masks):
    h, w = frame.shape[:2]
    for box in saved_person_masks.get(frame_idx, []):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            continue
        frame[y1:y2, x1:x2] = 0
    return frame


def run_two_stage_inference(frame, person_model, ppe_model, person_conf, ppe_conf, manual_person_boxes=None):
    h, w = frame.shape[:2]
    annotated = frame.copy()
    person_count = 0
    ppe_count = 0
    person_boxes = []
    manual_person_boxes = manual_person_boxes or []

    def process_person_region(x1, y1, x2, y2, person_label):
        nonlocal person_count, ppe_count
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 <= x1 or y2 <= y1:
            return
        person_count += 1
        person_boxes.append((x1, y1, x2, y2, 1.0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
        cv2.putText(
            annotated,
            person_label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        ppe_results = ppe_model.predict(crop, conf=ppe_conf, verbose=False)[0]
        if ppe_results.boxes is None:
            return

        for bbox in ppe_results.boxes:
            bconf = float(bbox.conf[0])
            bcls = int(bbox.cls[0])
            label = ppe_results.names.get(bcls, str(bcls))
            cx1, cy1, cx2, cy2 = map(int, bbox.xyxy[0])

            gx1, gy1 = x1 + cx1, y1 + cy1
            gx2, gy2 = x1 + cx2, y1 + cy2
            gx1, gy1, gx2, gy2 = clamp_box(gx1, gy1, gx2, gy2, w, h)
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            ppe_count += 1
            cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"{label} {bconf:.2f}",
                (gx1, max(18, gy1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

    person_results = person_model.predict(frame, conf=person_conf, verbose=False)[0]
    if person_results.boxes is None:
        return annotated, person_count, ppe_count

    for pbox in person_results.boxes:
        cls_id = int(pbox.cls[0])
        if cls_id != PERSON_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, pbox.xyxy[0])
        pconf = float(pbox.conf[0])
        process_person_region(x1, y1, x2, y2, f"person {pconf:.2f}")

    for mb in manual_person_boxes:
        mx1, my1, mx2, my2 = map(int, mb[:4])
        process_person_region(mx1, my1, mx2, my2, "person-manual")

    return annotated, person_count, ppe_count, person_boxes


def save_person_crop(frame, frame_idx, person_idx, person_box, output_dir=None,
                     video_name=None):
    if output_dir is None:
        output_dir = PERSON_CROPS_DIR
    h, w = frame.shape[:2]
    if len(person_box) >= 4:
        x1, y1, x2, y2 = map(int, person_box[:4])
    else:
        return None
    x1p = max(0, x1 - PERSON_CROP_PADDING)
    y1p = max(0, y1 - PERSON_CROP_PADDING)
    x2p = min(w - 1, x2 + PERSON_CROP_PADDING)
    y2p = min(h - 1, y2 + PERSON_CROP_PADDING)

    if x2p <= x1p or y2p <= y1p:
        return None

    os.makedirs(output_dir, exist_ok=True)
    # <video>_<frame>_<yolo cx-cy-w-h>.jpg (shared across every tool).
    save_name = cropnames.yolo_crop_name(video_name or "video", frame_idx,
                                         (x1p, y1p, x2p, y2p), w, h)
    save_path = os.path.join(output_dir, save_name)
    crop = frame[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        return None
    cv2.imwrite(save_path, crop)
    # Crop saves do NOT log to Convex — the database only updates when an image
    # gets annotated (see db_log.record_annotation).
    return save_path


def log_centroid(frame, frame_idx, person_box, source):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, person_box[:4])
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
    if x2 <= x1 or y2 <= y1:
        return

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    record = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source": source,
        "frame_idx": int(frame_idx),
        "centroid_x": cx,
        "centroid_y": cy,
        "frame_width": int(w),
        "frame_height": int(h),
    }
    with open(CENTROID_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def draw_pending_manual_boxes(frame, frame_idx, pending_manual_boxes, pending_manual_boxes_frame):
    if pending_manual_boxes_frame != frame_idx or not pending_manual_boxes:
        return frame
    for box in pending_manual_boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 7, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y1), 7, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 7, (0, 255, 0), -1)
        cv2.circle(frame, (x1, y2), 7, (0, 255, 0), -1)
    cv2.putText(
        frame,
        f"{len(pending_manual_boxes)} manual box(es) selected. Press S to save all.",
        (10, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return frame


def select_manual_boxes_with_handles(base_frame, initial_boxes=None):
    h, w = base_frame.shape[:2]
    handle_radius = 7
    hit_radius = 14

    state = {
        "rects": [],
        "drawing": False,
        "resizing": False,
        "active_box": None,
        "active_corner": None,
        "start": (0, 0),
        "temp_rect": None,
    }

    for box in initial_boxes or []:
        x1, y1, x2, y2 = map(int, box[:4])
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        if x2 > x1 and y2 > y1:
            state["rects"].append([x1, y1, x2, y2])

    def corners(rect):
        x1, y1, x2, y2 = rect
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def nearest_corner(px, py):
        best = None
        best_dist = 10**9
        for b_idx, rect in enumerate(state["rects"]):
            pts = corners(rect)
            for c_idx, (cx, cy) in enumerate(pts):
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                if d2 < best_dist:
                    best_dist = d2
                    best = (b_idx, c_idx)
        if best is not None and best_dist <= hit_radius * hit_radius:
            return best
        return None

    def normalize_rect(xa, ya, xb, yb):
        x1, x2 = sorted((xa, xb))
        y1, y2 = sorted((ya, yb))
        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
        return [x1, y1, x2, y2]

    def on_mouse(event, x, y, flags, param):
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)

        if event == cv2.EVENT_LBUTTONDOWN:
            hit = nearest_corner(x, y)
            if hit is not None:
                b_idx, c_idx = hit
                state["resizing"] = True
                state["active_box"] = b_idx
                state["active_corner"] = c_idx
                return
            state["drawing"] = True
            state["start"] = (x, y)
            state["temp_rect"] = [x, y, x, y]

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["drawing"]:
                sx, sy = state["start"]
                state["temp_rect"] = normalize_rect(sx, sy, x, y)
            elif state["resizing"] and state["active_box"] is not None:
                rect = state["rects"][state["active_box"]]
                x1, y1, x2, y2 = rect
                corner = state["active_corner"]
                if corner == 0:  # top-left
                    x1, y1 = x, y
                elif corner == 1:  # top-right
                    x2, y1 = x, y
                elif corner == 2:  # bottom-right
                    x2, y2 = x, y
                elif corner == 3:  # bottom-left
                    x1, y2 = x, y
                state["rects"][state["active_box"]] = normalize_rect(x1, y1, x2, y2)

        elif event == cv2.EVENT_LBUTTONUP:
            if state["drawing"] and state["temp_rect"] is not None:
                x1, y1, x2, y2 = state["temp_rect"]
                if x2 > x1 and y2 > y1:
                    state["rects"].append([x1, y1, x2, y2])
            state["drawing"] = False
            state["resizing"] = False
            state["active_box"] = None
            state["active_corner"] = None
            state["temp_rect"] = None

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            view = base_frame.copy()
            for rect in state["rects"]:
                x1, y1, x2, y2 = map(int, rect)
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for cx, cy in corners(rect):
                    cv2.circle(view, (int(cx), int(cy)), handle_radius, (0, 255, 0), -1)
            if state["temp_rect"] is not None:
                x1, y1, x2, y2 = map(int, state["temp_rect"])
                cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                view,
                "Draw boxes. S: save as person | B: save as background | Backspace: undo | Esc: cancel",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKeyEx(20)
            key_char = key & 0xFF
            if key_char in (ord("s"), ord("S")) or key in (13, 10, 32):  # S or Enter or Space -> person
                if not state["rects"]:
                    continue
                boxes = [(x1, y1, x2, y2, 1.0) for (x1, y1, x2, y2) in state["rects"]]
                return (boxes, "person")
            if key_char in (ord("b"), ord("B")):  # B -> background
                if not state["rects"]:
                    continue
                boxes = [(x1, y1, x2, y2, 1.0) for (x1, y1, x2, y2) in state["rects"]]
                return (boxes, "background")
            if key in (8, 255):
                if state["rects"]:
                    state["rects"].pop()
            if key in (27, ord("q"), ord("Q")):
                return None
    finally:
        cv2.setMouseCallback(WINDOW_NAME, lambda *args: None)


def select_person_index_in_window(base_frame, person_boxes):
    if not person_boxes:
        return None

    overlay = base_frame.copy()
    for i, box in enumerate(person_boxes, start=1):
        x1, y1, x2, y2, _ = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            overlay,
            str(i),
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        overlay,
        "Press 1-9 to save person, ESC to cancel",
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    cv2.imshow(WINDOW_NAME, overlay)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            return None
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(person_boxes):
                return idx


def overlay_info(frame, frame_idx, total_frames, fps, playing, ppe_conf, person_conf, person_count, ppe_count):
    status = "Playing" if playing else "Paused"
    time_sec = frame_idx / fps if fps > 0 else 0

    text = (
        f"{status} | Frame: {frame_idx}/{total_frames} | Time: {time_sec:.2f}s | "
        f"PersonConf: {person_conf:.2f} | PPEConf: {ppe_conf:.2f} | "
        f"Persons: {person_count} | PPE Boxes: {ppe_count}"
    )
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(
        frame,
        "Space: Play/Pause | A/D: -/+30f | Arrows/J/L: -/+1f | M: Manual infer | G: Draw box | S: Save person | B: Save person to BG | Q: Quit",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
    )
    return frame


def rewrite_video_with_edits(
    source,
    output_path,
    person_model,
    ppe_model,
    person_conf,
    ppe_conf,
    saved_person_masks,
    deleted_frames,
    manual_annotation_boxes,
):
    cap = open_video(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-6 else 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        print(f"Error: Unable to create output video: {output_path}")
        return False

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in deleted_frames:
            frame_idx += 1
            continue

        masked = apply_saved_masks(frame.copy(), frame_idx, saved_person_masks)
        annotated, person_count, ppe_count, _ = run_two_stage_inference(
            masked,
            person_model,
            ppe_model,
            person_conf,
            ppe_conf,
            manual_person_boxes=manual_annotation_boxes.get(frame_idx, []),
        )
        annotated = overlay_info(
            annotated,
            frame_idx,
            total_frames,
            fps,
            True,
            ppe_conf,
            person_conf,
            person_count,
            ppe_count,
        )

        writer.write(annotated)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Rewritten video saved to: {output_path} (frames written: {written})")
    return True


# ----------------------------- Main ----------------------------- #
def main():
    args = parse_args()

    if not (0.0 <= args.conf <= 1.0) or not (0.0 <= args.person_conf <= 1.0):
        print("Error: --conf and --person-conf must be between 0 and 1.")
        sys.exit(1)

    if not os.path.exists(args.ppe_model):
        print(f"PPE model not found: {args.ppe_model}")
        sys.exit(1)

    person_model = load_person_model(args.person_model)
    ppe_model = YOLO(args.ppe_model)
    source = choose_video_source()
    cap = open_video(source)

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-6 else 25.0
    frame_interval_ms = 1000.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    playing = True
    frame_idx = 0
    frame = None
    cached_annotated = None
    cached_counts = (0, 0)
    current_person_boxes = []
    pending_manual_boxes = []
    pending_manual_boxes_frame = -1
    manual_annotation_boxes = {}
    saved_person_masks = {}
    delete_candidate_frames = set()
    deleted_frames = set()
    trackbar_busy = False

    def infer_current_frame(current_frame, current_frame_idx):
        masked = apply_saved_masks(current_frame.copy(), current_frame_idx, saved_person_masks)
        return run_two_stage_inference(
            frame=masked,
            person_model=person_model,
            ppe_model=ppe_model,
            person_conf=args.person_conf,
            ppe_conf=args.conf,
            manual_person_boxes=manual_annotation_boxes.get(current_frame_idx, []),
        )

    def refresh_current_inference():
        nonlocal cached_annotated, cached_counts, current_person_boxes, deleted_frames
        cached_annotated, person_count, ppe_count, current_person_boxes = infer_current_frame(frame, frame_idx)
        cached_counts = (person_count, ppe_count)
        if frame_idx in delete_candidate_frames and person_count == 0:
            deleted_frames.add(frame_idx)
            return False
        return True

    def seek_and_load(target_idx, direction):
        nonlocal frame_idx, frame, pending_manual_boxes, pending_manual_boxes_frame
        pending_manual_boxes = []
        pending_manual_boxes_frame = -1
        if total_frames <= 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target_idx))
            ret, new_frame = cap.read()
            if not ret:
                return False
            frame = new_frame
            frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            return refresh_current_inference()

        idx = min(max(target_idx, 0), total_frames - 1)
        step = 1 if direction >= 0 else -1
        while 0 <= idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, new_frame = cap.read()
            if not ret:
                return False
            frame = new_frame
            frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            if refresh_current_inference():
                return True
            idx += step
        return False

    def on_trackbar(val):
        nonlocal frame_idx, frame, trackbar_busy, playing
        if trackbar_busy:
            return
        frame_idx = val
        playing = False
        if not seek_and_load(frame_idx, 1):
            seek_and_load(frame_idx - 1, -1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if total_frames > 0:
        cv2.createTrackbar("Seek", WINDOW_NAME, 0, total_frames - 1, on_trackbar)

    while True:
        loop_start = time.perf_counter()
        if playing:
            loaded = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                if refresh_current_inference():
                    loaded = True
                    break
            if not loaded:
                break

        if frame is None:
            continue

        if cached_annotated is None:
            if not refresh_current_inference():
                if not seek_and_load(frame_idx + 1, 1):
                    break
                continue

        annotated = overlay_info(
            cached_annotated.copy(),
            frame_idx,
            total_frames,
            fps,
            playing,
            args.conf,
            args.person_conf,
            cached_counts[0],
            cached_counts[1],
        )
        annotated = draw_pending_manual_boxes(
            annotated,
            frame_idx,
            pending_manual_boxes,
            pending_manual_boxes_frame,
        )

        if total_frames > 0:
            trackbar_busy = True
            cv2.setTrackbarPos("Seek", WINDOW_NAME, min(max(frame_idx, 0), total_frames - 1))
            trackbar_busy = False

        cv2.imshow(WINDOW_NAME, annotated)
        if playing:
            elapsed_ms = (time.perf_counter() - loop_start) * 1000.0
            wait_ms = max(1, int(frame_interval_ms - elapsed_ms))
        else:
            wait_ms = 30
        key = cv2.waitKeyEx(wait_ms)
        key_char = key & 0xFF
        left_arrow_keys = {2424832, 81}
        right_arrow_keys = {2555904, 83}

        if key_char == ord(" "):
            playing = not playing
        elif key_char in (ord("q"), ord("Q"), 27):
            break
        elif key_char in (ord("g"), ord("G")):
            playing = False
            if frame is None:
                continue
            initial_boxes = []
            if pending_manual_boxes and pending_manual_boxes_frame == frame_idx:
                initial_boxes = pending_manual_boxes
            result = select_manual_boxes_with_handles(cached_annotated.copy(), initial_boxes=initial_boxes)
            if result is not None:
                new_boxes, dest = result
                output_dir = PERSON_CROPS_DIR if dest == "person" else BACKGROUND_CROPS_DIR
                saved_count = 0
                for manual_idx, manual_box in enumerate(new_boxes):
                    save_path = save_person_crop(frame, frame_idx, manual_idx, manual_box, output_dir=output_dir, video_name=source)
                    if save_path is None:
                        continue
                    if dest == "person":
                        x1, y1, x2, y2 = map(int, manual_box[:4])
                        saved_person_masks.setdefault(frame_idx, []).append((x1, y1, x2, y2))
                        log_centroid(frame, frame_idx, manual_box, source)
                        print(f"Saved person crop: {save_path}")
                    else:
                        print(f"Saved background crop: {save_path}")
                    saved_count += 1

                if saved_count == 0:
                    print("Could not save any manual crop from G selection.")
                    continue

                if not refresh_current_inference():
                    if not seek_and_load(frame_idx + 1, 1):
                        if not seek_and_load(frame_idx - 1, -1):
                            break
            else:
                print("Manual box cancelled.")
        elif key_char in (ord("m"), ord("M")):
            playing = False
            if frame is None:
                continue
            existing_boxes = manual_annotation_boxes.get(frame_idx, [])
            result = select_manual_boxes_with_handles(cached_annotated.copy(), initial_boxes=existing_boxes)
            if result is not None:
                new_boxes, _ = result
                manual_annotation_boxes[frame_idx] = new_boxes
                if not refresh_current_inference():
                    if not seek_and_load(frame_idx + 1, 1):
                        if not seek_and_load(frame_idx - 1, -1):
                            break
            else:
                print("Manual inference box selection cancelled.")
        elif key_char in (ord("s"), ord("S")):
            playing = False
            selected_box = None
            selected_idx = 0
            is_manual_selection = False

            if pending_manual_boxes and pending_manual_boxes_frame == frame_idx:
                is_manual_selection = True
                saved_count = 0
                for manual_idx, manual_box in enumerate(pending_manual_boxes):
                    save_path = save_person_crop(frame, frame_idx, manual_idx, manual_box, video_name=source)
                    if save_path is None:
                        continue
                    x1, y1, x2, y2 = map(int, manual_box[:4])
                    saved_person_masks.setdefault(frame_idx, []).append((x1, y1, x2, y2))
                    log_centroid(frame, frame_idx, manual_box, source)
                    saved_count += 1
                    print(f"Saved person crop: {save_path}")
                pending_manual_boxes = []
                pending_manual_boxes_frame = -1
                if saved_count == 0:
                    print("Could not save any manual crop.")
                    continue
            else:
                if not current_person_boxes:
                    print("No visible person detected in current frame. Press G to draw a manual box.")
                    continue
                selected_idx = select_person_index_in_window(cached_annotated.copy(), current_person_boxes)
                if selected_idx is None:
                    continue
                selected_box = current_person_boxes[selected_idx]

                if not is_manual_selection:
                    save_path = save_person_crop(frame, frame_idx, selected_idx, selected_box, video_name=source)
                    if save_path is None:
                        print("Could not save person crop for this selection.")
                        continue
                    x1, y1, x2, y2 = map(int, selected_box[:4])
                    saved_person_masks.setdefault(frame_idx, []).append((x1, y1, x2, y2))
                    delete_candidate_frames.add(frame_idx)
                    log_centroid(frame, frame_idx, selected_box, source)
                    print(f"Saved person crop: {save_path}")

            if not refresh_current_inference():
                if not seek_and_load(frame_idx + 1, 1):
                    if not seek_and_load(frame_idx - 1, -1):
                        break
        elif key_char in (ord("b"), ord("B")):
            playing = False
            if frame is None:
                continue
            if pending_manual_boxes and pending_manual_boxes_frame == frame_idx:
                saved_count = 0
                for manual_idx, manual_box in enumerate(pending_manual_boxes):
                    save_path = save_person_crop(frame, frame_idx, manual_idx, manual_box, output_dir=BACKGROUND_CROPS_DIR, video_name=source)
                    if save_path:
                        saved_count += 1
                        print(f"Saved background crop: {save_path}")
                pending_manual_boxes = []
                pending_manual_boxes_frame = -1
                if saved_count == 0:
                    print("Could not save any manual crop to background.")
            else:
                if not current_person_boxes:
                    print("No visible person detected in current frame. Press G to draw a manual box.")
                    continue
                selected_idx = select_person_index_in_window(cached_annotated.copy(), current_person_boxes)
                if selected_idx is None:
                    continue
                selected_box = current_person_boxes[selected_idx]
                save_path = save_person_crop(frame, frame_idx, selected_idx, selected_box, output_dir=BACKGROUND_CROPS_DIR, video_name=source)
                if save_path:
                    print(f"Saved background crop: {save_path}")
                else:
                    print("Could not save background crop.")
        elif key_char == ord("a"):
            frame_idx = max(0, frame_idx - 30)
            seek_and_load(frame_idx, -1)
        elif key_char == ord("d"):
            if total_frames > 0:
                frame_idx = min(total_frames - 1, frame_idx + 30)
            else:
                frame_idx = frame_idx + 30
            seek_and_load(frame_idx, 1)
        elif key_char == ord("j") or key in left_arrow_keys:
            frame_idx = max(0, frame_idx - 1)
            seek_and_load(frame_idx, -1)
        elif key_char == ord("l") or key in right_arrow_keys:
            if total_frames > 0:
                frame_idx = min(total_frames - 1, frame_idx + 1)
            else:
                frame_idx = frame_idx + 1
            seek_and_load(frame_idx, 1)

    cap.release()
    cv2.destroyAllWindows()

    output_path = args.output.strip()
    if deleted_frames and not output_path:
        source_root, _ = os.path.splitext(source)
        output_path = f"{source_root}_edited.mp4"

    if output_path:
        rewrite_video_with_edits(
            source=source,
            output_path=output_path,
            person_model=person_model,
            ppe_model=ppe_model,
            person_conf=args.person_conf,
            ppe_conf=args.conf,
            saved_person_masks=saved_person_masks,
            deleted_frames=deleted_frames,
            manual_annotation_boxes=manual_annotation_boxes,
        )


if __name__ == "__main__":
    main()