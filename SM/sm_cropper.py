#!/usr/bin/env python3
"""
SM video viewer + frame copier.

Features kept:
1) Navigate video in OpenCV UI.
2) Run SM detection with confidence 0.7 by default.
3) Press C to copy current frame to an output folder.
4) Output folder is selected once and remembered for future runs.
"""

import argparse
import json
import os
import sys
import time

import cv2
from ultralytics import YOLO


# Repo root on the path so this stays runnable standalone (python3 SM/sm_cropper.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import paths

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Weights come from the suite-wide model pool, shared with every other tool.
DEFAULT_SINGLE_MODEL_PATH = str(
    paths.MODELS_DIR / "single-model-pt" / "single-model-pt.pt"
)
VIDEO_HISTORY_FILE = str(paths.STATE_DIR / "sm_saved_video_paths.json")
COPY_PATH_HISTORY_FILE = str(paths.STATE_DIR / "sm_cropper_paths.json")
WINDOW_NAME = "SM Cropper"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def parse_args():
    parser = argparse.ArgumentParser(description="SM video viewer + frame copier")
    parser.add_argument("--model", type=str, default=DEFAULT_SINGLE_MODEL_PATH, help="Path to SM .pt model")
    parser.add_argument("--conf", type=float, default=0.7, help="Detection confidence threshold (default: 0.7)")
    return parser.parse_args()


def load_list_history(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_list_history(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def list_videos_in_folder(folder_path):
    videos = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if os.path.isfile(full_path) and name.lower().endswith(VIDEO_EXTENSIONS):
            videos.append(full_path)
    videos.sort()
    return videos


def choose_folder_from_history(history_file, prompt_name, allow_clear=False, create_if_missing=False):
    history = load_list_history(history_file)

    while True:
        print(f"\nSelect {prompt_name} folder path:")
        for idx, src in enumerate(history, start=1):
            print(f"{idx}. {src}")
        print("N. Enter new folder")
        if allow_clear:
            print("C. Clear saved folders")

        choice = input("Choice: ").strip()

        if allow_clear and choice.lower() == "c":
            history = []
            save_list_history(history_file, history)
            print("Saved folder history cleared.")
            continue

        if choice.lower() == "n" or not choice.isdigit():
            new_src = input(f"Enter {prompt_name} folder path: ").strip()
            new_src = os.path.abspath(os.path.expanduser(new_src))

            if create_if_missing:
                os.makedirs(new_src, exist_ok=True)
            elif not os.path.isdir(new_src):
                print("Error: Folder does not exist.")
                continue

            if new_src not in history:
                history.append(new_src)
                save_list_history(history_file, history)
            return new_src

        idx = int(choice) - 1
        if 0 <= idx < len(history):
            selected = os.path.abspath(os.path.expanduser(history[idx]))
            if create_if_missing:
                os.makedirs(selected, exist_ok=True)
            elif not os.path.isdir(selected):
                print("Saved folder no longer exists. Please choose another folder.")
                continue
            return selected

        print("Invalid selection.")


def choose_video_source():
    while True:
        folder = choose_folder_from_history(
            VIDEO_HISTORY_FILE,
            prompt_name="video",
            allow_clear=True,
            create_if_missing=False,
        )
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
            return videos[idx]
        print("Invalid selection.")


def open_video(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        sys.exit(1)
    return cap


def draw_detections(frame, model, conf):
    result = model.predict(frame, conf=conf, verbose=False)[0]
    if result.boxes is None:
        return frame, 0

    count = 0
    for box in result.boxes:
        dconf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = result.names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {dconf:.2f}",
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )
        count += 1
    return frame, count


def save_full_frame(frame, frame_idx, video_source, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_source))[0]
    save_name = f"{base}_frame_{frame_idx:06d}_{int(time.time() * 1000)}.jpg"
    save_path = os.path.join(output_dir, save_name)
    cv2.imwrite(save_path, frame)
    return save_path


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}")
        sys.exit(1)
    if not (0.0 <= args.conf <= 1.0):
        print("--conf must be between 0 and 1")
        sys.exit(1)

    model = YOLO(args.model)
    source = choose_video_source()
    copy_output_dir = choose_folder_from_history(
        COPY_PATH_HISTORY_FILE,
        prompt_name="output copy",
        allow_clear=True,
        create_if_missing=True,
    )

    cap = open_video(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1e-6 else 25.0
    frame_interval_ms = 1000.0 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    playing = True
    frame_idx = 0
    frame = None
    annotated = None
    det_count = 0
    trackbar_busy = False

    def refresh_from_current_frame():
        nonlocal annotated, det_count
        if frame is None:
            return
        annotated, det_count = draw_detections(frame.copy(), model, args.conf)

    def seek_and_load(target_idx):
        nonlocal frame_idx, frame
        if total_frames > 0:
            target_idx = min(max(target_idx, 0), total_frames - 1)
        else:
            target_idx = max(target_idx, 0)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ok, new_frame = cap.read()
        if not ok:
            return False
        frame = new_frame
        frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        refresh_from_current_frame()
        return True

    def on_trackbar(val):
        nonlocal frame_idx, playing
        if trackbar_busy:
            return
        playing = False
        frame_idx = val
        seek_and_load(frame_idx)

    def prompt_and_jump_to_frame():
        nonlocal playing
        playing = False
        typed = ""
        while True:
            if annotated is None:
                return

            popup = annotated.copy()
            h, w = popup.shape[:2]
            box_w = min(700, w - 40)
            box_h = 140
            x1 = max(20, (w - box_w) // 2)
            y1 = max(20, (h - box_h) // 2)
            x2 = x1 + box_w
            y2 = y1 + box_h

            # Dim background for popup effect
            dim = popup.copy()
            cv2.rectangle(dim, (0, 0), (w, h), (0, 0, 0), -1)
            popup = cv2.addWeighted(popup, 0.45, dim, 0.55, 0)

            cv2.rectangle(popup, (x1, y1), (x2, y2), (40, 40, 40), -1)
            cv2.rectangle(popup, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                popup,
                "Jump to frame:",
                (x1 + 20, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                popup,
                typed if typed else "_",
                (x1 + 20, y1 + 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.95,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                popup,
                "Enter: jump | Backspace: delete | Esc: cancel",
                (x1 + 20, y1 + 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
            )

            cv2.imshow(WINDOW_NAME, popup)
            k = cv2.waitKeyEx(30)
            kc = k & 0xFF

            if k in (13, 10):  # Enter
                if not typed:
                    return
                try:
                    target = int(typed)
                except ValueError:
                    return
                seek_and_load(target)
                return
            if kc in (27,):  # Esc
                return
            if k in (8, 255):  # Backspace
                typed = typed[:-1]
                continue
            if ord("0") <= kc <= ord("9"):
                typed += chr(kc)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if total_frames > 0:
        cv2.createTrackbar("Seek", WINDOW_NAME, 0, total_frames - 1, on_trackbar)

    while True:
        loop_start = time.perf_counter()

        if playing:
            ok, new_frame = cap.read()
            if not ok:
                break
            frame = new_frame
            frame_idx = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            refresh_from_current_frame()

        if frame is None or annotated is None:
            if not seek_and_load(frame_idx):
                break

        view = annotated.copy()
        status = "Playing" if playing else "Paused"
        time_sec = frame_idx / fps if fps > 0 else 0.0
        cv2.putText(
            view,
            f"{status} | Frame: {frame_idx}/{total_frames} | Time: {time_sec:.2f}s | Conf: {args.conf:.2f} | Detections: {det_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            view,
            "Space: Play/Pause | A/D: -/+30f | Arrows/J/L: -/+1f | F: Jump frame | C: Copy frame | Q: Quit",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
        )

        if total_frames > 0:
            trackbar_busy = True
            cv2.setTrackbarPos("Seek", WINDOW_NAME, min(max(frame_idx, 0), total_frames - 1))
            trackbar_busy = False

        cv2.imshow(WINDOW_NAME, view)

        wait_ms = max(1, int(frame_interval_ms - (time.perf_counter() - loop_start) * 1000.0)) if playing else 30
        key = cv2.waitKeyEx(wait_ms)
        key_char = key & 0xFF
        left_arrow_keys = {2424832, 81}
        right_arrow_keys = {2555904, 83}

        if key_char == ord(" "):
            playing = not playing
        elif key_char in (ord("q"), ord("Q"), 27):
            break
        elif key_char in (ord("c"), ord("C")):
            if frame is not None:
                save_path = save_full_frame(frame, frame_idx, source, copy_output_dir)
                print(f"Copied frame to: {save_path}")
        elif key_char in (ord("f"), ord("F")):
            prompt_and_jump_to_frame()
        elif key_char == ord("a"):
            playing = False
            seek_and_load(frame_idx - 30)
        elif key_char == ord("d"):
            playing = False
            seek_and_load(frame_idx + 30)
        elif key_char == ord("j") or key in left_arrow_keys:
            playing = False
            seek_and_load(frame_idx - 1)
        elif key_char == ord("l") or key in right_arrow_keys:
            playing = False
            seek_and_load(frame_idx + 1)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
