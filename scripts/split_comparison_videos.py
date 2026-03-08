"""
Split comparison videos into individual view videos.

Each comparison video is 5 panels stitched horizontally (equal width):
  Panel 1: human input  -> human/
  Panel 2: (discard)
  Panel 3: front view   -> robot/
  Panel 4: back view    -> robot/
  Panel 5: wrist view   -> robot/

Output structure follows dataset convention:
  dataset/<task_name>/human/ep{N}_front.mp4
  dataset/<task_name>/robot/ep{N}_front.mp4
  dataset/<task_name>/robot/ep{N}_back.mp4
  dataset/<task_name>/robot/ep{N}_wrist.mp4
"""

import os
import glob
import re
import cv2
from pathlib import Path


SRC_DIR = "/home/guian/data/Mitty/comparison_results_02_25_download/comparison_results_02_25"
DST_DIR = "/home/guian/data/Mitty/dataset/comparison_results_02_25"

# Panel index -> (subfolder, suffix)
PANELS = {
    0: ("human", "front"),    # panel 1: human input
    # 1: discard
    2: ("robot", "front"),    # panel 3: front view
    3: ("robot", "back"),     # panel 4: back view
    4: ("robot", "wrist"),    # panel 5: wrist view
}


def split_video(src_path, ep_num):
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {src_path}")
        return

    total_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    panel_w = total_w // 5

    # Create writers for each panel we want to keep
    writers = {}
    for panel_idx, (subfolder, suffix) in PANELS.items():
        out_dir = os.path.join(DST_DIR, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ep{ep_num}_{suffix}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (panel_w, h))
        writers[panel_idx] = writer

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for panel_idx, writer in writers.items():
            x_start = panel_idx * panel_w
            x_end = x_start + panel_w
            panel_frame = frame[:, x_start:x_end]
            writer.write(panel_frame)
        frame_count += 1

    cap.release()
    for writer in writers.values():
        writer.release()

    print(f"  Done: {frame_count} frames split into {panel_w}x{h} panels")


def main():
    video_files = sorted(glob.glob(os.path.join(SRC_DIR, "comparison_*.mp4")))
    print(f"Found {len(video_files)} videos to process")

    for vf in video_files:
        basename = os.path.basename(vf)
        match = re.search(r"comparison_(\d+)\.mp4", basename)
        if not match:
            print(f"Skipping {basename}: cannot parse episode number")
            continue
        ep_num = int(match.group(1))
        print(f"Processing {basename} -> ep{ep_num}")
        split_video(vf, ep_num)

    # Summary
    human_count = len(glob.glob(os.path.join(DST_DIR, "human", "*.mp4")))
    robot_count = len(glob.glob(os.path.join(DST_DIR, "robot", "*.mp4")))
    print(f"\nDone! Output: {DST_DIR}")
    print(f"  human/: {human_count} videos")
    print(f"  robot/: {robot_count} videos")


if __name__ == "__main__":
    main()
