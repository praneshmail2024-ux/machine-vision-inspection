"""Extract a contact sheet from a recording for detector troubleshooting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ring_detector import RingDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "contact_sheets" / "recording_contact_sheet.jpg",
    )
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--extract-dir", type=Path)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--replay-output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "replays" / "recording_replay.mp4",
    )
    parser.add_argument("--spacetime-output", type=Path)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = np.linspace(0, max(0, frame_count - 1), args.samples, dtype=int)
    thumbnails: list[np.ndarray] = []
    for frame_index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        thumb_width = 480
        thumb_height = round(frame.shape[0] * thumb_width / frame.shape[1])
        thumb = cv2.resize(frame, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)
        seconds = frame_index / max(fps, 0.001)
        cv2.putText(
            thumb,
            f"{seconds:.1f}s / frame {frame_index}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
        if args.extract_dir is not None:
            args.extract_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.extract_dir / f"frame_{frame_index:04d}.jpg"), frame)
    capture.release()

    if not thumbnails:
        raise SystemExit("The video contained no readable frames.")
    columns = 3
    rows = int(np.ceil(len(thumbnails) / columns))
    while len(thumbnails) < rows * columns:
        thumbnails.append(np.zeros_like(thumbnails[0]))
    sheet = np.vstack([
        np.hstack(thumbnails[row * columns : (row + 1) * columns])
        for row in range(rows)
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), sheet):
        raise SystemExit(f"Could not write {args.output}")
    print({
        "frames": frame_count,
        "fps": round(fps, 3),
        "duration_seconds": round(frame_count / max(fps, 0.001), 3),
        "resolution": f"{width}x{height}",
        "contact_sheet": str(args.output.resolve()),
    })

    if args.spacetime_output is not None:
        args.spacetime_output.parent.mkdir(parents=True, exist_ok=True)
        timeline_capture = cv2.VideoCapture(str(args.video))
        profiles: list[np.ndarray] = []
        while True:
            ok, screen = timeline_capture.read()
            if not ok:
                break
            gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            # Belt lane around the ring sides; high percentile preserves thin rims.
            lane = gray_screen[22:692, 245:430]
            profiles.append(np.percentile(lane, 82, axis=1).astype(np.uint8))
        timeline_capture.release()
        timeline = np.stack(profiles, axis=1)
        timeline = cv2.normalize(timeline, None, 0, 255, cv2.NORM_MINMAX)
        timeline = cv2.resize(timeline, (max(884, timeline.shape[1] * 2), 670), interpolation=cv2.INTER_NEAREST)
        cv2.line(timeline, (0, 336), (timeline.shape[1] - 1, 336), 255, 1)
        cv2.imwrite(str(args.spacetime_output), timeline)
        print({"spacetime": str(args.spacetime_output.resolve())})

    if args.replay:
        # ScreenSketch recording coordinates of the 4:3 camera panel. The cyan
        # rectangle in this recording covers x=15..60% of the camera image.
        crop_x, crop_y, crop_w, crop_h = 23, 22, 890, 670
        replay = cv2.VideoCapture(str(args.video))
        replay.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 40))
        ok, empty_screen = replay.read()
        if not ok:
            raise SystemExit("Could not read an empty-belt calibration frame.")

        def camera_crop(screen: np.ndarray) -> np.ndarray:
            crop = screen[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
            return cv2.resize(crop, (640, 480), interpolation=cv2.INTER_AREA)

        detector = RingDetector(roi_left=0.15, roi_top=0.03, roi_right=0.60, roi_bottom=0.97)
        detector.calibrate_background(camera_crop(empty_screen))
        replay.set(cv2.CAP_PROP_POS_FRAMES, 0)
        args.replay_output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.replay_output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (640, 480),
        )
        max_visible = 0
        processed = 0
        gate_events: list[dict[str, object]] = []
        previous_total = 0
        while True:
            ok, screen = replay.read()
            if not ok:
                break
            annotated, status = detector.process(camera_crop(screen))
            writer.write(annotated)
            max_visible = max(max_visible, int(status["visible"]))
            if detector.total_count != previous_total:
                gate_events.append({
                    "frame": processed,
                    "seconds": round(processed / max(fps, 0.001), 2),
                    "centers": [detection.center for detection in detector.detect(camera_crop(screen))[0]],
                })
                previous_total = detector.total_count
            processed += 1
        writer.release()
        replay.release()
        print({
            "replay_frames": processed,
            "maximum_simultaneous_detections": max_visible,
            "replayed_total_count": detector.total_count,
            "track_crossing_count": detector.track_crossing_count,
            "gate_events": gate_events,
            "replay_video": str(args.replay_output.resolve()),
        })


if __name__ == "__main__":
    main()
