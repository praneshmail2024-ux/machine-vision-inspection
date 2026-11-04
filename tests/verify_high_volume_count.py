"""Stress the production counter with repeated dense ten-ring batches."""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ring_detector import RingDetector


def dense_ten_ring_batch(shift_y: int) -> np.ndarray:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cardboard = (125, 175, 215)
    for y in (550, 625, 700, 775, 850):
        for x in (285, 355):
            center = (x, y + shift_y)
            cv2.ellipse(
                frame, center, (48, 42), 0, 0, 360, cardboard, -1
            )
            cv2.ellipse(
                frame, center, (33, 27), 0, 0, 360, (12, 12, 12), -1
            )
    return frame


def main() -> None:
    empty = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = RingDetector()
    detector.calibrate_background(empty)

    checkpoints: list[int] = []
    status: dict[str, int | float] | None = None
    for batch_index in range(5):
        for shift_y in range(0, -1001, -10):
            _, status = detector.process(dense_ten_ring_batch(shift_y))
        # Expire the previous batch's tracks before the next batch enters.
        for _ in range(detector.max_missing_frames + 6):
            _, status = detector.process(empty)
        checkpoints.append(int(status["total"]))

    print(
        {
            "expected_checkpoints": [10, 20, 30, 40, 50],
            "actual_checkpoints": checkpoints,
            "final_visible": int(status["visible"]) if status else -1,
        }
    )
    if checkpoints != [10, 20, 30, 40, 50]:
        raise SystemExit(1)
    if status is None or status["visible"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
