import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import _calibration_frame_contains_rings


def main() -> None:
    settings = (0.14, 0.03, 0.62, 0.97, 0.50)
    empty_belt = np.full((480, 640, 3), (45, 45, 45), dtype=np.uint8)

    belt_join = empty_belt.copy()
    cv2.rectangle(belt_join, (90, 230), (397, 244), (68, 68, 68), -1)

    occupied_belt = empty_belt.copy()
    cardboard = (125, 175, 215)
    cv2.ellipse(occupied_belt, (260, 350), (52, 42), 0, 0, 360, cardboard, -1)
    cv2.ellipse(occupied_belt, (260, 350), (34, 25), 0, 0, 360, (20, 20, 20), -1)

    results = {
        "empty_belt_rejected": _calibration_frame_contains_rings(
            empty_belt, settings
        ),
        "belt_join_rejected": _calibration_frame_contains_rings(
            belt_join, settings
        ),
        "ring_frame_rejected": _calibration_frame_contains_rings(
            occupied_belt, settings
        ),
    }
    print(results)
    if results != {
        "empty_belt_rejected": False,
        "belt_join_rejected": False,
        "ring_frame_rejected": True,
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
